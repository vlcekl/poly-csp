from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import openmm as mm
from openmm import unit

from poly_csp.mm.anneal import run_temperature_ramp


def _backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    idx: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def _resolve_amber_paths(amber_summary: Dict[str, object]) -> tuple[Path, Path]:
    if not bool(amber_summary.get("parameterized", False)):
        raise RuntimeError(
            "Parameterized relaxation requires AmberTools parameterized artifacts "
            "(amber_summary.parameterized=true)."
        )
    if str(amber_summary.get("parameter_backend", "")).strip().lower() not in (
        "ambertools", "residue_aware",
    ):
        raise RuntimeError(
            "Parameterized relaxation requires amber_summary.parameter_backend='ambertools'."
        )
    files = amber_summary.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Amber summary is missing artifact file paths.")
    prmtop = files.get("prmtop")
    inpcrd = files.get("inpcrd")
    if not isinstance(prmtop, str) or not isinstance(inpcrd, str):
        raise RuntimeError("Amber summary must include 'prmtop' and 'inpcrd' paths.")
    prmtop_path = Path(prmtop)
    inpcrd_path = Path(inpcrd)
    if not prmtop_path.exists() or not inpcrd_path.exists():
        raise RuntimeError(
            "Parameterized relaxation could not find Amber artifacts: "
            f"{prmtop_path} / {inpcrd_path}"
        )
    return prmtop_path, inpcrd_path


def _reference_positions_nm_from_mol(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must contain a conformer for restrained relaxation.")
    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    return xyz_A / 10.0


def _add_backbone_positional_restraints(
    system: mm.System,
    mol: Chem.Mol,
    k_kj_per_mol_nm2: float,
) -> mm.CustomExternalForce:
    ref_nm = _reference_positions_nm_from_mol(mol)
    force = mm.CustomExternalForce("0.5*k_pos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k_pos", float(k_kj_per_mol_nm2))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for idx in _backbone_heavy_indices(mol):
        x0, y0, z0 = ref_nm[int(idx)]
        force.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(force)
    return force


def _update_rdkit_coords(mol: Chem.Mol, positions_nm: unit.Quantity) -> Chem.Mol:
    xyz_A = np.asarray(positions_nm.value_in_unit(unit.nanometer), dtype=float) * 10.0
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz_A):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def run_parameterized_relaxation(
    mol: Chem.Mol,
    amber_summary: Dict[str, object],
    positional_k: float,
    n_stages: int,
    max_iterations: int,
    anneal_enabled: bool,
    t_start_K: float,
    t_end_K: float,
    anneal_steps: int,
) -> tuple[Chem.Mol, Dict[str, object]]:
    try:
        import parmed as pmd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "AmberTools-parameterized relaxation requires ParmEd. "
            "Install 'parmed' in the current environment."
        ) from exc

    prmtop_path, inpcrd_path = _resolve_amber_paths(amber_summary)
    structure = pmd.load_file(str(prmtop_path), str(inpcrd_path))

    # Detect periodic box vectors from the AMBER summary.
    box_A = amber_summary.get("box_vectors_A")
    is_periodic = bool(amber_summary.get("periodic", False)) and box_A is not None

    if is_periodic:
        from openmm import app as mmapp, Vec3
        system = structure.createSystem(
            nonbondedMethod=mmapp.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
        )
        Lx, Ly, Lz = [float(v) / 10.0 for v in box_A]  # Å → nm
        system.setDefaultPeriodicBoxVectors(
            Vec3(Lx, 0.0, 0.0) * unit.nanometer,
            Vec3(0.0, Ly, 0.0) * unit.nanometer,
            Vec3(0.0, 0.0, Lz) * unit.nanometer,
        )
    else:
        system = structure.createSystem(nonbondedMethod=mm.NoCutoff, constraints=None)

    if structure.positions is None:
        raise RuntimeError("Amber structure does not contain positions.")

    positions = structure.positions
    if int(system.getNumParticles()) != mol.GetNumAtoms():
        raise RuntimeError(
            "Atom count mismatch between RDKit molecule and Amber artifacts: "
            f"{mol.GetNumAtoms()} vs {system.getNumParticles()}."
        )

    pos_force = _add_backbone_positional_restraints(
        system=system,
        mol=mol,
        k_kj_per_mol_nm2=float(positional_k),
    )

    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context = mm.Context(system, integrator)
    context.setPositions(positions)

    stage_factors = np.linspace(1.0, 0.15, max(1, int(n_stages)))
    stage_energies: list[float] = []
    for factor in stage_factors:
        context.setParameter("k_pos", float(positional_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    if bool(anneal_enabled) and int(anneal_steps) > 0:
        run_temperature_ramp(
            context=context,
            integrator=integrator,
            t_start_K=float(t_start_K),
            t_end_K=float(t_end_K),
            n_steps=int(anneal_steps),
            n_segments=10,
        )
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    final_state = context.getState(getPositions=True, getEnergy=True)
    out = _update_rdkit_coords(mol, final_state.getPositions(asNumpy=True))
    summary: Dict[str, object] = {
        "enabled": True,
        "force_model": "ambertools_parameterized",
        "n_stages": int(n_stages),
        "stage_energies_kj_mol": stage_energies,
        "anneal_enabled": bool(anneal_enabled),
    }
    return out, summary
