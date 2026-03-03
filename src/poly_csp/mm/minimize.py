from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

import openmm as mm
from openmm import unit

from poly_csp.chemistry.selectors import SelectorTemplate
from poly_csp.geometry.dihedrals import measure_dihedral_rad
from poly_csp.mm.anneal import run_heat_cool_cycle, run_temperature_ramp
from poly_csp.mm.openmm_system import build_relaxation_system, _sigma_nm
from poly_csp.mm.restraints import (
    add_dihedral_restraints,
    add_hbond_distance_restraints,
    add_positional_restraints,
)


@dataclass(frozen=True)
class RelaxSpec:
    enabled: bool
    positional_k: float
    dihedral_k: float
    hbond_k: float
    mode: Literal[
        "geometry_pre_relax", "ambertools_parameterized", "hybrid_pre_relax"
    ] = "geometry_pre_relax"
    n_stages: int = 3
    max_iterations: int = 200
    anneal_enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    anneal_steps: int = 2000
    anneal_cool_down: bool = True


def _backbone_heavy_indices(mol: Chem.Mol) -> list[int]:
    idx: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def _selector_mappings(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    mappings: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        mappings.setdefault(inst, {})[local] = atom.GetIdx()
    return mappings


def _selector_dihedral_targets(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
) -> list[tuple[int, int, int, int, float]]:
    if selector is None:
        return []
    if mol.GetNumConformers() == 0:
        return []
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    mappings = _selector_mappings(mol)
    out: list[tuple[int, int, int, int, float]] = []
    for mapping in mappings.values():
        for _, (a_l, b_l, c_l, d_l) in selector.dihedrals.items():
            if selector.attach_dummy_idx is not None and selector.attach_dummy_idx in {
                a_l,
                b_l,
                c_l,
                d_l,
            }:
                continue
            if any(local not in mapping for local in (a_l, b_l, c_l, d_l)):
                continue
            a, b, c, d = mapping[a_l], mapping[b_l], mapping[c_l], mapping[d_l]
            theta0 = float(measure_dihedral_rad(xyz, a, b, c, d))
            out.append((a, b, c, d, theta0))
    return out


def _hbond_pairs(
    mol: Chem.Mol,
    selector: SelectorTemplate | None,
    max_dist_A: float = 3.3,
) -> list[tuple[int, int, float]]:
    if selector is None:
        return []
    if mol.GetNumConformers() == 0:
        return []
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    donors: list[tuple[int, int]] = []
    acceptors: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        if local in selector.donors:
            donors.append((inst, atom.GetIdx()))
        if local in selector.acceptors:
            acceptors.append((inst, atom.GetIdx()))

    pairs: list[tuple[int, int, float]] = []
    max_dist_nm = float(max_dist_A / 10.0)
    for d_inst, d_idx in donors:
        for a_inst, a_idx in acceptors:
            if d_inst == a_inst:
                continue
            dist_nm = float(np.linalg.norm(xyz[d_idx] - xyz[a_idx]) / 10.0)
            if dist_nm <= max_dist_nm:
                pairs.append((d_idx, a_idx, dist_nm))
    return pairs


def _update_rdkit_coords(mol: Chem.Mol, positions_nm: unit.Quantity) -> Chem.Mol:
    xyz_A = positions_nm.value_in_unit(unit.nanometer) * 10.0
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz_A):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def run_staged_relaxation(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    amber_artifacts: Dict[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    if not spec.enabled:
        return Chem.Mol(mol), {"enabled": False}

    if spec.mode == "geometry_pre_relax":
        return _run_geometry_pre_relax(mol=mol, spec=spec, selector=selector)
    if spec.mode == "hybrid_pre_relax":
        if amber_artifacts is None:
            raise RuntimeError(
                "hybrid_pre_relax requires Amber artifact metadata "
                "(amber.enabled=true with a parameterized backend)."
            )
        return _run_hybrid_pre_relax(
            mol=mol, spec=spec, selector=selector, amber_artifacts=amber_artifacts,
        )
    if spec.mode == "ambertools_parameterized":
        if amber_artifacts is None:
            raise RuntimeError(
                "ambertools_parameterized relaxation requires Amber artifact metadata."
            )
        from poly_csp.mm.parameterized_relax import run_parameterized_relaxation

        return run_parameterized_relaxation(
            mol=mol,
            amber_summary=amber_artifacts,
            positional_k=float(spec.positional_k),
            n_stages=int(spec.n_stages),
            max_iterations=int(spec.max_iterations),
            anneal_enabled=bool(spec.anneal_enabled),
            t_start_K=float(spec.t_start_K),
            t_end_K=float(spec.t_end_K),
            anneal_steps=int(spec.anneal_steps),
        )
    raise ValueError(f"Unsupported relaxation mode {spec.mode!r}")


def _run_geometry_pre_relax(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    built = build_relaxation_system(mol)
    system = built.system
    positions_nm = built.positions_nm

    pos_force = add_positional_restraints(
        system=system,
        atom_indices=_backbone_heavy_indices(mol),
        reference_positions_nm=positions_nm,
        k_kj_per_mol_nm2=float(spec.positional_k),
    )
    tors_force = add_dihedral_restraints(
        system=system,
        dihedrals=_selector_dihedral_targets(mol, selector),
        k_kj_per_mol=float(spec.dihedral_k),
    )
    hb_force = add_hbond_distance_restraints(
        system=system,
        pairs=_hbond_pairs(mol, selector),
        k_kj_per_mol_nm2=float(spec.hbond_k),
    )

    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context = mm.Context(system, integrator)
    context.setPositions(positions_nm)

    stage_factors = np.linspace(1.0, 0.15, max(1, int(spec.n_stages)))
    stage_energies: list[float] = []
    for factor in stage_factors:
        context.setParameter("k_pos", float(spec.positional_k) * float(factor))
        context.setParameter("k_tors", float(spec.dihedral_k) * float(factor))
        context.setParameter("k_hb", float(spec.hbond_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        run_temperature_ramp(
            context=context,
            integrator=integrator,
            t_start_K=float(spec.t_start_K),
            t_end_K=float(spec.t_end_K),
            n_steps=int(spec.anneal_steps),
            n_segments=10,
        )
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    final_state = context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions(asNumpy=True)
    out = _update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "force_model": "geometric_pre_relax",
        "n_stages": int(spec.n_stages),
        "stage_energies_kj_mol": stage_energies,
        "anneal_enabled": bool(spec.anneal_enabled),
    }
    return out, summary


# ---------------------------------------------------------------------------
# Hybrid pre-relax: real bonded forces (from prmtop) + soft non-bonded
# ---------------------------------------------------------------------------

def _resolve_amber_paths_hybrid(
    amber_artifacts: Dict[str, object],
) -> tuple:
    """Extract prmtop/inpcrd paths, accepting both ambertools and residue_aware backends."""
    from pathlib import Path

    if not bool(amber_artifacts.get("parameterized", False)):
        raise RuntimeError(
            "hybrid_pre_relax requires parameterized AMBER artifacts "
            "(amber_summary.parameterized=true)."
        )
    backend = str(amber_artifacts.get("parameter_backend", "")).strip().lower()
    if backend not in ("ambertools", "residue_aware"):
        raise RuntimeError(
            f"hybrid_pre_relax requires parameter_backend='ambertools' or "
            f"'residue_aware', got {backend!r}."
        )
    files = amber_artifacts.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Amber summary is missing artifact file paths.")
    prmtop = files.get("prmtop")
    inpcrd = files.get("inpcrd")
    if not isinstance(prmtop, str) or not isinstance(inpcrd, str):
        raise RuntimeError("Amber summary must include 'prmtop' and 'inpcrd' paths.")
    prmtop_path, inpcrd_path = Path(prmtop), Path(inpcrd)
    if not prmtop_path.exists() or not inpcrd_path.exists():
        raise RuntimeError(
            f"hybrid_pre_relax could not find Amber artifacts: "
            f"{prmtop_path} / {inpcrd_path}"
        )
    return prmtop_path, inpcrd_path


def _run_hybrid_pre_relax(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    amber_artifacts: Dict[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """Bonded forces from AMBER topology + soft repulsion non-bonded.

    This keeps the molecule covalently intact during Langevin annealing
    while using a gentle, non-singular non-bonded potential to resolve
    steric clashes.
    """
    try:
        import parmed as pmd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "hybrid_pre_relax requires ParmEd. "
            "Install 'parmed' in the current environment."
        ) from exc

    prmtop_path, inpcrd_path = _resolve_amber_paths_hybrid(amber_artifacts)
    structure = pmd.load_file(str(prmtop_path), str(inpcrd_path))

    # --- Build system from prmtop (all bonded + non-bonded forces). ---
    system = structure.createSystem(nonbondedMethod=mm.NoCutoff, constraints=None)
    if int(system.getNumParticles()) != mol.GetNumAtoms():
        raise RuntimeError(
            f"Atom count mismatch: RDKit={mol.GetNumAtoms()} vs "
            f"AMBER={system.getNumParticles()}."
        )

    # --- Disable all non-bonded forces (LJ + Coulomb). ---
    # We remove every NonbondedForce and CustomNonbondedForce that came from
    # the prmtop, then add our own soft repulsion.
    forces_to_remove: list[int] = []
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, (mm.NonbondedForce, mm.CustomNonbondedForce)):
            forces_to_remove.append(i)
    # Remove in reverse order to keep indices stable.
    for idx in reversed(forces_to_remove):
        system.removeForce(idx)

    # --- Add soft repulsion (same model as geometry_pre_relax). ---
    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", 800.0)
    repulsive.addPerParticleParameter("sigma")
    repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsive.setCutoffDistance(0.6 * unit.nanometer)

    for atom in mol.GetAtoms():
        repulsive.addParticle([_sigma_nm(atom)])

    # Exclusions: same 1-2 and 1-3 exclusions as geometry_pre_relax.
    from poly_csp.mm.openmm_system import exclusion_pairs_from_mol
    excluded = exclusion_pairs_from_mol(mol, exclude_13=True, exclude_14=False)
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)

    # --- Use RDKit coords (our "truth") rather than inpcrd (may differ). ---
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates for hybrid pre-relax.")
    xyz_A = np.asarray(
        mol.GetConformer(0).GetPositions(), dtype=float
    ).reshape((-1, 3))
    positions_nm = (xyz_A / 10.0) * unit.nanometer

    # --- Add restraints (same as geometry_pre_relax). ---
    pos_force = add_positional_restraints(
        system=system,
        atom_indices=_backbone_heavy_indices(mol),
        reference_positions_nm=positions_nm,
        k_kj_per_mol_nm2=float(spec.positional_k),
    )
    tors_force = add_dihedral_restraints(
        system=system,
        dihedrals=_selector_dihedral_targets(mol, selector),
        k_kj_per_mol=float(spec.dihedral_k),
    )
    hb_force = add_hbond_distance_restraints(
        system=system,
        pairs=_hbond_pairs(mol, selector),
        k_kj_per_mol_nm2=float(spec.hbond_k),
    )

    # --- Simulation. ---
    integrator = mm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    context = mm.Context(system, integrator)
    context.setPositions(positions_nm)

    stage_factors = np.linspace(1.0, 0.15, max(1, int(spec.n_stages)))
    stage_energies: list[float] = []
    for factor in stage_factors:
        context.setParameter("k_pos", float(spec.positional_k) * float(factor))
        context.setParameter("k_tors", float(spec.dihedral_k) * float(factor))
        context.setParameter("k_hb", float(spec.hbond_k) * float(factor))
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        if spec.anneal_cool_down:
            run_heat_cool_cycle(
                context=context,
                integrator=integrator,
                t_start_K=float(spec.t_start_K),
                t_peak_K=float(spec.t_end_K),
                n_steps=int(spec.anneal_steps),
                n_segments=10,
            )
        else:
            run_temperature_ramp(
                context=context,
                integrator=integrator,
                t_start_K=float(spec.t_start_K),
                t_end_K=float(spec.t_end_K),
                n_steps=int(spec.anneal_steps),
                n_segments=10,
            )
        mm.LocalEnergyMinimizer.minimize(
            context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        state = context.getState(getEnergy=True)
        stage_energies.append(
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        )

    final_state = context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions(asNumpy=True)
    out = _update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "force_model": "hybrid_pre_relax",
        "n_stages": int(spec.n_stages),
        "stage_energies_kj_mol": stage_energies,
        "anneal_enabled": bool(spec.anneal_enabled),
        "anneal_cool_down": bool(spec.anneal_cool_down),
    }
    return out, summary
