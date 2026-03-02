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
from poly_csp.mm.anneal import run_temperature_ramp
from poly_csp.mm.openmm_system import build_relaxation_system
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
    mode: Literal["geometry_pre_relax", "ambertools_parameterized"] = (
        "geometry_pre_relax"
    )
    n_stages: int = 3
    max_iterations: int = 200
    anneal_enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    anneal_steps: int = 2000


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
