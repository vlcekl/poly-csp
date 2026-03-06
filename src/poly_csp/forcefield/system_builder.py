from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from rdkit import Chem

import openmm as mm
from openmm import unit


@dataclass(frozen=True)
class SystemBuildResult:
    system: mm.System
    positions_nm: unit.Quantity
    excluded_pairs: set[tuple[int, int]]


_SIGMA_A_BY_Z = {
    1: 1.10,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
}


def _atomic_mass_dalton(z: int) -> float:
    if z <= 1:
        return 1.008
    if z == 6:
        return 12.011
    if z == 7:
        return 14.007
    if z == 8:
        return 15.999
    if z == 16:
        return 32.06
    return 12.0


def _sigma_nm(atom: Chem.Atom) -> float:
    return float(_SIGMA_A_BY_Z.get(atom.GetAtomicNum(), 1.70) / 10.0)


def exclusion_pairs_from_mol(
    mol: Chem.Mol,
    exclude_13: bool = True,
    exclude_14: bool = False,
) -> set[tuple[int, int]]:
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    max_depth = 1 + int(exclude_13) + int(exclude_14)
    excluded: set[tuple[int, int]] = set()
    for src in range(n):
        q: deque[tuple[int, int]] = deque([(src, 0)])
        seen = {src}
        while q:
            node, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nbr in adj[node]:
                i, j = (src, nbr) if src < nbr else (nbr, src)
                if src != nbr:
                    excluded.add((i, j))
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, depth + 1))
    return excluded


def build_relaxation_system(
    mol: Chem.Mol,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
    exclude_13: bool = True,
) -> SystemBuildResult:
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")

    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    positions_nm = (xyz_A / 10.0) * unit.nanometer

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    # Pairwise soft repulsion to resolve overlaps without full parameterization.
    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
    repulsive.addPerParticleParameter("sigma")
    repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsive.setCutoffDistance(float(repulsion_cutoff_nm) * unit.nanometer)

    for atom in mol.GetAtoms():
        repulsive.addParticle([_sigma_nm(atom)])

    excluded = exclusion_pairs_from_mol(mol, exclude_13=exclude_13, exclude_14=False)
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)

    return SystemBuildResult(system=system, positions_nm=positions_nm, excluded_pairs=excluded)


# ---------------------------------------------------------------------------
# Covalent radii (Å) for generic bond-length estimation.
# ---------------------------------------------------------------------------
_COVALENT_RADIUS_A = {
    1: 0.31, 6: 0.77, 7: 0.73, 8: 0.73, 9: 0.64,
    15: 1.07, 16: 1.02, 17: 0.99,
}


def _covalent_bond_length_nm(z1: int, z2: int) -> float:
    """Equilibrium bond length (nm) as sum of covalent radii."""
    r1 = _COVALENT_RADIUS_A.get(z1, 0.77)
    r2 = _COVALENT_RADIUS_A.get(z2, 0.77)
    return (r1 + r2) / 10.0  # Å → nm


def _equilibrium_angle_rad(central_atom: Chem.Atom) -> float:
    """Guess equilibrium angle from the number of heavy neighbours (hybridization proxy)."""
    import math
    n_neighbors = central_atom.GetDegree()
    if n_neighbors <= 2:
        return math.pi          # sp  → 180°
    if n_neighbors == 3:
        return 2.0943951        # sp2 → 120°
    return 1.9106332            # sp3 → 109.5°


def build_bonded_relaxation_system(
    mol: Chem.Mol,
    bond_k_kj_per_mol_nm2: float = 200_000.0,
    angle_k_kj_per_mol_rad2: float = 500.0,
    repulsion_k_kj_per_mol_nm2: float = 800.0,
    repulsion_cutoff_nm: float = 0.6,
    exclude_13: bool = True,
) -> SystemBuildResult:
    """Build an OpenMM system with **generic bonded forces** derived from RDKit.

    Unlike ``build_relaxation_system`` (soft repulsion only), this builder
    adds ``HarmonicBondForce`` and ``HarmonicAngleForce`` so that the
    molecule stays covalently intact during Langevin dynamics / annealing.
    The parameters are approximate (covalent-radii bond lengths,
    hybridisation-based angles) — sufficient to preserve topology, but not
    production-quality force-field values.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before OpenMM system build.")

    xyz_A = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    positions_nm = (xyz_A / 10.0) * unit.nanometer

    system = mm.System()
    for atom in mol.GetAtoms():
        system.addParticle(_atomic_mass_dalton(atom.GetAtomicNum()) * unit.dalton)

    # --- Harmonic bonds from RDKit bond graph. ---
    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        z1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        z2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        r0 = _covalent_bond_length_nm(z1, z2)
        bond_force.addBond(i, j, r0, bond_k_kj_per_mol_nm2)
    system.addForce(bond_force)

    # --- Harmonic angles from bond-graph i-j-k triples. ---
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    angle_force = mm.HarmonicAngleForce()
    for j in range(n):
        nbrs = adj[j]
        theta0 = _equilibrium_angle_rad(mol.GetAtomWithIdx(j))
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                angle_force.addAngle(nbrs[ii], j, nbrs[jj], theta0, angle_k_kj_per_mol_rad2)
    system.addForce(angle_force)

    # --- Soft pairwise repulsion (same model as build_relaxation_system). ---
    repulsive = mm.CustomNonbondedForce(
        "k_rep*step(sigma-r)*(sigma-r)^2;"
        "sigma=0.5*(sigma1+sigma2)"
    )
    repulsive.addGlobalParameter("k_rep", float(repulsion_k_kj_per_mol_nm2))
    repulsive.addPerParticleParameter("sigma")
    repulsive.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    repulsive.setCutoffDistance(float(repulsion_cutoff_nm) * unit.nanometer)

    for atom in mol.GetAtoms():
        repulsive.addParticle([_sigma_nm(atom)])

    excluded = exclusion_pairs_from_mol(mol, exclude_13=exclude_13, exclude_14=False)
    for i, j in sorted(excluded):
        repulsive.addExclusion(int(i), int(j))
    system.addForce(repulsive)

    return SystemBuildResult(system=system, positions_nm=positions_nm, excluded_pairs=excluded)


def build_selector_bonded_forces(
    mol: Chem.Mol,
    selector_indices: set[int],
    bond_k: float = 200_000.0,
    angle_k: float = 500.0,
) -> tuple[mm.HarmonicBondForce, mm.HarmonicAngleForce]:
    """Build bonded forces for bonds/angles involving at least one selector atom.

    This covers:
    - Pure selector bonds/angles (both atoms are selectors)
    - Junction bonds/angles (one backbone atom, one selector atom)

    The forces use generic covalent-radius bond lengths and hybridisation-
    based equilibrium angles — not production-quality, but sufficient to
    keep the molecule intact during annealing.
    """
    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i not in selector_indices and j not in selector_indices:
            continue  # pure backbone — handled by AMBER forces
        z1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        z2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        r0 = _covalent_bond_length_nm(z1, z2)
        bond_force.addBond(i, j, r0, bond_k)

    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    angle_force = mm.HarmonicAngleForce()
    for j in range(n):
        nbrs = adj[j]
        theta0 = _equilibrium_angle_rad(mol.GetAtomWithIdx(j))
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                a, b = nbrs[ii], nbrs[jj]
                # Include this angle if any of the three atoms is a selector
                if a not in selector_indices and j not in selector_indices and b not in selector_indices:
                    continue
                angle_force.addAngle(a, j, b, theta0, angle_k)

    return bond_force, angle_force


def add_glycam_backbone(
    system: mm.System,
    mol: Chem.Mol,
    backbone_indices: set[int],
    glycam_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Attach backbone-specific parameters to an existing system (incremental API)."""
    _ = (mol, backbone_indices, glycam_params)
    return system


def add_gaff_selectors(
    system: mm.System,
    mol: Chem.Mol,
    selector_indices: set[int],
    gaff_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Attach selector-specific bonded parameters to an existing system."""
    if not gaff_params:
        return system
    bond_force, angle_force = build_selector_bonded_forces(mol, selector_indices)
    system.addForce(bond_force)
    system.addForce(angle_force)
    return system


def add_connectors(
    system: mm.System,
    mol: Chem.Mol,
    connector_indices: set[int],
    connector_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Attach connector parameters to an existing system (placeholder)."""
    _ = (mol, connector_indices, connector_params)
    return system


def create_system(
    mol: Chem.Mol,
    atom_map: Mapping[int, Any] | None = None,
    glycam_params: Mapping[str, Any] | None = None,
    gaff_params: Mapping[str, Any] | None = None,
    connector_params: Mapping[str, Any] | None = None,
) -> mm.System:
    """Construct a relaxation-ready OpenMM ``System`` from a fully built RDKit molecule."""
    from poly_csp.topology.atom_mapping import (
        backbone_indices,
        connector_indices,
        selector_indices,
    )
    from poly_csp.forcefield.exceptions import apply_mixing_rules

    built = build_bonded_relaxation_system(mol)
    system = built.system
    system = add_glycam_backbone(system, mol, backbone_indices(mol), glycam_params)
    system = add_gaff_selectors(system, mol, selector_indices(mol), gaff_params)
    system = add_connectors(system, mol, connector_indices(mol), connector_params)

    if atom_map is not None:
        apply_mixing_rules(system=system, atom_map=atom_map)

    return system
