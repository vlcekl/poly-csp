from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable

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
