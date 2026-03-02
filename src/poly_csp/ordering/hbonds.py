from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from rdkit import Chem

from poly_csp.chemistry.selectors import SelectorTemplate


@dataclass(frozen=True)
class HbondMetrics:
    satisfied_pairs: int
    total_pairs: int
    fraction: float
    mean_pair_distance_A: float


def _selector_atom_records(
    mol: Chem.Mol,
    local_indices: Iterable[int],
) -> List[Tuple[int, int]]:
    local_set = set(int(x) for x in local_indices)
    out: List[Tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        local_idx = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        if local_idx in local_set:
            residue = int(atom.GetIntProp("_poly_csp_residue_index"))
            out.append((residue, atom.GetIdx()))
    return out


def compute_hbond_metrics(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    max_distance_A: float = 3.3,
    neighbor_window: int = 1,
) -> HbondMetrics:
    """
    Lightweight pre-organization metric:
    donor N (selector donor atoms) to carbonyl O acceptors within residue-neighbor window.
    """
    if mol.GetNumConformers() == 0:
        return HbondMetrics(0, 0, 0.0, 0.0)

    donors = _selector_atom_records(mol, selector.donors)
    acceptors = _selector_atom_records(mol, selector.acceptors)
    if not donors or not acceptors:
        return HbondMetrics(0, 0, 0.0, 0.0)

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    total = 0
    satisfied = 0
    sat_distances: List[float] = []

    for d_res, d_idx in donors:
        for a_res, a_idx in acceptors:
            if abs(d_res - a_res) > int(neighbor_window):
                continue
            if d_idx == a_idx:
                continue
            total += 1
            dist = float(np.linalg.norm(xyz[d_idx] - xyz[a_idx]))
            if dist <= float(max_distance_A):
                satisfied += 1
                sat_distances.append(dist)

    fraction = float(satisfied / total) if total > 0 else 0.0
    mean_dist = float(np.mean(sat_distances)) if sat_distances else 0.0
    return HbondMetrics(
        satisfied_pairs=satisfied,
        total_pairs=total,
        fraction=fraction,
        mean_pair_distance_A=mean_dist,
    )
