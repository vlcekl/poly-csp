from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from rdkit import Chem

from poly_csp.chemistry.functionalization import apply_selector_pose_dihedrals
from poly_csp.chemistry.selectors import SelectorTemplate
from poly_csp.config.schema import Site
from poly_csp.ordering.hbonds import HbondMetrics, compute_hbond_metrics
from poly_csp.ordering.rotamers import (
    RotamerGridSpec,
    default_rotamer_grid,
    enumerate_pose_library,
)
from poly_csp.ordering.scoring import min_interatomic_distance


@dataclass(frozen=True)
class OrderingSpec:
    enabled: bool = False
    repeat_residues: int = 1
    max_distance_A: float = 3.3
    neighbor_window: int = 1
    clash_target_A: float = 1.1
    hbond_weight: float = 8.0
    clash_weight: float = 10.0
    max_candidates: int = 64


def _heavy_mask(mol: Chem.Mol) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def _objective(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    spec: OrderingSpec,
) -> tuple[float, HbondMetrics, float]:
    hb = compute_hbond_metrics(
        mol=mol,
        selector=selector,
        max_distance_A=spec.max_distance_A,
        neighbor_window=spec.neighbor_window,
    )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    dmin = float(min_interatomic_distance(xyz, _heavy_mask(mol)))
    clash_penalty = max(0.0, float(spec.clash_target_A) - dmin)
    score = float(spec.hbond_weight) * hb.fraction - float(spec.clash_weight) * clash_penalty
    return score, hb, dmin


def optimize_selector_ordering(
    mol: Chem.Mol,
    selector: SelectorTemplate,
    sites: Iterable[Site],
    dp: int,
    spec: OrderingSpec,
    grid: RotamerGridSpec | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """
    Deterministic stage-4 optimizer:
    choose one rotamer pose per site and apply it consistently across residues.
    """
    if not spec.enabled:
        baseline_score, baseline_hb, baseline_dmin = _objective(mol, selector, spec)
        return Chem.Mol(mol), {
            "enabled": False,
            "baseline_score": baseline_score,
            "baseline_hbond_fraction": baseline_hb.fraction,
            "baseline_min_heavy_distance_A": baseline_dmin,
            "final_score": baseline_score,
            "final_hbond_fraction": baseline_hb.fraction,
            "final_min_heavy_distance_A": baseline_dmin,
            "selected_pose_by_site": {},
        }

    grid_spec = grid or default_rotamer_grid(selector.name)
    if spec.max_candidates > 0:
        grid_spec = RotamerGridSpec(
            dihedral_values_deg=grid_spec.dihedral_values_deg,
            max_candidates=min(grid_spec.max_candidates, spec.max_candidates),
        )
    pose_library = enumerate_pose_library(grid_spec)

    work = Chem.Mol(mol)
    baseline_score, baseline_hb, baseline_dmin = _objective(work, selector, spec)
    selected: Dict[str, Dict[str, float]] = {}

    repeat = max(1, min(int(spec.repeat_residues), int(dp)))
    residues = list(range(int(dp)))

    for site in [str(s) for s in sites]:
        best_mol = Chem.Mol(work)
        best_score = float("-inf")
        best_pose: Dict[str, float] = {}

        for pose in pose_library:
            trial = Chem.Mol(work)
            # Apply one periodic pose pattern across the chain.
            for residue_index in residues:
                pattern_idx = residue_index % repeat
                if pattern_idx < repeat:
                    trial = apply_selector_pose_dihedrals(
                        mol=trial,
                        residue_index=residue_index,
                        site=site,  # type: ignore[arg-type]
                        pose_spec=pose,
                        selector=selector,
                    )
            score, _, _ = _objective(trial, selector, spec)
            if score > best_score:
                best_score = score
                best_mol = trial
                best_pose = dict(pose.dihedral_targets_deg)

        work = best_mol
        selected[site] = best_pose

    final_score, final_hb, final_dmin = _objective(work, selector, spec)
    summary: Dict[str, object] = {
        "enabled": True,
        "repeat_residues": repeat,
        "candidate_count": len(pose_library),
        "baseline_score": baseline_score,
        "baseline_hbond_fraction": baseline_hb.fraction,
        "baseline_min_heavy_distance_A": baseline_dmin,
        "final_score": final_score,
        "final_hbond_fraction": final_hb.fraction,
        "final_min_heavy_distance_A": final_dmin,
        "selected_pose_by_site": selected,
    }
    return work, summary
