from __future__ import annotations

import numpy as np

from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.scoring import bonded_exclusion_pairs, min_interatomic_distance
from poly_csp.ordering.optimize import OrderingSpec, optimize_selector_ordering


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0 * np.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _heavy_mask(mol) -> np.ndarray:
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def test_ordering_summary_min_distance_uses_qc_style_exclusions() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    dp = 4

    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    for i in range(dp):
        mol = attach_selector(
            mol_polymer=mol,
            template=template,
            residue_index=i,
            site="C6",
            selector=selector,
        )

    spec = OrderingSpec(enabled=True, repeat_residues=1, max_candidates=8)
    _, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=dp,
        spec=spec,
    )

    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    heavy = _heavy_mask(mol)
    naive = float(min_interatomic_distance(xyz, heavy, excluded_pairs=None))
    max_path_length = 1 + int(spec.exclude_13) + int(spec.exclude_14)
    excluded = bonded_exclusion_pairs(mol, max_path_length=max_path_length)
    exclusion_aware = float(min_interatomic_distance(xyz, heavy, excluded_pairs=excluded))

    summary_value = float(summary["baseline_min_heavy_distance_A"])
    assert abs(summary_value - exclusion_aware) < 1e-9
    assert summary_value >= naive
