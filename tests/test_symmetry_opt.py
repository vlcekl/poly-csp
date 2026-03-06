from __future__ import annotations

import numpy as np

from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
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


def test_optimize_selector_ordering_returns_summary() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    dp = 3

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
    out, summary = optimize_selector_ordering(
        mol=mol,
        selector=selector,
        sites=["C6"],
        dp=dp,
        spec=spec,
    )
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    assert out.GetNumConformers() == 1
    assert summary["enabled"] is True
    assert "final_hbond_geometric_fraction" in summary
    assert "final_class_min_distance_A" in summary
    assert "selected_pose_by_site" in summary
    assert "C6" in summary["selected_pose_by_site"]
