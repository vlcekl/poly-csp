from __future__ import annotations

import numpy as np

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template


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


def test_attach_selector_place_coords_false_builds_topology_without_conformer() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=1))
    n_before = mol.GetNumAtoms()

    out = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
        place_coords=False,
    )

    assert out.GetNumConformers() == 0
    assert out.GetNumAtoms() > n_before


def test_attach_selector_without_input_conformer_keeps_topology_only() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    out = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
        place_coords=False,
    )

    assert out.GetNumConformers() == 0
    assert out.GetNumAtoms() > mol.GetNumAtoms()
