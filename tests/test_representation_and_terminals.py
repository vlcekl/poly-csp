from __future__ import annotations

import json

import numpy as np

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.functionalization import attach_selector
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.chemistry.terminals import apply_terminal_mode
from poly_csp.config.schema import HelixSpec


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


def test_natural_oh_polymerize_removes_internal_o1() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    dp = 4
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")

    n = template.mol.GetNumAtoms()
    assert mol.GetNumAtoms() == dp * n - (dp - 1)
    assert mol.GetProp("_poly_csp_representation") == "natural_oh"

    maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    assert len(maps) == dp
    assert "O1" in maps[0]
    for r in range(1, dp):
        assert "O1" not in maps[r]


def test_natural_oh_coords_pruning_and_selector_attachment() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()
    dp = 3

    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    removed = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    keep_mask = np.ones((coords.shape[0],), dtype=bool)
    keep_mask[np.asarray(removed, dtype=int)] = False
    mol = assign_conformer(mol, coords[keep_mask])

    mol2 = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=1,
        site="C6",
        selector=selector,
    )
    assert mol2.GetNumConformers() == 1
    assert int(mol2.GetIntProp("_poly_csp_dp")) == dp


def test_apply_terminal_mode_sets_policy_metadata() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    n = mol.GetNumAtoms()

    open_mol = apply_terminal_mode(mol, mode="open", caps={}, representation="anhydro")
    capped_mol = apply_terminal_mode(
        mol, mode="capped", caps={"left": "acetyl", "right": "methoxy"}, representation="anhydro"
    )
    periodic_mol = apply_terminal_mode(
        mol, mode="periodic", caps={}, representation="anhydro"
    )

    assert open_mol.GetNumAtoms() == n
    assert capped_mol.GetNumAtoms() == n
    assert periodic_mol.GetNumAtoms() == n

    assert open_mol.GetProp("_poly_csp_end_mode") == "open"
    assert capped_mol.GetProp("_poly_csp_end_mode") == "capped"
    assert periodic_mol.GetProp("_poly_csp_end_mode") == "periodic"
