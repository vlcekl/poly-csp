from __future__ import annotations

import json

import numpy as np

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.all_atom import (
    build_structure_all_atom_molecule,
    select_residue_templates,
)
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.terminals import apply_terminal_mode


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def _assign_backbone_coords(mol, template, dp: int):
    from poly_csp.structure.build_helix import build_backbone_coords

    coords = build_backbone_coords(template, _helix(), dp)
    removed = json.loads(mol.GetProp("_poly_csp_removed_old_indices_json"))
    if removed:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed, dtype=int)] = False
        coords = coords[keep_mask]
    return assign_conformer(mol, coords)


def test_select_residue_templates_detects_substitution_and_termini() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=2)
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    states = select_residue_templates(mol)
    assert len(states) == 2
    assert states[0].substituted_sites == ("C6",)
    assert states[0].incoming_link is False
    assert states[0].outgoing_link is True
    assert states[1].incoming_link is True
    assert states[1].outgoing_link is False


def test_select_residue_templates_tracks_periodic_and_capped_anchor_states() -> None:
    natural = make_glucose_template("amylose", monomer_representation="natural_oh")
    periodic = polymerize(template=natural, dp=3, linkage="1-4", anomer="alpha")
    periodic = apply_terminal_mode(
        periodic,
        mode="periodic",
        caps={},
        representation="natural_oh",
    )
    periodic_states = select_residue_templates(periodic)
    assert periodic_states[0].has_o1 is False
    assert all(state.incoming_link for state in periodic_states)
    assert all(state.outgoing_link for state in periodic_states)

    anhydro = make_glucose_template("amylose")
    capped = polymerize(template=anhydro, dp=2, linkage="1-4", anomer="alpha")
    capped = apply_terminal_mode(
        capped,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="anhydro",
    )
    capped_states = select_residue_templates(capped)
    assert capped_states[0].left_cap == "acetyl"
    assert capped_states[0].left_anchor_label == "C1"
    assert capped_states[-1].right_cap == "methoxy"
    assert capped_states[-1].right_anchor_label == "O4"


def test_build_structure_all_atom_molecule_preserves_heavy_coords_and_backbone_hydrogens() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=2)
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    heavy_xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)

    out = build_structure_all_atom_molecule(mol, _helix()).mol
    all_xyz = np.asarray(out.GetConformer(0).GetPositions(), dtype=float)
    assert np.allclose(all_xyz[: mol.GetNumAtoms()], heavy_xyz, atol=1e-6)

    maps = json.loads(out.GetProp("_poly_csp_residue_label_map_json"))
    res0_o6 = out.GetAtomWithIdx(maps[0]["O6"])
    res0_o4 = out.GetAtomWithIdx(maps[0]["O4"])
    res1_o4 = out.GetAtomWithIdx(maps[1]["O4"])
    res0_c6 = out.GetAtomWithIdx(maps[0]["C6"])

    assert sum(1 for nbr in res0_o6.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert sum(1 for nbr in res0_o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert sum(1 for nbr in res1_o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 1
    assert sum(1 for nbr in res0_c6.GetNeighbors() if nbr.GetAtomicNum() == 1) == 2

    amide_n = next(
        atom
        for atom in out.GetAtoms()
        if atom.HasProp("_poly_csp_connector_role")
        and atom.GetProp("_poly_csp_connector_role") == "amide_n"
    )
    amide_h = [nbr for nbr in amide_n.GetNeighbors() if nbr.GetAtomicNum() == 1]
    assert len(amide_h) == 1
    assert amide_h[0].GetProp("_poly_csp_component") == "connector"


def test_build_structure_all_atom_molecule_handles_capped_termini() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=2)
    mol = apply_terminal_mode(
        mol,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="anhydro",
    )

    out = build_structure_all_atom_molecule(mol, _helix()).mol
    maps = json.loads(out.GetProp("_poly_csp_residue_label_map_json"))
    c1 = out.GetAtomWithIdx(maps[0]["C1"])
    o4 = out.GetAtomWithIdx(maps[-1]["O4"])

    assert sum(1 for nbr in c1.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert sum(1 for nbr in o4.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0


def test_build_structure_all_atom_molecule_rebuilds_backbone_without_conformer() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol.RemoveAllConformers()

    out = build_structure_all_atom_molecule(mol, _helix()).mol
    assert out.GetNumConformers() == 1
    assert out.GetNumAtoms() > mol.GetNumAtoms()
