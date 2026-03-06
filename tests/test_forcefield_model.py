from __future__ import annotations

import json

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.structure.all_atom import build_structure_all_atom_molecule
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
        import numpy as np

        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed, dtype=int)] = False
        coords = coords[keep_mask]
    return assign_conformer(mol, coords)


def test_build_forcefield_molecule_assigns_manifest_names_and_pdb_info() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=1)

    structure_result = build_structure_all_atom_molecule(mol, _helix())
    result = build_forcefield_molecule(structure_result.mol)

    assert len(result.manifest) == result.mol.GetNumAtoms()
    assert result.mol.GetIntProp("_poly_csp_manifest_schema_version") == 1
    assert all(atom.GetPDBResidueInfo() is not None for atom in result.mol.GetAtoms())

    names = {atom.GetProp("_poly_csp_atom_name") for atom in result.mol.GetAtoms()}
    assert {"C1", "O1", "H1", "HO1", "HO4", "H61", "H62", "HO6"}.issubset(names)


def test_build_forcefield_molecule_preserves_selector_connector_and_cap_identity() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=1)
    mol = apply_terminal_mode(
        mol,
        mode="capped",
        caps={"left": "acetyl", "right": "methoxy"},
        representation="natural_oh",
    )
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    result = build_forcefield_molecule(
        build_structure_all_atom_molecule(mol, _helix()).mol
    )
    maps = json.loads(result.mol.GetProp("_poly_csp_residue_label_map_json"))
    o1 = result.mol.GetAtomWithIdx(maps[0]["O1"])

    selector_entries = [entry for entry in result.manifest if entry.component == "selector"]
    connector_entries = [entry for entry in result.manifest if entry.component == "connector"]
    left_caps = [entry for entry in result.manifest if entry.source == "terminal_cap_left"]
    right_caps = [entry for entry in result.manifest if entry.source == "terminal_cap_right"]

    assert sum(1 for nbr in o1.GetNeighbors() if nbr.GetAtomicNum() == 1) == 0
    assert selector_entries
    assert all(entry.atom_name.startswith(("S", "H")) for entry in selector_entries)
    assert connector_entries
    assert any(entry.connector_role == "amide_n" for entry in connector_entries)
    assert left_caps
    assert right_caps


def test_build_forcefield_molecule_rejects_implicit_hydrogen_inputs() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = _assign_backbone_coords(mol, template, dp=1)

    try:
        build_forcefield_molecule(mol)
    except ValueError as exc:
        assert "implicit hydrogens" in str(exc) or "_poly_csp_component" in str(exc)
    else:
        raise AssertionError("Expected build_forcefield_molecule() to reject a heavy-only input.")
