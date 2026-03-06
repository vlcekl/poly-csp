# tests/test_pdb_residue_info.py
"""Verify PDB output contains residue names and chain IDs."""
from __future__ import annotations

import numpy as np

from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.structure.all_atom import build_structure_all_atom_molecule
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.model import build_forcefield_molecule
from poly_csp.io.pdb import write_pdb_from_rdkit


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


def test_pdb_contains_chain_ids(tmp_path) -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()
    dp = 2
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol, template=template,
        residue_index=0, site="C6", selector=selector,
    )

    pdb_path = tmp_path / "test.pdb"
    write_pdb_from_rdkit(mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    # Should contain both chain A (backbone) and chain B (selector)
    atom_lines = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert len(atom_lines) > 0

    chains = {line[21] for line in atom_lines if len(line) > 21}
    assert "A" in chains, "Expected chain A (backbone) in PDB"
    assert "B" in chains, "Expected chain B (selector) in PDB"


def test_pdb_contains_residue_names(tmp_path) -> None:
    template = make_glucose_template("amylose")
    dp = 3
    coords = build_backbone_coords(template, _helix(), dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    pdb_path = tmp_path / "test_backbone.pdb"
    write_pdb_from_rdkit(mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) > 0

    # All backbone atoms should have residue name GLC
    res_names = {line[17:20].strip() for line in atom_lines if len(line) > 20}
    assert "GLC" in res_names, f"Expected 'GLC' in residue names, got: {res_names}"


def test_pdb_uses_forcefield_model_atom_names(tmp_path) -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    selector = make_35_dmpc_template()
    coords = build_backbone_coords(template, _helix(), dp=1)

    mol = polymerize(template=template, dp=1, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )
    final_mol = build_forcefield_molecule(
        build_structure_all_atom_molecule(mol, _helix()).mol
    ).mol

    pdb_path = tmp_path / "test_all_atom.pdb"
    write_pdb_from_rdkit(final_mol, pdb_path)
    text = pdb_path.read_text(encoding="utf-8")

    atom_lines = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    atom_names = {line[12:16].strip() for line in atom_lines}
    assert "HO1" in atom_names
    assert "H61" in atom_names
    assert any(name.startswith("S") for name in atom_names)
