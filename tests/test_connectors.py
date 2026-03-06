from __future__ import annotations

import pytest
from rdkit import Chem

from poly_csp.forcefield.connectors import (
    CappedMonomerFragment,
    ConnectorParams,
    build_capped_monomer_fragment,
    extract_linkage_params,
    parameterize_capped_monomer,
)
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.selector_library.tmb import make_tmb_template


def test_extract_linkage_params_reports_stub_metadata(tmp_path) -> None:
    out = extract_linkage_params(tmp_path / "frag.prmtop", atom_map={0: 1, 2: 3})
    assert out["status"] == "not_implemented"
    assert out["atom_map_size"] == 2
    assert str(tmp_path / "frag.prmtop") == out["prmtop"]


def test_parameterize_capped_monomer_validates_inputs() -> None:
    with pytest.raises(ValueError, match="backbone_template"):
        parameterize_capped_monomer(Chem.Mol(), Chem.MolFromSmiles("CC"), site="C6")

    with pytest.raises(ValueError, match="selector_template"):
        parameterize_capped_monomer(Chem.MolFromSmiles("CC"), Chem.Mol(), site="C6")

    with pytest.raises(ValueError, match="site"):
        parameterize_capped_monomer(Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CC"), site="")


def test_parameterize_capped_monomer_returns_connector_params() -> None:
    out = parameterize_capped_monomer(
        Chem.MolFromSmiles("CC"),
        Chem.MolFromSmiles("CC"),
        site="C6",
    )
    assert isinstance(out, ConnectorParams)
    assert out.bond_params == {}
    assert out.angle_params == {}
    assert out.torsion_params == {}


def test_build_capped_monomer_fragment_assigns_backbone_and_selector_roles() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_35_dmpc_template(),
        site="C6",
    )

    assert isinstance(frag, CappedMonomerFragment)
    assert frag.mol.GetNumConformers() == 1
    assert "BB_O6" in frag.atom_roles
    assert "BB_C6" in frag.atom_roles
    assert any(role.startswith("SL_") for role in frag.atom_roles)
    assert set(frag.connector_roles) == {"carbonyl_c", "carbonyl_o", "amide_n"}
    assert len(frag.atom_roles) == frag.mol.GetNumAtoms()


def test_build_capped_monomer_fragment_handles_ester_selector() -> None:
    frag = build_capped_monomer_fragment(
        polymer="amylose",
        selector_template=make_tmb_template(),
        site="C6",
    )

    assert set(frag.connector_roles) == {"carbonyl_c", "carbonyl_o"}
    for atom_idx in frag.connector_roles.values():
        atom = frag.mol.GetAtomWithIdx(atom_idx)
        assert atom.GetProp("_poly_csp_component") == "connector"
        assert atom.HasProp("_poly_csp_fragment_role")
