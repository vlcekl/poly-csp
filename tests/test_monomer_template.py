from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from poly_csp.topology.monomers import make_glucose_template
from poly_csp.io.rdkit_io import export_glucose_template_sdf


def test_make_glucose_template_has_required_labels() -> None:
    template = make_glucose_template("amylose")

    required = {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "O2",
        "O3",
        "O4",
        "O5",
        "O6",
    }
    assert required.issubset(set(template.atom_idx.keys()))
    assert template.site_idx["C2"] == template.atom_idx["C2"]
    assert template.site_idx["O6"] == template.atom_idx["O6"]
    assert template.mol.GetNumAtoms() > 0


def test_make_glucose_template_natural_oh_has_o1() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    assert "O1" in template.atom_idx
    assert template.representation == "natural_oh"


def test_make_glucose_template_tracks_exchangeable_hydrogens() -> None:
    template = make_glucose_template("amylose", monomer_representation="natural_oh")
    for label in ("O1", "O2", "O3", "O4", "O6"):
        atom = template.mol.GetAtomWithIdx(template.atom_idx[label])
        assert atom.GetTotalNumHs(includeNeighbors=True) == 1

    ring_o = template.mol.GetAtomWithIdx(template.atom_idx["O5"])
    assert ring_o.GetTotalNumHs(includeNeighbors=True) == 0


def test_make_glucose_template_anhydro_free_hydroxyls_keep_total_hydrogens() -> None:
    template = make_glucose_template("amylose", monomer_representation="anhydro")
    for label in ("O2", "O3", "O4", "O6"):
        atom = template.mol.GetAtomWithIdx(template.atom_idx[label])
        assert atom.GetTotalNumHs(includeNeighbors=True) == 1


@pytest.mark.parametrize("polymer", ["amylose", "cellulose"])
def test_anhydro_template_reuses_natural_oh_geometry_except_o1(polymer: str) -> None:
    natural = make_glucose_template(polymer, monomer_representation="natural_oh")
    anhydro = make_glucose_template(polymer, monomer_representation="anhydro")

    natural_xyz = np.asarray(natural.mol.GetConformer(0).GetPositions(), dtype=float)
    anhydro_xyz = np.asarray(anhydro.mol.GetConformer(0).GetPositions(), dtype=float)

    assert "O1" in natural.atom_idx
    assert "O1" not in anhydro.atom_idx

    for label, anhydro_idx in anhydro.atom_idx.items():
        assert label in natural.atom_idx
        natural_idx = natural.atom_idx[label]
        assert np.allclose(
            anhydro_xyz[anhydro_idx],
            natural_xyz[natural_idx],
            atol=1e-6,
        )


def test_amylose_template_has_defined_ring_stereo() -> None:
    template = make_glucose_template("amylose")
    for label in ("C2", "C3", "C4", "C5"):
        atom = template.mol.GetAtomWithIdx(template.atom_idx[label])
        assert atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED


def test_export_glucose_template_sdf(tmp_path: Path) -> None:
    out = tmp_path / "amylose_glucose_template.sdf"
    export_glucose_template_sdf("amylose", out)
    assert out.exists()
    assert out.stat().st_size > 0
