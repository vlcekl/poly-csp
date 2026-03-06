from __future__ import annotations

import numpy as np

from rdkit import Chem

from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.backbone import polymerize
from tests.support import assign_conformer


def test_polymerize_atom_count_and_link_bonds() -> None:
    template = make_glucose_template("amylose")
    dp = 5
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")

    n = template.mol.GetNumAtoms()
    assert mol.GetNumAtoms() == dp * n

    for i in range(dp - 1):
        o4_i = i * n + template.atom_idx["O4"]
        c1_ip1 = (i + 1) * n + template.atom_idx["C1"]
        assert mol.GetBondBetweenAtoms(o4_i, c1_ip1) is not None


def test_polymerize_sanitizes() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=3, linkage="1-4", anomer="alpha")
    Chem.SanitizeMol(mol)


def test_assign_conformer_attaches_coordinates() -> None:
    template = make_glucose_template("amylose")
    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")

    coords = np.zeros((mol.GetNumAtoms(), 3), dtype=float)
    coords[:, 0] = np.arange(mol.GetNumAtoms(), dtype=float)
    with_conf = assign_conformer(mol, coords)

    assert with_conf.GetNumConformers() == 1
    conf = with_conf.GetConformer(0)
    p7 = conf.GetAtomPosition(7)
    assert np.isclose(p7.x, 7.0)
