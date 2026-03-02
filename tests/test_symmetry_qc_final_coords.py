from __future__ import annotations

import json

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.config.schema import HelixSpec
from poly_csp.ordering.scoring import screw_symmetry_rmsd_from_mol


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


def _set_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def test_screw_symmetry_rmsd_from_mol_uses_final_coordinates() -> None:
    template = make_glucose_template("amylose")
    helix = _helix()
    dp = 6

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    mol = polymerize(template=template, dp=dp, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, coords)

    baseline = screw_symmetry_rmsd_from_mol(mol, helix=helix, k=4)
    assert baseline < 1e-10

    maps = json.loads(mol.GetProp("_poly_csp_residue_label_map_json"))
    atom_to_perturb = int(maps[4]["C2"])
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    xyz[atom_to_perturb] += np.array([0.35, 0.0, 0.0], dtype=float)
    perturbed = _set_coords(mol, xyz)

    updated = screw_symmetry_rmsd_from_mol(perturbed, helix=helix, k=4)
    assert updated > 1e-3
