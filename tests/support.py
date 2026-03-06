"""Test-only wrappers around canonical runtime builders."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.backbone_builder import build_backbone_heavy_coords
from poly_csp.topology.monomers import GlucoseMonomerTemplate


def build_backbone_coords(
    template: GlucoseMonomerTemplate,
    helix: HelixSpec,
    dp: int,
) -> np.ndarray:
    """Return canonical heavy-atom backbone coordinates for tests."""
    return build_backbone_heavy_coords(template, helix, dp)


def assign_conformer(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """Attach coordinates to a test molecule without using any runtime fallback path."""
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3); got {xyz.shape}")
    if xyz.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: coords has {xyz.shape[0]}, mol has {mol.GetNumAtoms()}."
        )

    out = Chem.Mol(mol)
    out.RemoveAllConformers()

    conf = Chem.Conformer(out.GetNumAtoms())
    for atom_idx, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
    out.AddConformer(conf, assignId=True)
    return out
