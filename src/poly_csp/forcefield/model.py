from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from poly_csp.structure.naming import AtomManifestEntry, build_atom_manifest
from poly_csp.topology.utils import coords_from_mol


@dataclass(frozen=True)
class ForcefieldModelResult:
    mol: Chem.Mol
    manifest: list[AtomManifestEntry]


def _validate_explicit_hydrogens(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            if not atom.HasProp("_poly_csp_parent_heavy_idx"):
                raise ValueError(
                    f"Hydrogen atom {atom.GetIdx()} is missing _poly_csp_parent_heavy_idx."
                )
            continue
        if atom.GetNumImplicitHs() != 0:
            raise ValueError(
                f"Atom {atom.GetIdx()} still has {atom.GetNumImplicitHs()} implicit hydrogens."
            )


def _validate_backbone_metadata(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_component"):
            raise ValueError(f"Atom {atom.GetIdx()} is missing _poly_csp_component.")
        if atom.GetProp("_poly_csp_component") == "backbone":
            if atom.HasProp("_poly_csp_terminal_cap_side"):
                continue
            if not atom.HasProp("_poly_csp_residue_index"):
                raise ValueError(
                    f"Backbone atom {atom.GetIdx()} is missing _poly_csp_residue_index."
                )
            if not atom.HasProp("_poly_csp_residue_label"):
                raise ValueError(
                    f"Backbone atom {atom.GetIdx()} is missing _poly_csp_residue_label."
                )


def build_forcefield_molecule(
    mol_structure_all_atom: Chem.Mol,
) -> ForcefieldModelResult:
    """Validate and normalize the structure-domain all-atom molecule for forcefield use."""
    coords_before = coords_from_mol(mol_structure_all_atom)
    mol = Chem.Mol(mol_structure_all_atom)
    _validate_explicit_hydrogens(mol)
    _validate_backbone_metadata(mol)

    manifest = build_atom_manifest(mol)
    if len(manifest) != mol.GetNumAtoms():
        raise ValueError("Atom manifest does not cover every atom in the molecule.")

    coords_after = coords_from_mol(mol)
    if coords_before is not None and coords_after is not None:
        if not np.allclose(coords_before, coords_after, atol=1e-8):
            raise ValueError("Forcefield molecule normalization changed atomic coordinates.")

    return ForcefieldModelResult(mol=mol, manifest=manifest)
