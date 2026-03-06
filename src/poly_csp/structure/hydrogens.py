from __future__ import annotations

from typing import Literal

from rdkit import Chem
from rdkit.Chem import AllChem

from poly_csp.topology.utils import copy_mol_props

HydrogenOptimizeMode = Literal["none", "h_only"]


def _propagate_parent_metadata(mol: Chem.Mol, heavy_atom_count: int) -> None:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        if atom.GetDegree() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        parent_idx = int(parent.GetIdx())
        atom.SetIntProp("_poly_csp_parent_heavy_idx", parent_idx)
        for key in (
            "_poly_csp_component",
            "_poly_csp_selector_instance",
            "_poly_csp_residue_index",
            "_poly_csp_site",
        ):
            if not parent.HasProp(key):
                continue
            if key == "_poly_csp_selector_instance" or key == "_poly_csp_residue_index":
                atom.SetIntProp(key, int(parent.GetIntProp(key)))
            else:
                atom.SetProp(key, parent.GetProp(key))

    for atom_idx in range(min(heavy_atom_count, mol.GetNumAtoms())):
        heavy = mol.GetAtomWithIdx(int(atom_idx))
        if heavy.GetAtomicNum() == 1:
            continue
        if heavy.HasProp("_poly_csp_parent_heavy_idx"):
            heavy.ClearProp("_poly_csp_parent_heavy_idx")


def _optimize_hydrogens_only(
    mol: Chem.Mol,
    max_iterations: int = 200,
) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        return mol

    out = Chem.Mol(mol)
    heavy_indices = [
        atom.GetIdx()
        for atom in out.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    ff = None
    if AllChem.MMFFHasAllMoleculeParams(out):
        props = AllChem.MMFFGetMoleculeProperties(out, mmffVariant="MMFF94")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(out, props)
    if ff is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(out)
        except Exception:
            ff = None
    if ff is None:
        return out

    for atom_idx in heavy_indices:
        ff.AddFixedPoint(int(atom_idx))
    ff.Initialize()
    ff.Minimize(maxIts=int(max_iterations))
    return out


def complete_with_hydrogens(
    mol: Chem.Mol,
    *,
    add_coords: bool = True,
    optimize: HydrogenOptimizeMode = "h_only",
) -> Chem.Mol:
    """Return an all-atom derived structure from a hydrogen-suppressed master."""
    base = Chem.Mol(mol)
    heavy_atom_count = base.GetNumAtoms()
    out = Chem.AddHs(base, addCoords=bool(add_coords))
    copy_mol_props(base, out)
    _propagate_parent_metadata(out, heavy_atom_count)

    if optimize == "h_only":
        out = _optimize_hydrogens_only(out)
        copy_mol_props(base, out)
        _propagate_parent_metadata(out, heavy_atom_count)

    return out
