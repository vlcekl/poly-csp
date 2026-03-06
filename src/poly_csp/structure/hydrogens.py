from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rdkit import Chem
from rdkit.Chem import AllChem

from poly_csp.topology.utils import copy_mol_props

HydrogenOptimizeMode = Literal["none", "h_only"]


def _propagate_parent_metadata(mol: Chem.Mol, original_atom_count: int) -> None:
    for atom_idx in range(int(original_atom_count), mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(int(atom_idx))
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
            "_poly_csp_connector_atom",
            "_poly_csp_connector_role",
            "_poly_csp_selector_local_idx",
            "_poly_csp_selector_name",
            "_poly_csp_linkage_type",
            "_poly_csp_residue_label",
            "_poly_csp_terminal_cap_side",
        ):
            if not parent.HasProp(key):
                continue
            if key in {
                "_poly_csp_selector_instance",
                "_poly_csp_residue_index",
                "_poly_csp_connector_atom",
                "_poly_csp_selector_local_idx",
            }:
                atom.SetIntProp(key, int(parent.GetIntProp(key)))
            else:
                atom.SetProp(key, parent.GetProp(key))


def _optimize_hydrogens_only(
    mol: Chem.Mol,
    max_iterations: int = 200,
    movable_atom_indices: Sequence[int] | None = None,
) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        return mol

    out = Chem.Mol(mol)
    movable = None if movable_atom_indices is None else {int(idx) for idx in movable_atom_indices}
    if movable is not None and not movable:
        return out

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

    for atom in out.GetAtoms():
        atom_idx = int(atom.GetIdx())
        is_fixed = (
            atom.GetAtomicNum() > 1
            if movable is None
            else atom_idx not in movable
        )
        if is_fixed:
            ff.AddFixedPoint(atom_idx)
    ff.Initialize()
    ff.Minimize(maxIts=int(max_iterations))
    return out


def complete_with_hydrogens(
    mol: Chem.Mol,
    *,
    add_coords: bool = True,
    optimize: HydrogenOptimizeMode = "h_only",
    only_on_atoms: Sequence[int] | None = None,
) -> Chem.Mol:
    """Return an all-atom derived structure from a hydrogen-suppressed master."""
    base = Chem.Mol(mol)
    original_atom_count = base.GetNumAtoms()
    add_h_kwargs: dict[str, object] = {"addCoords": bool(add_coords)}
    if only_on_atoms is not None:
        add_h_kwargs["onlyOnAtoms"] = [int(idx) for idx in only_on_atoms]
    out = Chem.AddHs(base, **add_h_kwargs)
    copy_mol_props(base, out)
    _propagate_parent_metadata(out, original_atom_count)

    if optimize == "h_only":
        movable = range(original_atom_count, out.GetNumAtoms())
        out = _optimize_hydrogens_only(out, movable_atom_indices=movable)
        copy_mol_props(base, out)
        _propagate_parent_metadata(out, original_atom_count)

    return out
