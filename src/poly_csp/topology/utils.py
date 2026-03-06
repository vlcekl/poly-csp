# poly_csp/topology/utils.py
"""Shared chemistry utilities used across multiple modules."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from rdkit import Chem


def copy_mol_props(src: Chem.Mol, dst: Chem.Mol) -> None:
    """Copy all private/public properties from *src* to *dst*."""
    props = src.GetPropsAsDict(includePrivate=True, includeComputed=False)
    for key, value in props.items():
        if isinstance(value, bool):
            dst.SetBoolProp(key, bool(value))
        elif isinstance(value, int):
            dst.SetIntProp(key, int(value))
        elif isinstance(value, float):
            dst.SetDoubleProp(key, float(value))
        else:
            dst.SetProp(key, str(value))


def json_prop(mol: Chem.Mol, key: str, default: Any) -> Any:
    """Return a parsed JSON property or *default* when absent."""
    if not mol.HasProp(key):
        return default
    return json.loads(mol.GetProp(key))


def set_json_prop(mol: Chem.Mol, key: str, value: Any) -> None:
    """Serialize *value* into a JSON molecule property."""
    mol.SetProp(key, json.dumps(value))


def residue_label_maps(mol: Chem.Mol) -> List[Dict[str, int]]:
    """Parse the residue label map metadata from the molecule."""
    payload = json_prop(mol, "_poly_csp_residue_label_map_json", None)
    if payload is None:
        raise ValueError("Missing _poly_csp_residue_label_map_json metadata on molecule.")
    if not isinstance(payload, list):
        raise ValueError("Invalid residue label map metadata format.")
    maps: List[Dict[str, int]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Invalid residue label map entry.")
        maps.append({str(k): int(v) for k, v in item.items()})
    return maps


def set_residue_label_maps(mol: Chem.Mol, maps: List[Dict[str, int]]) -> None:
    """Write residue label map metadata onto the molecule."""
    set_json_prop(mol, "_poly_csp_residue_label_map_json", maps)


def removed_old_indices(mol: Chem.Mol) -> List[int]:
    """Parse removed-old-indices metadata."""
    payload = json_prop(mol, "_poly_csp_removed_old_indices_json", [])
    if not isinstance(payload, list):
        raise ValueError("Invalid _poly_csp_removed_old_indices_json metadata.")
    return [int(x) for x in payload]


def set_removed_old_indices(mol: Chem.Mol, removed: List[int]) -> None:
    """Write removed-old-indices metadata."""
    set_json_prop(mol, "_poly_csp_removed_old_indices_json", sorted(set(removed)))


def end_caps(mol: Chem.Mol) -> Dict[str, str]:
    """Return configured terminal caps."""
    payload = json_prop(mol, "_poly_csp_end_caps_json", {})
    if not isinstance(payload, dict):
        raise ValueError("Invalid _poly_csp_end_caps_json metadata.")
    return {str(k): str(v) for k, v in payload.items()}


def terminal_meta(mol: Chem.Mol) -> Dict[str, Any]:
    """Return parsed terminal metadata."""
    payload = json_prop(mol, "_poly_csp_terminal_meta_json", {})
    if not isinstance(payload, dict):
        raise ValueError("Invalid _poly_csp_terminal_meta_json metadata.")
    return dict(payload)


def terminal_cap_indices(mol: Chem.Mol) -> Dict[str, List[int]]:
    """Return atom indices for left/right terminal caps."""
    payload = json_prop(
        mol,
        "_poly_csp_terminal_cap_indices_json",
        {"left": [], "right": []},
    )
    if not isinstance(payload, dict):
        raise ValueError("Invalid _poly_csp_terminal_cap_indices_json metadata.")
    out: Dict[str, List[int]] = {"left": [], "right": []}
    for side, indices in payload.items():
        if side not in out:
            continue
        if not isinstance(indices, list):
            raise ValueError("Invalid terminal cap index list.")
        out[side] = [int(idx) for idx in indices]
    return out


def heavy_atom_mask(mol: Chem.Mol) -> np.ndarray:
    """Boolean mask: True for heavy atoms (Z > 1)."""
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, atom in enumerate(mol.GetAtoms()):
        mask[i] = atom.GetAtomicNum() > 1
    return mask


def backbone_heavy_indices(mol: Chem.Mol) -> List[int]:
    """Return global indices of heavy backbone atoms (not selectors)."""
    idx: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.HasProp("_poly_csp_selector_instance"):
            continue
        idx.append(atom.GetIdx())
    return idx


def coords_from_mol(mol: Chem.Mol) -> np.ndarray | None:
    """Extract (N,3) coordinate array from the first conformer, or None."""
    if mol.GetNumConformers() == 0:
        return None
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))


def set_coords(mol: Chem.Mol, coords: np.ndarray | None) -> None:
    """Assign coordinates to the molecule, replacing any existing conformer."""
    from rdkit.Geometry import Point3D

    if coords is None:
        mol.RemoveAllConformers()
        return
    xyz = np.asarray(coords, dtype=float).reshape((-1, 3))
    if xyz.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Coordinate size mismatch: {xyz.shape[0]} vs {mol.GetNumAtoms()} atoms."
        )
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
