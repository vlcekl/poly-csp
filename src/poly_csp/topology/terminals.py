from __future__ import annotations

import json
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.topology.utils import (
    copy_mol_props,
    removed_old_indices,
    residue_label_maps,
    set_json_prop,
    set_removed_old_indices,
    set_residue_label_maps,
)

EndMode = Literal["open", "capped", "periodic"]

_NO_CAP = {"none", "h", "hydrogen"}
_METHYL_CAP = {"methyl", "methoxy"}
_HYDROXYL_CAP = {"hydroxyl", "oh"}
_ACETYL_CAP = {"acetyl"}

def _coords_from_mol(mol: Chem.Mol) -> np.ndarray | None:
    if mol.GetNumConformers() == 0:
        return None
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))


def _set_coords(mol: Chem.Mol, coords: np.ndarray | None) -> None:
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


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _orthonormal_basis(main_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = _normalize(main_dir)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(u, trial))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    n = _normalize(np.cross(u, trial))
    b = _normalize(np.cross(u, n))
    return u, n, b


def _anchor_direction(rw: Chem.RWMol, coords: np.ndarray | None, anchor_idx: int) -> np.ndarray:
    if coords is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    anchor = coords[anchor_idx]
    atom = rw.GetAtomWithIdx(anchor_idx)
    pieces: list[np.ndarray] = []
    for nbr in atom.GetNeighbors():
        vec = anchor - coords[nbr.GetIdx()]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            pieces.append(vec / norm)
    if not pieces:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return _normalize(np.sum(np.asarray(pieces, dtype=float), axis=0))


def _choose_attachment_direction(
    rw: Chem.RWMol,
    coords: np.ndarray | None,
    anchor_idx: int,
    bond_len: float,
) -> np.ndarray:
    base = _anchor_direction(rw, coords, anchor_idx)
    if coords is None:
        return base

    anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)
    u, n, b = _orthonormal_basis(base)
    candidates = [
        u,
        -u,
        n,
        -n,
        b,
        -b,
        _normalize(u + n),
        _normalize(u - n),
        _normalize(u + b),
        _normalize(u - b),
    ]

    excluded = {int(anchor_idx)}
    for nbr in rw.GetAtomWithIdx(int(anchor_idx)).GetNeighbors():
        excluded.add(int(nbr.GetIdx()))

    best = candidates[0]
    best_score = float("-inf")
    for cand in candidates:
        trial = anchor_pos + float(bond_len) * cand
        dmin = float("inf")
        for i in range(coords.shape[0]):
            if i in excluded:
                continue
            d = float(np.linalg.norm(trial - coords[i]))
            if d < dmin:
                dmin = d
        if dmin > best_score:
            best_score = dmin
            best = cand
    return _normalize(best)


def _remove_atom(
    rw: Chem.RWMol,
    maps: list[dict[str, int]],
    coords: np.ndarray | None,
    remove_idx: int,
) -> tuple[list[dict[str, int]], np.ndarray | None]:
    rw.RemoveAtom(int(remove_idx))
    if coords is not None:
        coords = np.delete(coords, int(remove_idx), axis=0)

    out_maps: list[dict[str, int]] = []
    for mapping in maps:
        new_map: dict[str, int] = {}
        for label, idx in mapping.items():
            if idx == remove_idx:
                continue
            new_map[label] = int(idx - 1) if idx > remove_idx else int(idx)
        out_maps.append(new_map)
    return out_maps, coords


def _add_atom_with_coords(
    rw: Chem.RWMol,
    coords: np.ndarray | None,
    atomic_num: int,
    position: np.ndarray,
) -> tuple[int, np.ndarray | None]:
    idx = rw.AddAtom(Chem.Atom(int(atomic_num)))
    if coords is not None:
        xyz = np.asarray(position, dtype=float).reshape((1, 3))
        coords = np.concatenate([coords, xyz], axis=0)
    return int(idx), coords


def _canonical_cap_name(cap: str | None) -> str:
    if cap is None:
        return "none"
    return str(cap).strip().lower()


def _apply_cap(
    rw: Chem.RWMol,
    coords: np.ndarray | None,
    anchor_idx: int,
    cap_name: str,
) -> tuple[np.ndarray | None, list[int]]:
    cap = _canonical_cap_name(cap_name)
    if cap in _NO_CAP:
        return coords, []

    anchor = rw.GetAtomWithIdx(int(anchor_idx))
    anchor_z = int(anchor.GetAtomicNum())
    if coords is None:
        anchor_pos = np.zeros((3,), dtype=float)
    else:
        anchor_pos = np.asarray(coords[int(anchor_idx)], dtype=float)

    if cap in _METHYL_CAP:
        bond_len = 1.43 if anchor_z == 8 else 1.50
        direction = _choose_attachment_direction(
            rw=rw, coords=coords, anchor_idx=int(anchor_idx), bond_len=bond_len
        )
        c_pos = anchor_pos + bond_len * direction
        c_idx, coords = _add_atom_with_coords(rw, coords, atomic_num=6, position=c_pos)
        rw.AddBond(int(anchor_idx), c_idx, Chem.BondType.SINGLE)
        return coords, [c_idx]

    if cap in _HYDROXYL_CAP:
        if anchor_z != 6:
            raise ValueError("Hydroxyl cap is only supported on carbon anchor atoms.")
        direction = _choose_attachment_direction(
            rw=rw, coords=coords, anchor_idx=int(anchor_idx), bond_len=1.43
        )
        o_pos = anchor_pos + 1.43 * direction
        o_idx, coords = _add_atom_with_coords(rw, coords, atomic_num=8, position=o_pos)
        rw.AddBond(int(anchor_idx), o_idx, Chem.BondType.SINGLE)
        return coords, [o_idx]

    if cap in _ACETYL_CAP:
        bond_len = 1.43 if anchor_z == 8 else 1.50
        direction = _choose_attachment_direction(
            rw=rw, coords=coords, anchor_idx=int(anchor_idx), bond_len=bond_len
        )
        c_pos = anchor_pos + bond_len * direction
        c_idx, coords = _add_atom_with_coords(rw, coords, atomic_num=6, position=c_pos)
        rw.AddBond(int(anchor_idx), c_idx, Chem.BondType.SINGLE)

        u, n, _ = _orthonormal_basis(direction)
        o_pos_a = c_pos + 1.23 * (0.5 * u + 0.866 * n)
        m_pos_a = c_pos + 1.52 * (0.5 * u - 0.866 * n)
        o_pos_b = c_pos + 1.23 * (0.5 * u - 0.866 * n)
        m_pos_b = c_pos + 1.52 * (0.5 * u + 0.866 * n)

        if coords is None:
            o_pos, m_pos = o_pos_a, m_pos_a
        else:
            existing = coords
            score_a = min(
                float(np.linalg.norm(o_pos_a - p)) + float(np.linalg.norm(m_pos_a - p))
                for p in existing
            )
            score_b = min(
                float(np.linalg.norm(o_pos_b - p)) + float(np.linalg.norm(m_pos_b - p))
                for p in existing
            )
            o_pos, m_pos = (o_pos_a, m_pos_a) if score_a >= score_b else (o_pos_b, m_pos_b)

        o_idx, coords = _add_atom_with_coords(rw, coords, atomic_num=8, position=o_pos)
        m_idx, coords = _add_atom_with_coords(rw, coords, atomic_num=6, position=m_pos)
        rw.AddBond(c_idx, o_idx, Chem.BondType.DOUBLE)
        rw.AddBond(c_idx, m_idx, Chem.BondType.SINGLE)
        return coords, [c_idx, o_idx, m_idx]

    raise ValueError(
        f"Unsupported cap type {cap_name!r}. Supported: none, methyl/methoxy, hydroxyl, acetyl."
    )


def _apply_periodic_topology(
    rw: Chem.RWMol,
    maps: list[dict[str, int]],
    coords: np.ndarray | None,
    representation: str,
    removed_old: list[int],
) -> tuple[list[dict[str, int]], np.ndarray | None, list[int], dict[str, object]]:
    periodic_meta: dict[str, object] = {"removed_labels": []}

    if representation == "natural_oh" and "O1" in maps[0]:
        o1_idx = int(maps[0]["O1"])
        maps, coords = _remove_atom(rw=rw, maps=maps, coords=coords, remove_idx=o1_idx)
        removed_old.append(o1_idx)
        periodic_meta["removed_labels"] = ["res0:O1"]

    left_c1 = int(maps[0]["C1"])
    right_o4 = int(maps[-1]["O4"])
    if rw.GetBondBetweenAtoms(left_c1, right_o4) is None:
        rw.AddBond(right_o4, left_c1, Chem.BondType.SINGLE)
    periodic_meta["closure_bond"] = [right_o4, left_c1]

    return maps, coords, removed_old, periodic_meta


def apply_terminal_mode(
    mol: Chem.Mol,
    mode: EndMode,
    caps: dict[str, str] | None,
    representation: str,
) -> Chem.Mol:
    """
    Apply terminal policy chemistry + metadata.

    Modes:
    - `open`: no topology edits.
    - `capped`: apply deterministic configured caps at left/right termini.
    - `periodic`: connect right-end O4 to left-end C1 (and remove res0 O1 for natural_oh).
    """
    if mode not in {"open", "capped", "periodic"}:
        raise ValueError(f"Unsupported end mode {mode!r}")

    cap_cfg = {str(k): str(v) for k, v in (caps or {}).items()}
    maps = residue_label_maps(mol)
    removed_old = removed_old_indices(mol)
    coords = _coords_from_mol(mol)

    rw = Chem.RWMol(mol)
    terminal_meta: dict[str, object] = {}
    cap_indices: dict[str, list[int]] = {"left": [], "right": []}

    if mode == "periodic":
        maps, coords, removed_old, terminal_meta = _apply_periodic_topology(
            rw=rw,
            maps=maps,
            coords=coords,
            representation=str(representation),
            removed_old=removed_old,
        )

    elif mode == "capped":
        left_anchor_label = "O1" if "O1" in maps[0] else "C1"
        left_anchor = int(maps[0][left_anchor_label])
        right_anchor = int(maps[-1]["O4"])

        if "left" not in cap_cfg or "right" not in cap_cfg:
            raise ValueError(
                "Capped mode requires end_caps with both 'left' and 'right' keys."
            )

        coords, left_added = _apply_cap(
            rw=rw,
            coords=coords,
            anchor_idx=left_anchor,
            cap_name=cap_cfg["left"],
        )
        coords, right_added = _apply_cap(
            rw=rw,
            coords=coords,
            anchor_idx=right_anchor,
            cap_name=cap_cfg["right"],
        )
        cap_indices["left"] = left_added
        cap_indices["right"] = right_added
        terminal_meta = {
            "left_anchor_label": left_anchor_label,
            "left_anchor_idx": left_anchor,
            "right_anchor_label": "O4",
            "right_anchor_idx": right_anchor,
        }

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    copy_mol_props(mol, out)
    _set_coords(out, coords)
    set_residue_label_maps(out, maps)
    set_removed_old_indices(out, removed_old)

    out.SetProp("_poly_csp_end_mode", mode)
    out.SetProp("_poly_csp_representation", representation)
    set_json_prop(out, "_poly_csp_end_caps_json", cap_cfg)
    out.SetBoolProp("_poly_csp_terminal_topology_pending", False)
    set_json_prop(out, "_poly_csp_terminal_meta_json", terminal_meta)
    set_json_prop(out, "_poly_csp_terminal_cap_indices_json", cap_indices)

    return out
