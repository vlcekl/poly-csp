from __future__ import annotations

import json
from typing import Literal

import numpy as np
from rdkit import Chem
from poly_csp.topology.utils import (
    copy_mol_props,
    removed_old_indices,
    residue_label_maps,
    set_json_prop,
    set_removed_old_indices,
    set_residue_label_maps,
)

EndMode = Literal["open", "capped", "periodic"]

def _remove_atom(
    rw: Chem.RWMol,
    maps: list[dict[str, int]],
    remove_idx: int,
) -> list[dict[str, int]]:
    rw.RemoveAtom(int(remove_idx))

    out_maps: list[dict[str, int]] = []
    for mapping in maps:
        new_map: dict[str, int] = {}
        for label, idx in mapping.items():
            if idx == remove_idx:
                continue
            new_map[label] = int(idx - 1) if idx > remove_idx else int(idx)
        out_maps.append(new_map)
    return out_maps


def _apply_periodic_topology(
    rw: Chem.RWMol,
    maps: list[dict[str, int]],
    representation: str,
    removed_old: list[int],
) -> tuple[list[dict[str, int]], list[int], dict[str, object]]:
    periodic_meta: dict[str, object] = {"removed_labels": []}

    if representation == "natural_oh" and "O1" in maps[0]:
        o1_idx = int(maps[0]["O1"])
        maps = _remove_atom(rw=rw, maps=maps, remove_idx=o1_idx)
        removed_old.append(o1_idx)
        periodic_meta["removed_labels"] = ["res0:O1"]

    left_c1 = int(maps[0]["C1"])
    right_o4 = int(maps[-1]["O4"])
    if rw.GetBondBetweenAtoms(left_c1, right_o4) is None:
        rw.AddBond(right_o4, left_c1, Chem.BondType.SINGLE)
    periodic_meta["closure_bond"] = [right_o4, left_c1]

    return maps, removed_old, periodic_meta


def apply_terminal_mode(
    mol: Chem.Mol,
    mode: EndMode,
    caps: dict[str, str] | None,
    representation: str,
) -> Chem.Mol:
    """
    Apply topology-domain terminal chemistry metadata prior to structure building.

    Modes:
    - `open`: no topology edits.
    - `capped`: record deterministic cap intent and anchor metadata.
    - `periodic`: connect right-end O4 to left-end C1 (and remove res0 O1 for natural_oh).
    """
    if mode not in {"open", "capped", "periodic"}:
        raise ValueError(f"Unsupported end mode {mode!r}")

    cap_cfg = {str(k): str(v) for k, v in (caps or {}).items()}
    maps = residue_label_maps(mol)
    removed_old = removed_old_indices(mol)
    rw = Chem.RWMol(mol)
    terminal_meta: dict[str, object] = {}
    cap_indices: dict[str, list[int]] = {"left": [], "right": []}

    if mode == "periodic":
        maps, removed_old, terminal_meta = _apply_periodic_topology(
            rw=rw,
            maps=maps,
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
        terminal_meta = {
            "left_anchor_label": left_anchor_label,
            "left_anchor_idx": left_anchor,
            "right_anchor_label": "O4",
            "right_anchor_idx": right_anchor,
        }

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    copy_mol_props(mol, out)
    set_residue_label_maps(out, maps)
    set_removed_old_indices(out, removed_old)

    out.SetProp("_poly_csp_end_mode", mode)
    out.SetProp("_poly_csp_representation", representation)
    set_json_prop(out, "_poly_csp_end_caps_json", cap_cfg)
    out.SetBoolProp("_poly_csp_terminal_topology_pending", False)
    set_json_prop(out, "_poly_csp_terminal_meta_json", terminal_meta)
    set_json_prop(out, "_poly_csp_terminal_cap_indices_json", cap_indices)

    return out
