from __future__ import annotations

import json
from typing import Literal

from rdkit import Chem

EndMode = Literal["open", "capped", "periodic"]


def apply_terminal_mode(
    mol: Chem.Mol,
    mode: EndMode,
    caps: dict[str, str] | None,
    representation: str,
) -> Chem.Mol:
    """
    Apply terminal policy metadata to a polymer.

    Stage-2.5 implementation currently keeps topology unchanged and records mode/caps
    deterministically. Chemistry-changing terminal edits are planned for later stages.
    """
    if mode not in {"open", "capped", "periodic"}:
        raise ValueError(f"Unsupported end mode {mode!r}")

    out = Chem.Mol(mol)
    out.SetProp("_poly_csp_end_mode", mode)
    out.SetProp("_poly_csp_representation", representation)
    out.SetProp("_poly_csp_end_caps_json", json.dumps(caps or {}))

    if mode in {"capped", "periodic"}:
        out.SetBoolProp("_poly_csp_terminal_topology_pending", True)
    else:
        out.SetBoolProp("_poly_csp_terminal_topology_pending", False)
    return out
