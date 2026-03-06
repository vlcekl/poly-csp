from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from poly_csp.config.schema import EndMode, MonomerRepresentation, PolymerKind, Site
from poly_csp.topology.utils import end_caps, residue_label_maps, terminal_meta


_SITES: tuple[Site, ...] = ("C2", "C3", "C6")


@dataclass(frozen=True)
class ResidueTemplateState:
    residue_index: int
    polymer: PolymerKind
    representation: MonomerRepresentation
    end_mode: EndMode
    incoming_link: bool
    outgoing_link: bool
    has_o1: bool
    substituted_sites: tuple[Site, ...]
    left_cap: str | None
    right_cap: str | None
    left_anchor_label: str | None = None
    right_anchor_label: str | None = None


def _normalize_cap(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "h", "hydrogen"}:
        return None
    return text


def _molecule_end_mode(mol: Chem.Mol) -> EndMode:
    raw = mol.GetProp("_poly_csp_end_mode") if mol.HasProp("_poly_csp_end_mode") else "open"
    mode = str(raw).strip().lower()
    if mode not in {"open", "capped", "periodic"}:
        raise ValueError(f"Unsupported _poly_csp_end_mode {raw!r}.")
    return mode  # type: ignore[return-value]


def _molecule_polymer(mol: Chem.Mol) -> PolymerKind:
    if not mol.HasProp("_poly_csp_polymer"):
        raise ValueError("Missing _poly_csp_polymer metadata on molecule.")
    polymer = str(mol.GetProp("_poly_csp_polymer")).strip().lower()
    if polymer not in {"amylose", "cellulose"}:
        raise ValueError(f"Unsupported _poly_csp_polymer {polymer!r}.")
    return polymer  # type: ignore[return-value]


def _molecule_representation(mol: Chem.Mol) -> MonomerRepresentation:
    raw = (
        mol.GetProp("_poly_csp_representation")
        if mol.HasProp("_poly_csp_representation")
        else "anhydro"
    )
    representation = str(raw).strip().lower()
    if representation not in {"anhydro", "natural_oh"}:
        raise ValueError(f"Unsupported _poly_csp_representation {raw!r}.")
    return representation  # type: ignore[return-value]


def _is_site_substituted(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
    mapping: dict[str, int],
) -> bool:
    oxygen_label = f"O{site[1:]}"
    oxygen_idx = mapping.get(oxygen_label)
    if oxygen_idx is None:
        return False
    oxygen = mol.GetAtomWithIdx(int(oxygen_idx))
    same_residue = set(mapping.values())
    for nbr in oxygen.GetNeighbors():
        nbr_idx = int(nbr.GetIdx())
        if nbr_idx in same_residue:
            continue
        if nbr.GetAtomicNum() == 1:
            continue
        if nbr.HasProp("_poly_csp_selector_instance"):
            return True
        if nbr.HasProp("_poly_csp_component"):
            component = nbr.GetProp("_poly_csp_component").strip().lower()
            if component in {"selector", "connector"}:
                return True
    return False


def resolve_residue_template_states(mol: Chem.Mol) -> list[ResidueTemplateState]:
    maps = residue_label_maps(mol)
    polymer = _molecule_polymer(mol)
    representation = _molecule_representation(mol)
    mode = _molecule_end_mode(mol)
    caps = end_caps(mol)
    term_meta = terminal_meta(mol)
    dp = len(maps)

    out: list[ResidueTemplateState] = []
    for residue_index, mapping in enumerate(maps):
        incoming_link = residue_index > 0 or mode == "periodic"
        outgoing_link = residue_index < (dp - 1) or mode == "periodic"
        substituted_sites = tuple(
            site
            for site in _SITES
            if _is_site_substituted(mol, residue_index, site, mapping)
        )

        state = ResidueTemplateState(
            residue_index=residue_index,
            polymer=polymer,
            representation=representation,
            end_mode=mode,
            incoming_link=incoming_link,
            outgoing_link=outgoing_link,
            has_o1="O1" in mapping,
            substituted_sites=substituted_sites,
            left_cap=(
                _normalize_cap(caps.get("left"))
                if residue_index == 0
                else None
            ),
            right_cap=(
                _normalize_cap(caps.get("right"))
                if residue_index == (dp - 1)
                else None
            ),
            left_anchor_label=(
                str(term_meta.get("left_anchor_label"))
                if residue_index == 0 and term_meta.get("left_anchor_label") is not None
                else None
            ),
            right_anchor_label=(
                str(term_meta.get("right_anchor_label"))
                if residue_index == (dp - 1) and term_meta.get("right_anchor_label") is not None
                else None
            ),
        )
        out.append(state)

    return out
