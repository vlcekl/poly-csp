from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from poly_csp.config.schema import MonomerRepresentation, PolymerKind
from poly_csp.structure.hydrogens import _optimize_hydrogens_only
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.residue_state import ResidueTemplateState


@dataclass(frozen=True)
class ExplicitResidueTemplate:
    mol: Chem.Mol
    heavy_label_to_idx: dict[str, int]
    atom_name_by_idx: dict[int, str]
    hydrogen_parent_label: dict[int, str]


def _backbone_hydrogen_name(parent_label: str, n_h: int, serial: int) -> str:
    if parent_label.startswith("O"):
        return f"HO{parent_label[1:]}"
    if parent_label == "C6":
        return f"H6{serial}"
    if parent_label.startswith("C"):
        suffix = parent_label[1:]
        return f"H{suffix}" if n_h == 1 else f"H{suffix}{serial}"
    return f"H{serial}"


def _build_base_explicit_template(
    polymer: PolymerKind,
    representation: MonomerRepresentation,
) -> ExplicitResidueTemplate:
    template = make_glucose_template(polymer, monomer_representation=representation)
    explicit = Chem.AddHs(Chem.Mol(template.mol), addCoords=True)
    movable = [atom.GetIdx() for atom in explicit.GetAtoms() if atom.GetAtomicNum() == 1]
    explicit = _optimize_hydrogens_only(explicit, movable_atom_indices=movable)

    heavy_label_to_idx = {str(label): int(idx) for label, idx in template.atom_idx.items()}
    atom_name_by_idx: dict[int, str] = {
        int(idx): str(label) for label, idx in heavy_label_to_idx.items()
    }
    reverse_heavy = {int(idx): str(label) for label, idx in heavy_label_to_idx.items()}

    hydrogen_parent_label: dict[int, str] = {}
    by_parent: dict[str, list[int]] = {}
    for atom in explicit.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            continue
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        parent_label = reverse_heavy.get(parent_idx)
        if parent_label is None:
            continue
        hydrogen_parent_label[int(atom.GetIdx())] = parent_label
        by_parent.setdefault(parent_label, []).append(int(atom.GetIdx()))

    for parent_label, h_indices in by_parent.items():
        ordered = sorted(h_indices)
        for serial, h_idx in enumerate(ordered, start=1):
            atom_name_by_idx[h_idx] = _backbone_hydrogen_name(
                parent_label,
                len(ordered),
                serial,
            )

    return ExplicitResidueTemplate(
        mol=explicit,
        heavy_label_to_idx=heavy_label_to_idx,
        atom_name_by_idx=atom_name_by_idx,
        hydrogen_parent_label=hydrogen_parent_label,
    )


_BASE_TEMPLATE_CACHE: dict[tuple[str, str], ExplicitResidueTemplate] = {}


def load_explicit_backbone_template(
    polymer: PolymerKind,
    representation: MonomerRepresentation,
) -> ExplicitResidueTemplate:
    """Return the complete explicit-H residue template for a backbone representation."""
    key = (str(polymer), str(representation))
    cached = _BASE_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    built = _build_base_explicit_template(polymer, representation)
    _BASE_TEMPLATE_CACHE[key] = built
    return built


def _remove_indices_from_template(
    template: ExplicitResidueTemplate,
    remove_indices: set[int],
) -> ExplicitResidueTemplate:
    if not remove_indices:
        return template

    rw = Chem.RWMol(template.mol)
    for atom_idx in sorted(remove_indices, reverse=True):
        rw.RemoveAtom(int(atom_idx))
    out = rw.GetMol()
    Chem.SanitizeMol(out)

    index_map: dict[int, int] = {}
    shift = 0
    removed = sorted(remove_indices)
    next_removed = 0
    for old_idx in range(template.mol.GetNumAtoms()):
        if next_removed < len(removed) and old_idx == removed[next_removed]:
            shift += 1
            next_removed += 1
            continue
        index_map[old_idx] = old_idx - shift

    heavy_label_to_idx = {
        label: index_map[idx]
        for label, idx in template.heavy_label_to_idx.items()
        if idx not in remove_indices
    }
    atom_name_by_idx = {
        index_map[idx]: name
        for idx, name in template.atom_name_by_idx.items()
        if idx not in remove_indices
    }
    hydrogen_parent_label = {
        index_map[idx]: parent_label
        for idx, parent_label in template.hydrogen_parent_label.items()
        if idx not in remove_indices
    }
    return ExplicitResidueTemplate(
        mol=out,
        heavy_label_to_idx=heavy_label_to_idx,
        atom_name_by_idx=atom_name_by_idx,
        hydrogen_parent_label=hydrogen_parent_label,
    )


def _hydrogen_indices_for_parent(
    template: ExplicitResidueTemplate,
    parent_label: str,
) -> set[int]:
    return {
        int(atom_idx)
        for atom_idx, label in template.hydrogen_parent_label.items()
        if label == parent_label
    }


def build_residue_variant(
    base_template: ExplicitResidueTemplate,
    state: ResidueTemplateState,
) -> ExplicitResidueTemplate:
    """Prune a complete explicit-H residue template into the exact residue-state variant.

    The geometry rule is deliberate: build the full chemically complete residue
    first, then remove atoms or hydrogens that the resolved topology state says
    should be absent. The variant geometry is never embedded from an already
    pruned graph.
    """
    remove_indices: set[int] = set()

    if not state.has_o1 and "O1" in base_template.heavy_label_to_idx:
        remove_indices.add(int(base_template.heavy_label_to_idx["O1"]))
        remove_indices.update(_hydrogen_indices_for_parent(base_template, "O1"))

    if state.left_cap and state.left_anchor_label == "O1":
        remove_indices.update(_hydrogen_indices_for_parent(base_template, "O1"))

    if state.left_cap and state.left_anchor_label == "C1":
        remove_indices.update(_hydrogen_indices_for_parent(base_template, "C1"))

    if state.outgoing_link or (state.right_cap and state.right_anchor_label == "O4"):
        remove_indices.update(_hydrogen_indices_for_parent(base_template, "O4"))

    for site in state.substituted_sites:
        oxygen_label = f"O{site[1:]}"
        remove_indices.update(_hydrogen_indices_for_parent(base_template, oxygen_label))

    return _remove_indices_from_template(base_template, remove_indices)
