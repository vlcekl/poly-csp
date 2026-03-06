from __future__ import annotations

from enum import Enum
from typing import Dict, Set

from rdkit import Chem


class ComponentTag(str, Enum):
    BACKBONE = "backbone"
    SELECTOR = "selector"
    CONNECTOR = "connector"


def _tag_for_atom(atom: Chem.Atom) -> ComponentTag:
    if atom.HasProp("_poly_csp_component"):
        raw = atom.GetProp("_poly_csp_component").strip().lower()
        if raw in {t.value for t in ComponentTag}:
            return ComponentTag(raw)
    if atom.HasProp("_poly_csp_selector_instance"):
        return ComponentTag.SELECTOR
    if atom.HasProp("_poly_csp_connector_atom"):
        return ComponentTag.CONNECTOR
    return ComponentTag.BACKBONE


def build_atom_map(mol: Chem.Mol) -> Dict[int, ComponentTag]:
    """Map every atom index in *mol* to its topological component tag."""
    return {atom.GetIdx(): _tag_for_atom(atom) for atom in mol.GetAtoms()}


def attachment_instance_maps(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    """Return {instance_id: {selector_local_idx: global_idx}} for all attached atoms."""
    mappings: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        instance = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        mappings.setdefault(instance, {})[local] = atom.GetIdx()
    return mappings


def selector_instance_maps(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    """Return {instance_id: {selector_local_idx: global_idx}} for selector-core atoms."""
    mappings: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if _tag_for_atom(atom) is not ComponentTag.SELECTOR:
            continue
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        instance = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        mappings.setdefault(instance, {})[local] = atom.GetIdx()
    return mappings


def connector_instance_maps(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    """Return {instance_id: {selector_local_idx: global_idx}} for connector atoms."""
    mappings: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if _tag_for_atom(atom) is not ComponentTag.CONNECTOR:
            continue
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        instance = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        mappings.setdefault(instance, {})[local] = atom.GetIdx()
    return mappings


def _indices_by_component(mol: Chem.Mol, tag: ComponentTag) -> Set[int]:
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if _tag_for_atom(atom) is tag
    }


def backbone_indices(mol: Chem.Mol) -> Set[int]:
    return _indices_by_component(mol, ComponentTag.BACKBONE)


def selector_indices(mol: Chem.Mol) -> Set[int]:
    return _indices_by_component(mol, ComponentTag.SELECTOR)


def connector_indices(mol: Chem.Mol) -> Set[int]:
    return _indices_by_component(mol, ComponentTag.CONNECTOR)
