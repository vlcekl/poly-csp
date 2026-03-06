from __future__ import annotations

from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.topology.monomers import GlucoseMonomerTemplate
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.utils import copy_mol_props, residue_label_maps
from poly_csp.config.schema import Site


def residue_atom_global_index(
    residue_index: int,
    monomer_atom_count: int,
    local_atom_index: int,
) -> int:
    """Map residue-local atom index -> polymer-global atom index."""
    if residue_index < 0:
        raise ValueError(f"residue_index must be >= 0, got {residue_index}")
    if monomer_atom_count <= 0:
        raise ValueError(f"monomer_atom_count must be > 0, got {monomer_atom_count}")
    if local_atom_index < 0 or local_atom_index >= monomer_atom_count:
        raise ValueError(f"local_atom_index out of range: {local_atom_index}")
    return residue_index * monomer_atom_count + local_atom_index


def site_to_oxygen_label(site: Site) -> str:
    return f"O{site[1:]}"


def _annotate_selector_atoms(
    rw: Chem.RWMol,
    selector: SelectorTemplate,
    offset: int,
    residue_index: int,
    site: Site,
    instance_id: int,
) -> None:
    for local_idx in range(selector.mol.GetNumAtoms()):
        atom = rw.GetAtomWithIdx(offset + local_idx)
        atom.SetIntProp("_poly_csp_selector_instance", instance_id)
        atom.SetIntProp("_poly_csp_residue_index", residue_index)
        atom.SetProp("_poly_csp_site", site)
        atom.SetIntProp("_poly_csp_selector_local_idx", local_idx)
        atom.SetProp("_poly_csp_selector_name", selector.name)
        atom.SetProp("_poly_csp_linkage_type", selector.linkage_type)
        if local_idx in selector.connector_local_roles:
            atom.SetProp("_poly_csp_component", "connector")
            atom.SetIntProp("_poly_csp_connector_atom", 1)
            atom.SetProp(
                "_poly_csp_connector_role",
                selector.connector_local_roles[local_idx],
            )
        else:
            atom.SetProp("_poly_csp_component", "selector")


def _selector_instance_local_to_global(
    mol: Chem.Mol,
    instance_id: int,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if int(atom.GetIntProp("_poly_csp_selector_instance")) != instance_id:
            continue
        if not atom.HasProp("_poly_csp_selector_local_idx"):
            continue
        mapping[int(atom.GetIntProp("_poly_csp_selector_local_idx"))] = atom.GetIdx()
    return mapping


def _validate_attachment_hydrogen_counts(
    mol: Chem.Mol,
    sugar_o_global: int,
    selector: SelectorTemplate,
    instance_id: int,
) -> None:
    sugar_o = mol.GetAtomWithIdx(int(sugar_o_global))
    if sugar_o.GetTotalNumHs(includeNeighbors=True) != 0:
        raise ValueError(
            "Selector attachment left a hydrogen on the sugar attachment oxygen."
        )

    if not selector.connector_local_roles:
        return

    local_to_global = _selector_instance_local_to_global(mol, instance_id)
    for local_idx, role in selector.connector_local_roles.items():
        if role != "amide_n":
            continue
        atom_idx = local_to_global.get(int(local_idx))
        if atom_idx is None:
            raise ValueError("Attached selector is missing the connector amide nitrogen.")
        amide_n = mol.GetAtomWithIdx(int(atom_idx))
        if amide_n.GetTotalNumHs(includeNeighbors=True) != 1:
            raise ValueError(
                "Selector attachment did not preserve the carbamate NH hydrogen count."
            )


def residue_label_global_index(mol: Chem.Mol, residue_index: int, label: str) -> int:
    maps = residue_label_maps(mol)
    if residue_index < 0 or residue_index >= len(maps):
        raise ValueError(f"residue_index {residue_index} out of range [0, {len(maps)})")
    mapping = maps[residue_index]
    if label not in mapping:
        raise ValueError(f"Label {label!r} is unavailable for residue {residue_index}.")
    return int(mapping[label])


def attach_selector(
    mol_polymer: Chem.Mol,
    template: GlucoseMonomerTemplate,
    residue_index: int,
    site: Site,
    selector: SelectorTemplate,
    mode: Literal["bond_from_OH_oxygen"] = "bond_from_OH_oxygen",
    linkage_type: str | None = None,
    place_coords: bool = True,
) -> Chem.Mol:
    """Attach selector to polymer with optional deterministic coordinate placement.

    The topology step always performs bond creation and metadata tagging.
    When ``place_coords`` is true and input coordinates exist, selector
    coordinates are also placed through ``structure.alignment``.
    """
    if mode != "bond_from_OH_oxygen":
        raise ValueError(f"Unsupported attachment mode: {mode!r}")

    dp = (
        int(mol_polymer.GetIntProp("_poly_csp_dp"))
        if mol_polymer.HasProp("_poly_csp_dp")
        else (mol_polymer.GetNumAtoms() // template.mol.GetNumAtoms())
    )
    if residue_index < 0 or residue_index >= dp:
        raise ValueError(f"residue_index {residue_index} out of range [0, {dp})")

    oxygen_label = site_to_oxygen_label(site)
    if oxygen_label not in template.site_idx:
        raise ValueError(f"Site {site} is not available in template.site_idx")
    resolved_linkage_type = linkage_type or selector.linkage_type

    sugar_o_global = residue_label_global_index(mol_polymer, residue_index, oxygen_label)

    existing_coords = None
    if mol_polymer.GetNumConformers() > 0:
        existing_coords = np.asarray(
            mol_polymer.GetConformer(0).GetPositions(), dtype=float
        ).reshape((-1, 3))

    rw = Chem.RWMol(Chem.CombineMols(mol_polymer, selector.mol))
    offset = mol_polymer.GetNumAtoms()
    attach_global = offset + selector.attach_atom_idx

    prev_count = (
        int(mol_polymer.GetIntProp("_poly_csp_selector_count"))
        if mol_polymer.HasProp("_poly_csp_selector_count")
        else 0
    )
    instance_id = prev_count + 1
    _annotate_selector_atoms(
        rw=rw,
        selector=selector,
        offset=offset,
        residue_index=residue_index,
        site=site,
        instance_id=instance_id,
    )

    rw.AddBond(sugar_o_global, attach_global, Chem.BondType.SINGLE)

    if selector.attach_dummy_idx is not None:
        rw.RemoveAtom(offset + selector.attach_dummy_idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    copy_mol_props(mol_polymer, mol)
    mol.SetIntProp("_poly_csp_selector_count", instance_id)
    _validate_attachment_hydrogen_counts(
        mol=mol,
        sugar_o_global=sugar_o_global,
        selector=selector,
        instance_id=instance_id,
    )

    if not place_coords:
        # Topology-only mode: strip coordinates so downstream geometry assembly
        # can assign them deterministically in the structure domain.
        mol.RemoveAllConformers()
        return mol

    if place_coords and existing_coords is not None and selector.mol.GetNumConformers() > 0:
        from poly_csp.structure.alignment import place_selector_with_ideal_linkage

        selector_coords = place_selector_with_ideal_linkage(
            existing_coords=existing_coords,
            mol_polymer=mol_polymer,
            residue_index=residue_index,
            site=site,
            selector=selector,
            linkage_type=resolved_linkage_type,
        )
        merged = np.concatenate([existing_coords, selector_coords], axis=0)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(merged):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

    return mol
