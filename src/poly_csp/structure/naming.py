from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdchem

from poly_csp.topology.utils import residue_label_maps, terminal_cap_indices


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AtomManifestEntry:
    atom_index: int
    parent_heavy_index: int
    component: str
    residue_index: int | None
    residue_label: str | None
    site: str | None
    selector_instance: int | None
    selector_local_idx: int | None
    connector_role: str | None
    atom_name: str
    canonical_name: str
    source: str


def _component_for_atom(atom: Chem.Atom) -> str:
    if atom.HasProp("_poly_csp_component"):
        component = atom.GetProp("_poly_csp_component").strip().lower()
        if component in {"backbone", "selector", "connector"}:
            return component
    if atom.HasProp("_poly_csp_selector_instance"):
        return "selector"
    if atom.HasProp("_poly_csp_connector_atom"):
        return "connector"
    return "backbone"


def _parent_heavy_index(atom: Chem.Atom) -> int:
    if atom.GetAtomicNum() != 1:
        return int(atom.GetIdx())
    if atom.HasProp("_poly_csp_parent_heavy_idx"):
        return int(atom.GetIntProp("_poly_csp_parent_heavy_idx"))
    if atom.GetDegree() == 1:
        return int(atom.GetNeighbors()[0].GetIdx())
    return int(atom.GetIdx())


def _backbone_hydrogen_name(parent_label: str, n_h: int, serial: int) -> str:
    if parent_label.startswith("O"):
        return f"HO{parent_label[1:]}"
    if parent_label == "C6":
        return f"H6{serial}"
    if parent_label.startswith("C"):
        suffix = parent_label[1:]
        return f"H{suffix}" if n_h == 1 else f"H{suffix}{serial}"
    return f"H{serial}"


def _pdb_atom_name(atom_name: str) -> str:
    if len(atom_name) < 4:
        return f" {atom_name:<3s}"
    return atom_name[:4]


def _set_pdb_info(
    atom: Chem.Atom,
    *,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    chain_id: str,
    is_hetero: bool,
) -> None:
    info = rdchem.AtomPDBResidueInfo()
    info.SetName(_pdb_atom_name(atom_name))
    info.SetResidueName(str(residue_name)[:3].upper())
    info.SetResidueNumber(int(residue_number))
    info.SetChainId(str(chain_id)[:1] or "A")
    info.SetIsHeteroAtom(bool(is_hetero))
    info.SetOccupancy(1.0)
    info.SetTempFactor(0.0)
    atom.SetPDBResidueInfo(info)


def build_atom_manifest(mol: Chem.Mol) -> list[AtomManifestEntry]:
    residue_for_atom: dict[int, int] = {}
    residue_label_for_atom: dict[int, str] = {}
    n_residues = 1
    if mol.HasProp("_poly_csp_residue_label_map_json"):
        maps = residue_label_maps(mol)
        n_residues = max(1, len(maps))
        for residue_index, mapping in enumerate(maps):
            for label, atom_idx in mapping.items():
                residue_for_atom[int(atom_idx)] = int(residue_index)
                residue_label_for_atom[int(atom_idx)] = str(label)

    cap_heavy_by_side = terminal_cap_indices(mol)
    cap_side_for_atom: dict[int, str] = {}
    cap_heavy_serial: dict[int, int] = {}
    for side in ("left", "right"):
        ordered = sorted(int(idx) for idx in cap_heavy_by_side.get(side, []))
        for serial, atom_idx in enumerate(ordered, start=1):
            cap_side_for_atom[atom_idx] = side
            cap_heavy_serial[atom_idx] = serial

    hydrogen_neighbors_by_parent: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
            continue
        parent_idx = int(atom.GetNeighbors()[0].GetIdx())
        hydrogen_neighbors_by_parent.setdefault(parent_idx, []).append(int(atom.GetIdx()))
    for indices in hydrogen_neighbors_by_parent.values():
        indices.sort()

    manifest: list[AtomManifestEntry] = []
    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        parent_heavy_index = _parent_heavy_index(atom)
        parent_atom = mol.GetAtomWithIdx(parent_heavy_index)
        component = _component_for_atom(parent_atom if atom.GetAtomicNum() == 1 else atom)

        residue_index = (
            int(parent_atom.GetIntProp("_poly_csp_residue_index"))
            if parent_atom.HasProp("_poly_csp_residue_index")
            else residue_for_atom.get(parent_heavy_index)
        )
        residue_label = (
            parent_atom.GetProp("_poly_csp_residue_label")
            if parent_atom.HasProp("_poly_csp_residue_label")
            else residue_label_for_atom.get(parent_heavy_index)
        )
        site = (
            parent_atom.GetProp("_poly_csp_site")
            if parent_atom.HasProp("_poly_csp_site")
            else None
        )
        selector_instance = (
            int(parent_atom.GetIntProp("_poly_csp_selector_instance"))
            if parent_atom.HasProp("_poly_csp_selector_instance")
            else None
        )
        selector_local_idx = (
            int(parent_atom.GetIntProp("_poly_csp_selector_local_idx"))
            if parent_atom.HasProp("_poly_csp_selector_local_idx")
            else None
        )
        connector_role = (
            parent_atom.GetProp("_poly_csp_connector_role")
            if parent_atom.HasProp("_poly_csp_connector_role")
            else None
        )

        if residue_label is not None:
            if atom.GetAtomicNum() == 1:
                siblings = hydrogen_neighbors_by_parent.get(parent_heavy_index, [])
                serial = siblings.index(atom_idx) + 1 if atom_idx in siblings else 1
                atom_name = _backbone_hydrogen_name(
                    residue_label,
                    len(siblings),
                    serial,
                )
            else:
                atom_name = residue_label
            canonical_name = (
                f"res{int(residue_index) + 1}.{atom_name}"
                if residue_index is not None
                else atom_name
            )
            source = "backbone"
            pdb_res_name = "GLC"
            pdb_res_num = int(residue_index) + 1 if residue_index is not None else 1
            pdb_chain = "A"
            pdb_hetero = False
        elif selector_instance is not None and selector_local_idx is not None:
            if atom.GetAtomicNum() == 1:
                siblings = hydrogen_neighbors_by_parent.get(parent_heavy_index, [])
                serial = siblings.index(atom_idx) + 1 if atom_idx in siblings else 1
                atom_name = f"H{selector_local_idx % 100:02d}{serial}"
                canonical_name = f"sel{selector_instance}.a{selector_local_idx}.h{serial}"
            else:
                atom_name = f"S{selector_local_idx:03d}"
                canonical_name = f"sel{selector_instance}.a{selector_local_idx}"
            source = component
            pdb_res_name = "SEL"
            pdb_res_num = int(selector_instance)
            pdb_chain = "B"
            pdb_hetero = True
        else:
            side = cap_side_for_atom.get(parent_heavy_index, "left")
            side_code = "L" if side == "left" else "R"
            heavy_serial = cap_heavy_serial.get(parent_heavy_index, 1)
            if atom.GetAtomicNum() == 1:
                siblings = hydrogen_neighbors_by_parent.get(parent_heavy_index, [])
                serial = siblings.index(atom_idx) + 1 if atom_idx in siblings else 1
                atom_name = f"H{side_code}{heavy_serial}{serial}"
                canonical_name = f"cap.{side}.a{heavy_serial}.h{serial}"
            else:
                atom_name = f"{side_code}{atom.GetSymbol()}{heavy_serial}"
                canonical_name = f"cap.{side}.{atom.GetSymbol().lower()}{heavy_serial}"
            source = f"terminal_cap_{side}"
            pdb_res_name = "CAP"
            pdb_res_num = 1 if side == "left" else n_residues
            pdb_chain = "A"
            pdb_hetero = False

        manifest.append(
            AtomManifestEntry(
                atom_index=atom_idx,
                parent_heavy_index=parent_heavy_index,
                component=component,
                residue_index=residue_index,
                residue_label=residue_label,
                site=site,
                selector_instance=selector_instance,
                selector_local_idx=selector_local_idx,
                connector_role=connector_role,
                atom_name=atom_name,
                canonical_name=canonical_name,
                source=source,
            )
        )

        atom.SetProp("_poly_csp_atom_name", atom_name)
        atom.SetProp("_poly_csp_canonical_name", canonical_name)
        atom.SetProp("_poly_csp_manifest_source", source)
        _set_pdb_info(
            atom,
            atom_name=atom_name,
            residue_name=pdb_res_name,
            residue_number=pdb_res_num,
            chain_id=pdb_chain,
            is_hetero=pdb_hetero,
        )

    mol.SetIntProp("_poly_csp_manifest_schema_version", MANIFEST_SCHEMA_VERSION)
    return manifest
