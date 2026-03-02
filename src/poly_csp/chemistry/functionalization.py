# poly_csp/chemistry/functionalization.py
from __future__ import annotations

from collections import deque
import json
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.chemistry.monomers import GlucoseMonomerTemplate
from poly_csp.chemistry.selectors import SelectorRegistry, SelectorTemplate
from poly_csp.config.schema import SelectorPoseSpec, Site
from poly_csp.geometry.dihedrals import set_dihedral_rad
from poly_csp.geometry.local_frames import compute_residue_local_frame, pose_selector_in_frame


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


def _site_to_oxygen_label(site: Site) -> str:
    return f"O{site[1:]}"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Degenerate vector for selector placement.")
    return v / n


def _copy_props(src: Chem.Mol, dst: Chem.Mol) -> None:
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


def _residue_label_maps(mol: Chem.Mol) -> list[dict[str, int]]:
    if not mol.HasProp("_poly_csp_residue_label_map_json"):
        raise ValueError("Missing _poly_csp_residue_label_map_json metadata on molecule.")
    raw = mol.GetProp("_poly_csp_residue_label_map_json")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Invalid residue label map metadata format.")
    out: list[dict[str, int]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Invalid residue label map entry.")
        out.append({str(k): int(v) for k, v in item.items()})
    return out


def _residue_label_global_index(mol: Chem.Mol, residue_index: int, label: str) -> int:
    maps = _residue_label_maps(mol)
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
) -> Chem.Mol:
    """
    Chemically attach selector at a site.
    Default assumption: bond from sugar hydroxyl oxygen (O2/O3/O6) to selector
    attach atom (carbonyl carbon), consistent with carbamate construction used in
    legacy/build_dimer.py.
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

    oxygen_label = _site_to_oxygen_label(site)
    if oxygen_label not in template.site_idx:
        raise ValueError(f"Site {site} is not available in template.site_idx")

    sugar_o_global = _residue_label_global_index(mol_polymer, residue_index, oxygen_label)

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

    selector_coords = None
    if existing_coords is not None and selector.mol.GetNumConformers() > 0:
        sel_xyz = np.asarray(
            selector.mol.GetConformer(0).GetPositions(), dtype=float
        ).reshape((-1, 3))
        label_map = _residue_label_maps(mol_polymer)[residue_index]
        frame_labels = ("C1", "C2", "C3", "C4", "O4")
        coords_res = np.array([existing_coords[label_map[label]] for label in frame_labels])
        frame_idx = {label: i for i, label in enumerate(frame_labels)}
        r, _ = compute_residue_local_frame(coords_res, frame_idx)

        p_o = existing_coords[_residue_label_global_index(mol_polymer, residue_index, oxygen_label)]
        p_c = existing_coords[_residue_label_global_index(mol_polymer, residue_index, str(site))]
        bond_dir = _normalize(p_o - p_c)
        # Place carbonyl carbon near the target sugar oxygen with realistic O-C distance.
        t_attach = p_o + 1.36 * bond_dir

        selector_coords = pose_selector_in_frame(
            selector_coords=sel_xyz,
            pose=SelectorPoseSpec(carbonyl_dir_local=(-1.0, 0.0, 0.0)),
            r_res=r,
            t_res=t_attach,
            attach_atom_idx=selector.attach_atom_idx,
        )

    if selector.attach_dummy_idx is not None:
        rw.RemoveAtom(offset + selector.attach_dummy_idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    _copy_props(mol_polymer, mol)
    mol.SetIntProp("_poly_csp_selector_count", instance_id)

    if existing_coords is not None and selector_coords is not None:
        merged = np.concatenate([existing_coords, selector_coords], axis=0)
        if selector.attach_dummy_idx is not None:
            dummy_global = offset + selector.attach_dummy_idx
            merged = np.delete(merged, dummy_global, axis=0)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(merged):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

    return mol


def place_selector_coords(
    poly_coords: np.ndarray,
    coords_res: np.ndarray,
    selector_coords: np.ndarray,
    pose: SelectorPoseSpec,
) -> np.ndarray:
    """Rigidly place selector coordinates using residue-local frame + pose rules."""
    # Default labels for the current monomer template atom ordering.
    default_labels = {"C1": 0, "C2": 1, "C3": 3, "C4": 5, "O4": 6}
    r, t = compute_residue_local_frame(coords_res, default_labels)
    return pose_selector_in_frame(
        selector_coords=selector_coords,
        pose=pose,
        r_res=r,
        t_res=t,
        attach_atom_idx=0,
    )


def merge_conformers(poly_coords: np.ndarray, selector_coords: np.ndarray) -> np.ndarray:
    """Concatenate coordinate arrays; mapping handled at attachment time."""
    p = np.asarray(poly_coords, dtype=float).reshape((-1, 3))
    s = np.asarray(selector_coords, dtype=float).reshape((-1, 3))
    return np.concatenate([p, s], axis=0)


def _site_oxygen_global_index(mol: Chem.Mol, residue_index: int, site: Site) -> int:
    o_label = _site_to_oxygen_label(site)
    if mol.HasProp("_poly_csp_residue_label_map_json"):
        return _residue_label_global_index(mol, residue_index, o_label)
    if not mol.HasProp("_poly_csp_template_atom_count"):
        raise ValueError("Missing _poly_csp_template_atom_count metadata on molecule.")
    n_monomer = int(mol.GetIntProp("_poly_csp_template_atom_count"))
    prop = f"_poly_csp_siteidx_{o_label}"
    if not mol.HasProp(prop):
        raise ValueError(f"Missing {prop} metadata on molecule.")
    local_o = int(mol.GetIntProp(prop))
    return residue_atom_global_index(residue_index, n_monomer, local_o)


def _selector_local_to_global_map(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
) -> dict[int, int]:
    instances: dict[int, dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != residue_index:
            continue
        if atom.GetProp("_poly_csp_site") != site:
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        instances.setdefault(inst, {})[local] = atom.GetIdx()

    if not instances:
        raise ValueError(
            f"No selector atoms found for residue {residue_index}, site {site}."
        )
    selected_inst = max(instances.keys())
    return instances[selected_inst]


def _downstream_mask(mol: Chem.Mol, b: int, c: int) -> np.ndarray:
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    mask = np.zeros((n,), dtype=bool)
    q: deque[int] = deque([c])
    mask[c] = True
    while q:
        x = q.popleft()
        for nbr in adj[x]:
            if x == c and nbr == b:
                continue
            if not mask[nbr]:
                mask[nbr] = True
                q.append(nbr)
    return mask


def apply_selector_pose_dihedrals(
    mol: Chem.Mol,
    residue_index: int,
    site: Site,
    pose_spec: SelectorPoseSpec,
    selector: SelectorTemplate | None = None,
) -> Chem.Mol:
    """
    Apply target selector dihedrals (in degrees) for one attached selector.
    If multiple selectors exist at residue/site, the latest attached instance is used.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have coordinates before applying dihedrals.")
    if not pose_spec.dihedral_targets_deg:
        return Chem.Mol(mol)

    tpl = selector if selector is not None else SelectorRegistry.get("35dmpc")
    local_to_global = _selector_local_to_global_map(mol, residue_index, site)
    sugar_o_global = _site_oxygen_global_index(mol, residue_index, site)

    conf = mol.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))

    for name, target_deg in pose_spec.dihedral_targets_deg.items():
        if name not in tpl.dihedrals:
            raise KeyError(f"Unknown selector dihedral {name!r} for {tpl.name}.")

        mapped = []
        for local_idx in tpl.dihedrals[name]:
            if local_idx in local_to_global:
                mapped.append(local_to_global[local_idx])
            elif tpl.attach_dummy_idx is not None and local_idx == tpl.attach_dummy_idx:
                mapped.append(sugar_o_global)
            else:
                raise ValueError(
                    f"Could not map selector local index {local_idx} for dihedral {name!r}."
                )

        a, b, c, d = mapped
        rotate_mask = _downstream_mask(mol, b, c)
        coords = set_dihedral_rad(
            coords=coords,
            a=a,
            b=b,
            c=c,
            d=d,
            target_angle_rad=np.deg2rad(float(target_deg)),
            rotate_mask=rotate_mask,
        )

    out = Chem.Mol(mol)
    new_conf = Chem.Conformer(out.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        new_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    out.RemoveAllConformers()
    out.AddConformer(new_conf, assignId=True)
    return out
