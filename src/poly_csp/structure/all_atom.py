from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.structure.hydrogens import complete_with_hydrogens
from poly_csp.structure.matrix import kabsch_align
from poly_csp.structure.naming import AtomManifestEntry, build_atom_manifest
from poly_csp.structure.templates import (
    build_residue_variant,
    load_explicit_backbone_template,
)
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.residue_state import (
    ResidueTemplateState,
    resolve_residue_template_states,
)
from poly_csp.topology.utils import (
    coords_from_mol,
    removed_old_indices,
    residue_label_maps,
    set_coords,
    terminal_cap_indices,
)


@dataclass(frozen=True)
class AllAtomBuildResult:
    mol: Chem.Mol
    manifest: list[AtomManifestEntry]
    residue_states: list[ResidueTemplateState]


def select_residue_templates(topology_mol: Chem.Mol) -> list[ResidueTemplateState]:
    return resolve_residue_template_states(topology_mol)


def _backbone_heavy_indices(mol: Chem.Mol) -> set[int]:
    indices: set[int] = set()
    for mapping in residue_label_maps(mol):
        indices.update(int(idx) for idx in mapping.values())
    return indices


def _backbone_coords(
    topology_mol: Chem.Mol,
    helix_spec: HelixSpec,
) -> np.ndarray:
    coords = coords_from_mol(topology_mol)
    if coords is not None:
        return coords

    backbone_heavy = _backbone_heavy_indices(topology_mol)
    non_backbone_heavy = [
        atom.GetIdx()
        for atom in topology_mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in backbone_heavy
    ]
    if non_backbone_heavy:
        raise ValueError(
            "Molecule is missing coordinates for non-backbone heavy atoms; "
            "cannot rebuild all-atom coordinates from helix metadata alone."
        )

    polymer = topology_mol.GetProp("_poly_csp_polymer")
    representation = (
        topology_mol.GetProp("_poly_csp_representation")
        if topology_mol.HasProp("_poly_csp_representation")
        else "anhydro"
    )
    dp = len(residue_label_maps(topology_mol))
    template = make_glucose_template(
        polymer,  # type: ignore[arg-type]
        monomer_representation=representation,  # type: ignore[arg-type]
    )
    coords = build_backbone_coords(template=template, helix=helix_spec, dp=dp)
    removed = removed_old_indices(topology_mol)
    if removed:
        keep = np.ones((coords.shape[0],), dtype=bool)
        keep[np.asarray(removed, dtype=int)] = False
        coords = coords[keep]
    return coords


def _annotate_existing_heavy_atoms(mol: Chem.Mol) -> None:
    maps = residue_label_maps(mol)
    for residue_index, mapping in enumerate(maps):
        for label, atom_idx in mapping.items():
            atom = mol.GetAtomWithIdx(int(atom_idx))
            atom.SetProp("_poly_csp_component", "backbone")
            atom.SetIntProp("_poly_csp_residue_index", int(residue_index))
            atom.SetProp("_poly_csp_residue_label", str(label))
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.UpdatePropertyCache(strict=False)

    cap_indices = terminal_cap_indices(mol)
    for side in ("left", "right"):
        for atom_idx in cap_indices.get(side, []):
            atom = mol.GetAtomWithIdx(int(atom_idx))
            if not atom.HasProp("_poly_csp_component"):
                atom.SetProp("_poly_csp_component", "backbone")
            atom.SetProp("_poly_csp_terminal_cap_side", side)


def build_all_atom_backbone_structure(
    topology_mol: Chem.Mol,
    helix_spec: HelixSpec,
    residue_states: list[ResidueTemplateState],
) -> AllAtomBuildResult:
    heavy_coords = _backbone_coords(topology_mol, helix_spec)

    rw = Chem.RWMol(Chem.Mol(topology_mol))
    _annotate_existing_heavy_atoms(rw)

    maps = residue_label_maps(topology_mol)
    added_positions: list[np.ndarray] = []

    for state in residue_states:
        mapping = maps[state.residue_index]
        base_template = load_explicit_backbone_template(
            polymer=state.polymer,
            representation=state.representation,
        )
        variant = build_residue_variant(base_template, state)
        template_coords = np.asarray(
            variant.mol.GetConformer(0).GetPositions(),
            dtype=float,
        ).reshape((-1, 3))

        shared_labels = [
            label
            for label in variant.heavy_label_to_idx
            if label in mapping
        ]
        if len(shared_labels) < 3:
            raise ValueError(
                f"Residue {state.residue_index} does not have enough heavy labels for "
                "rigid alignment."
            )

        template_heavy = np.asarray(
            [template_coords[variant.heavy_label_to_idx[label]] for label in shared_labels],
            dtype=float,
        )
        target_heavy = np.asarray(
            [heavy_coords[mapping[label]] for label in shared_labels],
            dtype=float,
        )
        rotation, translation = kabsch_align(template_heavy, target_heavy)
        aligned_coords = template_coords @ rotation.T + translation

        for template_idx, parent_label in sorted(variant.hydrogen_parent_label.items()):
            parent_global = mapping.get(parent_label)
            if parent_global is None:
                raise ValueError(
                    f"Residue {state.residue_index} is missing parent label {parent_label!r}."
                )
            atom_idx = rw.AddAtom(Chem.Atom(1))
            atom = rw.GetAtomWithIdx(int(atom_idx))
            atom.SetIntProp("_poly_csp_parent_heavy_idx", int(parent_global))
            atom.SetProp("_poly_csp_component", "backbone")
            atom.SetIntProp("_poly_csp_residue_index", int(state.residue_index))
            atom.SetProp("_poly_csp_residue_label", str(parent_label))
            rw.AddBond(int(parent_global), int(atom_idx), Chem.BondType.SINGLE)
            added_positions.append(np.asarray(aligned_coords[int(template_idx)], dtype=float))

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    if added_positions:
        coords = np.concatenate(
            [heavy_coords, np.asarray(added_positions, dtype=float)],
            axis=0,
        )
        set_coords(out, coords)
    else:
        set_coords(out, heavy_coords)
    manifest = build_atom_manifest(out)
    return AllAtomBuildResult(mol=out, manifest=manifest, residue_states=residue_states)


def build_structure_all_atom_molecule(
    topology_mol: Chem.Mol,
    helix_spec: HelixSpec,
) -> AllAtomBuildResult:
    residue_states = select_residue_templates(topology_mol)
    backbone_result = build_all_atom_backbone_structure(
        topology_mol=topology_mol,
        helix_spec=helix_spec,
        residue_states=residue_states,
    )
    mol = Chem.Mol(backbone_result.mol)
    backbone_heavy = _backbone_heavy_indices(topology_mol)
    add_h_to = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in backbone_heavy
    ]
    if add_h_to:
        mol = complete_with_hydrogens(
            mol,
            add_coords=mol.GetNumConformers() > 0,
            optimize="h_only" if mol.GetNumConformers() > 0 else "none",
            only_on_atoms=add_h_to,
        )
    manifest = build_atom_manifest(mol)
    return AllAtomBuildResult(mol=mol, manifest=manifest, residue_states=residue_states)
