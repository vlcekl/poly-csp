from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly_csp.config.schema import HelixSpec
from poly_csp.structure.backbone_builder import build_backbone_structure
from poly_csp.structure.local_frames import compute_residue_local_frame
from poly_csp.structure.pbc import get_box_vectors_A
from poly_csp.topology.backbone import polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorRegistry, SelectorTemplate
from poly_csp.topology.terminals import apply_terminal_mode
from poly_csp.topology.utils import residue_label_maps

if TYPE_CHECKING:
    from poly_csp.forcefield.minimization import ExplicitPositionalRestraintGroup
    from poly_csp.forcefield.relaxation import RelaxSpec
    from poly_csp.forcefield.runtime_params import RuntimeParams


_FRAME_LABELS = ("C1", "C2", "C3", "C4", "O4")
_HELIX_CORE_BACKBONE_ATOM_NAMES = frozenset({"C1", "C2", "C3", "C4", "C5", "O4", "O5"})


@dataclass(frozen=True)
class PeriodicHandoffSpec:
    include_backbone_exocyclic: bool = True
    include_connector: bool = True
    include_selector: bool = True


@dataclass(frozen=True, order=True)
class PeriodicAtomKey:
    component: str
    atom_name: str
    site: str | None
    selector_local_idx: int | None
    connector_role: str | None
    is_hydrogen: bool
    parent_atom_name: str | None


@dataclass(frozen=True)
class PeriodicLocalAtomGeometry:
    key: PeriodicAtomKey
    global_atom_index: int
    local_coords_A: tuple[float, float, float]
    atomic_num: int


@dataclass(frozen=True)
class PeriodicResidueClassGeometry:
    class_index: int
    residue_index: int
    atom_geometries: tuple[PeriodicLocalAtomGeometry, ...]
    component_counts: dict[str, int]


@dataclass(frozen=True)
class PeriodicHandoffTemplate:
    unit_cell_dp: int
    periodic_box_A: tuple[float, float, float] | None
    helix_core_atom_names: tuple[str, ...]
    selector_sites: tuple[str, ...]
    residue_classes: tuple[PeriodicResidueClassGeometry, ...]


@dataclass(frozen=True)
class PeriodicHandoffResult:
    template: PeriodicHandoffTemplate
    extracted_atom_count: int
    extracted_backbone_atom_count: int
    extracted_selector_atom_count: int
    extracted_connector_atom_count: int


@dataclass(frozen=True)
class PeriodicOpenHandoffResult:
    mol: Chem.Mol
    expanded_dp: int
    n_cells: int
    selector_name: str | None
    selector_sites: tuple[str, ...]
    interior_residue_indices: tuple[int, ...]
    transferred_atom_count: int
    transfer_rmsd_A: float
    transfer_max_deviation_A: float
    interior_transferred_atom_count: int
    interior_transfer_rmsd_A: float
    interior_transfer_max_deviation_A: float


@dataclass(frozen=True)
class PeriodicHandoffCleanupSpec:
    enabled: bool = True
    interior_positional_k: float = 1000.0
    terminal_positional_k: float = 250.0


def _require_periodic_forcefield_molecule(mol: Chem.Mol) -> None:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Periodic handoff extraction requires a forcefield-domain molecule from "
            "build_forcefield_molecule()."
        )
    if not mol.HasProp("_poly_csp_end_mode"):
        raise ValueError("Periodic handoff extraction requires _poly_csp_end_mode metadata.")
    if str(mol.GetProp("_poly_csp_end_mode")).strip().lower() != "periodic":
        raise ValueError("Periodic handoff extraction requires end_mode='periodic'.")
    if mol.GetNumConformers() == 0:
        raise ValueError("Periodic handoff extraction requires 3D coordinates.")
    if get_box_vectors_A(mol) is None:
        raise ValueError(
            "Periodic handoff extraction requires stored periodic box vectors on the "
            "forcefield-domain molecule."
        )


def _manifest_source(atom: Chem.Atom) -> str:
    if not atom.HasProp("_poly_csp_manifest_source"):
        raise ValueError(
            f"Periodic handoff extraction requires _poly_csp_manifest_source on atom "
            f"{atom.GetIdx()}."
        )
    source = str(atom.GetProp("_poly_csp_manifest_source")).strip().lower()
    if source not in {"backbone", "connector", "selector"}:
        raise ValueError(
            "Periodic handoff extraction only supports backbone/connector/selector atoms; "
            f"atom {atom.GetIdx()} has source {source!r}."
        )
    return source


def _atom_name(atom: Chem.Atom) -> str:
    if not atom.HasProp("_poly_csp_atom_name"):
        raise ValueError(
            f"Periodic handoff extraction requires _poly_csp_atom_name on atom {atom.GetIdx()}."
        )
    return str(atom.GetProp("_poly_csp_atom_name"))


def _parent_heavy_atom(mol: Chem.Mol, atom: Chem.Atom) -> Chem.Atom:
    if atom.GetAtomicNum() != 1:
        return atom
    if atom.HasProp("_poly_csp_parent_heavy_idx"):
        return mol.GetAtomWithIdx(int(atom.GetIntProp("_poly_csp_parent_heavy_idx")))
    if atom.GetDegree() == 1:
        return atom.GetNeighbors()[0]
    raise ValueError(
        f"Hydrogen atom {atom.GetIdx()} is missing parent-heavy metadata for periodic handoff."
    )


def _selector_local_idx(atom: Chem.Atom) -> int | None:
    return int(atom.GetIntProp("_poly_csp_selector_local_idx")) if atom.HasProp(
        "_poly_csp_selector_local_idx"
    ) else None


def _site(atom: Chem.Atom) -> str | None:
    return str(atom.GetProp("_poly_csp_site")) if atom.HasProp("_poly_csp_site") else None


def _connector_role(atom: Chem.Atom) -> str | None:
    return (
        str(atom.GetProp("_poly_csp_connector_role"))
        if atom.HasProp("_poly_csp_connector_role")
        else None
    )


def _should_transfer_atom(
    mol: Chem.Mol,
    atom: Chem.Atom,
    spec: PeriodicHandoffSpec,
) -> bool:
    source = _manifest_source(atom)
    parent = _parent_heavy_atom(mol, atom)
    parent_name = _atom_name(parent)

    if source == "backbone":
        if not spec.include_backbone_exocyclic:
            return False
        return parent_name not in _HELIX_CORE_BACKBONE_ATOM_NAMES
    if source == "connector":
        return bool(spec.include_connector)
    if source == "selector":
        return bool(spec.include_selector)
    return False


def _periodic_atom_key(
    mol: Chem.Mol,
    atom: Chem.Atom,
) -> PeriodicAtomKey:
    source = _manifest_source(atom)
    parent = _parent_heavy_atom(mol, atom)
    is_hydrogen = bool(atom.GetAtomicNum() == 1)
    parent_name = _atom_name(parent) if is_hydrogen else None
    identity_atom = parent if is_hydrogen else atom
    selector_local_idx = _selector_local_idx(identity_atom)
    connector_role = _connector_role(identity_atom)
    site = _site(identity_atom)
    if source in {"selector", "connector"} and site is None:
        raise ValueError(
            "Periodic handoff extraction requires _poly_csp_site on every selector-bearing "
            f"atom; atom {atom.GetIdx()} is missing it."
        )
    if source in {"selector", "connector"} and selector_local_idx is None:
        raise ValueError(
            "Periodic handoff extraction requires _poly_csp_selector_local_idx on every "
            f"selector-bearing atom; atom {atom.GetIdx()} is missing it."
        )
    if source == "connector" and connector_role is None:
        raise ValueError(
            "Periodic handoff extraction requires _poly_csp_connector_role on every "
            f"connector atom; atom {atom.GetIdx()} is missing it."
        )
    return PeriodicAtomKey(
        component=source,
        atom_name=_atom_name(atom),
        site=site,
        selector_local_idx=selector_local_idx,
        connector_role=connector_role,
        is_hydrogen=is_hydrogen,
        parent_atom_name=parent_name,
    )


def _residue_local_frame(
    mol: Chem.Mol,
    residue_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    maps = residue_label_maps(mol)
    if residue_index < 0 or residue_index >= len(maps):
        raise ValueError(f"residue_index {residue_index} out of range for periodic handoff.")
    mapping = maps[residue_index]
    missing = [label for label in _FRAME_LABELS if label not in mapping]
    if missing:
        raise ValueError(
            f"Residue {residue_index} is missing local-frame labels required for periodic handoff: "
            f"{missing!r}."
        )
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    coords_res = np.array([xyz[int(mapping[label])] for label in _FRAME_LABELS], dtype=float)
    labels = {label: idx for idx, label in enumerate(_FRAME_LABELS)}
    return compute_residue_local_frame(coords_res, labels)


def _local_coords_A(global_coords_A: np.ndarray, frame_r: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
    return (np.asarray(global_coords_A, dtype=float) - np.asarray(frame_t, dtype=float)) @ frame_r


def _periodic_selector_name(mol: Chem.Mol) -> str | None:
    names = sorted(
        {
            str(atom.GetProp("_poly_csp_selector_name")).strip()
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_selector_name")
        }
    )
    if not names:
        return None
    if len(names) != 1:
        raise ValueError(
            "Periodic handoff open expansion currently supports exactly one selector name "
            f"in the periodic source, found {names!r}."
        )
    return names[0]


def _string_prop(mol: Chem.Mol, prop: str) -> str:
    if not mol.HasProp(prop):
        raise ValueError(f"Periodic handoff open expansion requires {prop} metadata.")
    return str(mol.GetProp(prop))


def _build_forcefield_molecule_checked(mol: Chem.Mol) -> Chem.Mol:
    from poly_csp.forcefield.model import build_forcefield_molecule

    return build_forcefield_molecule(mol).mol


def _run_staged_relaxation(
    **kwargs,
):
    from poly_csp.forcefield.relaxation import run_staged_relaxation

    return run_staged_relaxation(**kwargs)


def _residue_transferable_atom_indices_by_key(
    mol: Chem.Mol,
    residue_index: int,
    spec: PeriodicHandoffSpec | None = None,
) -> dict[PeriodicAtomKey, int]:
    resolved_spec = spec or PeriodicHandoffSpec()
    key_to_index: dict[PeriodicAtomKey, int] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_residue_index"):
            continue
        if int(atom.GetIntProp("_poly_csp_residue_index")) != residue_index:
            continue
        if not _should_transfer_atom(mol, atom, resolved_spec):
            continue
        key = _periodic_atom_key(mol, atom)
        if key in key_to_index:
            raise ValueError(
                "Periodic handoff atom mapping produced a duplicate transferable atom key "
                f"for residue {residue_index}: {key!r}."
            )
        key_to_index[key] = int(atom.GetIdx())
    return key_to_index


def _template_by_class(
    template: PeriodicHandoffTemplate,
) -> dict[int, PeriodicResidueClassGeometry]:
    return {
        int(residue_class.class_index): residue_class
        for residue_class in template.residue_classes
    }


def _interior_residue_indices(
    template: PeriodicHandoffTemplate,
    n_cells: int,
) -> tuple[int, ...]:
    middle_cell = int(n_cells // 2)
    start = middle_cell * int(template.unit_cell_dp)
    stop = (middle_cell + 1) * int(template.unit_cell_dp)
    return tuple(range(start, stop))


def _evaluate_open_handoff_result(
    mol: Chem.Mol,
    template: PeriodicHandoffTemplate,
    *,
    n_cells: int,
    selector_name: str | None,
    selector_sites: Sequence[str],
) -> PeriodicOpenHandoffResult:
    expanded_dp = int(template.unit_cell_dp) * int(n_cells)
    template_by_class = _template_by_class(template)
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    interior_residue_indices = _interior_residue_indices(template, n_cells)
    interior_set = set(interior_residue_indices)
    all_deviations: list[float] = []
    interior_deviations: list[float] = []

    for residue_index in range(expanded_dp):
        class_index = int(residue_index % int(template.unit_cell_dp))
        class_geometry = template_by_class.get(class_index)
        if class_geometry is None:
            raise ValueError(
                f"Periodic handoff template is missing residue class {class_index}."
            )
        target_by_key = _residue_transferable_atom_indices_by_key(mol, residue_index)
        frame_r, frame_t = _residue_local_frame(mol, residue_index)
        for atom_geometry in class_geometry.atom_geometries:
            target_idx = target_by_key.get(atom_geometry.key)
            if target_idx is None:
                raise ValueError(
                    "Periodic handoff open expansion could not find target atom "
                    f"{atom_geometry.key!r} on residue {residue_index}."
                )
            local = _local_coords_A(xyz[target_idx], frame_r, frame_t)
            deviation = float(
                np.linalg.norm(local - np.asarray(atom_geometry.local_coords_A, dtype=float))
            )
            all_deviations.append(deviation)
            if residue_index in interior_set:
                interior_deviations.append(deviation)

    transfer_rmsd = (
        float(np.sqrt(np.mean(np.square(all_deviations))))
        if all_deviations
        else 0.0
    )
    interior_transfer_rmsd = (
        float(np.sqrt(np.mean(np.square(interior_deviations))))
        if interior_deviations
        else 0.0
    )
    return PeriodicOpenHandoffResult(
        mol=mol,
        expanded_dp=int(expanded_dp),
        n_cells=int(n_cells),
        selector_name=selector_name,
        selector_sites=tuple(str(site) for site in selector_sites),
        interior_residue_indices=interior_residue_indices,
        transferred_atom_count=len(all_deviations),
        transfer_rmsd_A=transfer_rmsd,
        transfer_max_deviation_A=(max(all_deviations) if all_deviations else 0.0),
        interior_transferred_atom_count=len(interior_deviations),
        interior_transfer_rmsd_A=interior_transfer_rmsd,
        interior_transfer_max_deviation_A=(
            max(interior_deviations) if interior_deviations else 0.0
        ),
    )


def _build_handoff_reference_group(
    mol: Chem.Mol,
    template: PeriodicHandoffTemplate,
    residue_indices: Sequence[int],
    *,
    k_kj_per_mol_nm2: float,
    label: str,
    parameter_name: str,
) -> "ExplicitPositionalRestraintGroup" | None:
    from poly_csp.forcefield.minimization import ExplicitPositionalRestraintGroup

    if float(k_kj_per_mol_nm2) <= 0.0:
        return None
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    template_by_class = _template_by_class(template)
    atom_indices: list[int] = []
    reference_positions_A: list[tuple[float, float, float]] = []
    for residue_index in residue_indices:
        class_index = int(residue_index % int(template.unit_cell_dp))
        class_geometry = template_by_class.get(class_index)
        if class_geometry is None:
            raise ValueError(
                f"Periodic handoff template is missing residue class {class_index}."
            )
        target_by_key = _residue_transferable_atom_indices_by_key(mol, int(residue_index))
        for atom_geometry in class_geometry.atom_geometries:
            target_idx = target_by_key.get(atom_geometry.key)
            if target_idx is None:
                raise ValueError(
                    "Periodic handoff cleanup could not find target atom "
                    f"{atom_geometry.key!r} on residue {residue_index}."
                )
            atom_indices.append(int(target_idx))
            ref_xyz = xyz[int(target_idx)]
            reference_positions_A.append(
                (float(ref_xyz[0]), float(ref_xyz[1]), float(ref_xyz[2]))
            )
    if not atom_indices:
        return None
    return ExplicitPositionalRestraintGroup(
        atom_indices=tuple(atom_indices),
        reference_positions_A=tuple(reference_positions_A),
        k_kj_per_mol_nm2=float(k_kj_per_mol_nm2),
        parameter_name=str(parameter_name),
        label=str(label),
    )


def extract_periodic_handoff_template(
    mol: Chem.Mol,
    spec: PeriodicHandoffSpec | None = None,
) -> PeriodicHandoffResult:
    _require_periodic_forcefield_molecule(mol)
    resolved_spec = spec or PeriodicHandoffSpec()

    maps = residue_label_maps(mol)
    xyz = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float).reshape((-1, 3))
    residue_classes: list[PeriodicResidueClassGeometry] = []
    selector_sites: set[str] = set()
    extracted_backbone = 0
    extracted_selector = 0
    extracted_connector = 0

    for residue_index in range(len(maps)):
        frame_r, frame_t = _residue_local_frame(mol, residue_index)
        atom_geometries: list[PeriodicLocalAtomGeometry] = []
        component_counts = {"backbone": 0, "selector": 0, "connector": 0}
        seen_keys: set[PeriodicAtomKey] = set()

        for atom in mol.GetAtoms():
            if not atom.HasProp("_poly_csp_residue_index"):
                continue
            if int(atom.GetIntProp("_poly_csp_residue_index")) != residue_index:
                continue
            if not _should_transfer_atom(mol, atom, resolved_spec):
                continue

            key = _periodic_atom_key(mol, atom)
            if key in seen_keys:
                raise ValueError(
                    "Periodic handoff extraction produced a duplicate transferable atom key "
                    f"for residue {residue_index}: {key!r}."
                )
            seen_keys.add(key)

            source = key.component
            if source == "backbone":
                extracted_backbone += 1
            elif source == "selector":
                extracted_selector += 1
            elif source == "connector":
                extracted_connector += 1
            component_counts[source] += 1
            if key.site is not None:
                selector_sites.add(str(key.site))

            local_xyz = _local_coords_A(xyz[int(atom.GetIdx())], frame_r, frame_t)
            atom_geometries.append(
                PeriodicLocalAtomGeometry(
                    key=key,
                    global_atom_index=int(atom.GetIdx()),
                    local_coords_A=tuple(float(value) for value in local_xyz.tolist()),
                    atomic_num=int(atom.GetAtomicNum()),
                )
            )

        residue_classes.append(
            PeriodicResidueClassGeometry(
                class_index=residue_index,
                residue_index=residue_index,
                atom_geometries=tuple(sorted(atom_geometries, key=lambda item: item.key)),
                component_counts={
                    name: int(count)
                    for name, count in component_counts.items()
                    if count > 0
                },
            )
        )

    template = PeriodicHandoffTemplate(
        unit_cell_dp=len(maps),
        periodic_box_A=get_box_vectors_A(mol),
        helix_core_atom_names=tuple(sorted(_HELIX_CORE_BACKBONE_ATOM_NAMES)),
        selector_sites=tuple(sorted(selector_sites)),
        residue_classes=tuple(residue_classes),
    )
    return PeriodicHandoffResult(
        template=template,
        extracted_atom_count=int(extracted_backbone + extracted_selector + extracted_connector),
        extracted_backbone_atom_count=int(extracted_backbone),
        extracted_selector_atom_count=int(extracted_selector),
        extracted_connector_atom_count=int(extracted_connector),
    )


def build_open_handoff_receptor(
    periodic_mol: Chem.Mol,
    template: PeriodicHandoffTemplate,
    helix: HelixSpec,
    *,
    selector: SelectorTemplate | None = None,
    n_cells: int = 3,
    end_caps: dict[str, str] | None = None,
) -> PeriodicOpenHandoffResult:
    _require_periodic_forcefield_molecule(periodic_mol)
    if n_cells < 3:
        raise ValueError("Periodic handoff open expansion requires n_cells >= 3.")
    if n_cells % 2 == 0:
        raise ValueError("Periodic handoff open expansion requires an odd n_cells value.")
    if template.unit_cell_dp <= 0:
        raise ValueError("Periodic handoff template must define a positive unit_cell_dp.")
    if len(template.residue_classes) != int(template.unit_cell_dp):
        raise ValueError(
            "Periodic handoff template residue_classes must match unit_cell_dp exactly."
        )

    source_selector_name = _periodic_selector_name(periodic_mol)
    selector_sites = tuple(str(site) for site in template.selector_sites)
    resolved_selector = selector
    if selector_sites:
        if resolved_selector is None:
            if source_selector_name is None:
                raise ValueError(
                    "Periodic handoff open expansion requires a selector template when "
                    "selector sites are present in the template."
                )
            resolved_selector = SelectorRegistry.get(source_selector_name)
        if source_selector_name is not None and resolved_selector.name != source_selector_name:
            raise ValueError(
                "Periodic handoff open expansion selector mismatch: source periodic model "
                f"uses {source_selector_name!r}, received {resolved_selector.name!r}."
            )

    polymer = _string_prop(periodic_mol, "_poly_csp_polymer")
    representation = _string_prop(periodic_mol, "_poly_csp_representation")
    expanded_dp = int(template.unit_cell_dp) * int(n_cells)
    monomer = make_glucose_template(
        polymer,
        monomer_representation=representation,
    )
    topology = polymerize(
        template=monomer,
        dp=expanded_dp,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    topology = apply_terminal_mode(
        mol=topology,
        mode="open",
        caps=dict(end_caps or {}),
        representation=representation,
    )
    structure = build_backbone_structure(topology, helix_spec=helix).mol
    for residue_index in range(expanded_dp):
        for site in selector_sites:
            if resolved_selector is None:
                raise ValueError(
                    "Periodic handoff open expansion could not resolve the selector template."
                )
            structure = attach_selector(
                mol_polymer=structure,
                residue_index=residue_index,
                site=site,
                selector=resolved_selector,
                mode="bond_from_OH_oxygen",
            )

    handoff_mol = _build_forcefield_molecule_checked(structure)
    conf = handoff_mol.GetConformer(0)
    xyz = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))
    template_by_class = _template_by_class(template)

    for residue_index in range(expanded_dp):
        class_index = int(residue_index % int(template.unit_cell_dp))
        class_geometry = template_by_class.get(class_index)
        if class_geometry is None:
            raise ValueError(
                f"Periodic handoff template is missing residue class {class_index}."
            )
        target_by_key = _residue_transferable_atom_indices_by_key(handoff_mol, residue_index)
        frame_r, frame_t = _residue_local_frame(handoff_mol, residue_index)
        for atom_geometry in class_geometry.atom_geometries:
            if atom_geometry.key not in target_by_key:
                raise ValueError(
                    "Periodic handoff open expansion could not find target atom "
                    f"{atom_geometry.key!r} on residue {residue_index}."
                )
            target_idx = int(target_by_key[atom_geometry.key])
            placed_xyz = (
                np.asarray(atom_geometry.local_coords_A, dtype=float) @ frame_r.T
                + frame_t
            )
            xyz[target_idx] = placed_xyz

    for atom_idx, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))

    return _evaluate_open_handoff_result(
        handoff_mol,
        template,
        n_cells=int(n_cells),
        selector_name=(resolved_selector.name if resolved_selector is not None else source_selector_name),
        selector_sites=selector_sites,
    )


def run_open_handoff_cleanup_relaxation(
    handoff: PeriodicOpenHandoffResult,
    template: PeriodicHandoffTemplate,
    relax_spec: "RelaxSpec",
    *,
    cleanup_spec: PeriodicHandoffCleanupSpec | None = None,
    selector: SelectorTemplate | None = None,
    runtime_params: "RuntimeParams" | None = None,
    work_dir: str | Path | None = None,
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0,
    soft_repulsion_cutoff_nm: float = 0.6,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[PeriodicOpenHandoffResult, dict[str, object]]:
    resolved_cleanup = cleanup_spec or PeriodicHandoffCleanupSpec()
    if not resolved_cleanup.enabled:
        return handoff, {
            "enabled": False,
            "periodic_handoff_cleanup": {
                "enabled": False,
                "reason": "cleanup_disabled",
            },
        }

    selector_name = handoff.selector_name
    resolved_selector = selector
    if handoff.selector_sites:
        if resolved_selector is None:
            if selector_name is None:
                raise ValueError(
                    "Periodic handoff cleanup requires a selector template when selector "
                    "sites are present on the handoff model."
                )
            resolved_selector = SelectorRegistry.get(selector_name)
        if selector_name is not None and resolved_selector.name != selector_name:
            raise ValueError(
                "Periodic handoff cleanup selector mismatch: handoff model uses "
                f"{selector_name!r}, received {resolved_selector.name!r}."
            )

    interior_residue_indices = tuple(int(idx) for idx in handoff.interior_residue_indices)
    terminal_residue_indices = tuple(
        idx for idx in range(int(handoff.expanded_dp)) if idx not in set(interior_residue_indices)
    )
    restraint_groups = tuple(
        group
        for group in (
            _build_handoff_reference_group(
                handoff.mol,
                template,
                interior_residue_indices,
                k_kj_per_mol_nm2=float(resolved_cleanup.interior_positional_k),
                label="interior",
                parameter_name="k_pos_handoff_interior",
            ),
            _build_handoff_reference_group(
                handoff.mol,
                template,
                terminal_residue_indices,
                k_kj_per_mol_nm2=float(resolved_cleanup.terminal_positional_k),
                label="terminal",
                parameter_name="k_pos_handoff_terminal",
            ),
        )
        if group is not None
    )

    relaxed_mol, relax_summary = _run_staged_relaxation(
        mol=handoff.mol,
        spec=relax_spec,
        selector=resolved_selector,
        runtime_params=runtime_params,
        work_dir=work_dir,
        soft_repulsion_k_kj_per_mol_nm2=float(soft_repulsion_k_kj_per_mol_nm2),
        soft_repulsion_cutoff_nm=float(soft_repulsion_cutoff_nm),
        mixing_rules_cfg=mixing_rules_cfg,
        extra_positional_restraints=restraint_groups,
    )
    relaxed_result = _evaluate_open_handoff_result(
        relaxed_mol,
        template,
        n_cells=int(handoff.n_cells),
        selector_name=selector_name,
        selector_sites=handoff.selector_sites,
    )
    summary = dict(relax_summary)
    summary["periodic_handoff_cleanup"] = {
        "enabled": True,
        "n_cells": int(handoff.n_cells),
        "expanded_dp": int(handoff.expanded_dp),
        "interior_residue_indices": list(interior_residue_indices),
        "terminal_residue_indices": list(terminal_residue_indices),
        "pre_transfer_rmsd_A": float(handoff.transfer_rmsd_A),
        "post_transfer_rmsd_A": float(relaxed_result.transfer_rmsd_A),
        "pre_interior_transfer_rmsd_A": float(handoff.interior_transfer_rmsd_A),
        "post_interior_transfer_rmsd_A": float(relaxed_result.interior_transfer_rmsd_A),
        "restraint_groups": [
            {
                "label": str(group.label),
                "parameter_name": str(group.parameter_name),
                "k_kj_per_mol_nm2": float(group.k_kj_per_mol_nm2),
                "n_atoms": len(group.atom_indices),
            }
            for group in restraint_groups
        ],
    }
    return relaxed_result, summary
