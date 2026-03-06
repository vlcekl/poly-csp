from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
from rdkit import Chem

from poly_csp.config.schema import MonomerRepresentation, PolymerKind, Site
from poly_csp.topology.atom_mapping import attachment_instance_maps
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selectors import SelectorTemplate
from poly_csp.topology.utils import residue_label_maps


@dataclass(frozen=True)
class ConnectorParams:
    bond_params: Dict[str, tuple[float, float]] = field(default_factory=dict)
    angle_params: Dict[str, tuple[float, float]] = field(default_factory=dict)
    torsion_params: Dict[str, tuple[int, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CappedMonomerFragment:
    mol: Chem.Mol
    atom_roles: Dict[str, int] = field(default_factory=dict)
    connector_roles: Dict[str, int] = field(default_factory=dict)


def build_capped_monomer_fragment(
    polymer: PolymerKind,
    selector_template: SelectorTemplate,
    site: Site,
    monomer_representation: MonomerRepresentation = "natural_oh",
) -> CappedMonomerFragment:
    """Build a single-residue capped fragment with one attached selector.

    The fragment is the topology/structure precursor for capped-monomer
    connector parameter extraction. Atom roles are assigned in semantic
    space:
    - backbone atoms: ``BB_<label>`` from residue label metadata
    - attached selector atoms: ``SL_<local_idx>``
    """
    template = make_glucose_template(
        polymer,
        monomer_representation=monomer_representation,
    )
    frag = polymerize(
        template=template,
        dp=1,
        linkage="1-4",
        anomer="alpha" if polymer == "amylose" else "beta",
    )
    coords = np.asarray(
        template.mol.GetConformer(0).GetPositions(), dtype=float
    ).reshape((-1, 3))
    frag = assign_conformer(frag, coords)
    frag = attach_selector(
        mol_polymer=frag,
        template=template,
        residue_index=0,
        site=site,
        selector=selector_template,
        linkage_type=selector_template.linkage_type,
    )

    atom_roles: Dict[str, int] = {}
    connector_roles: Dict[str, int] = {}

    label_map = residue_label_maps(frag)[0]
    for label, atom_idx in label_map.items():
        role = f"BB_{label}"
        atom_roles[role] = int(atom_idx)
        frag.GetAtomWithIdx(int(atom_idx)).SetProp("_poly_csp_fragment_role", role)

    attachment_maps = attachment_instance_maps(frag)
    if not attachment_maps:
        raise ValueError("Attached selector fragment is missing instance metadata.")
    if len(attachment_maps) != 1:
        raise ValueError(f"Expected exactly one selector instance, got {len(attachment_maps)}.")
    instance_map = next(iter(attachment_maps.values()))
    for local_idx, atom_idx in instance_map.items():
        role = f"SL_{local_idx:03d}"
        atom_roles[role] = int(atom_idx)
        frag.GetAtomWithIdx(int(atom_idx)).SetProp("_poly_csp_fragment_role", role)

    for local_idx, role_name in selector_template.connector_local_roles.items():
        if local_idx not in instance_map:
            raise ValueError(
                f"Connector local index {local_idx} is missing from attached fragment."
            )
        connector_roles[role_name] = int(instance_map[local_idx])

    return CappedMonomerFragment(
        mol=frag,
        atom_roles=atom_roles,
        connector_roles=connector_roles,
    )


def extract_linkage_params(prmtop_path: str | Path, atom_map: Dict[int, int] | None = None) -> Dict[str, object]:
    """Placeholder extraction API for capped-monomer linkage parameters."""
    return {
        "prmtop": str(prmtop_path),
        "atom_map_size": 0 if atom_map is None else len(atom_map),
        "status": "not_implemented",
    }


def parameterize_capped_monomer(
    backbone_template: Chem.Mol,
    selector_template: Chem.Mol,
    site: str,
) -> ConnectorParams:
    """Build capped-monomer connector parameters (incremental stub)."""
    if backbone_template.GetNumAtoms() == 0:
        raise ValueError("backbone_template must contain atoms")
    if selector_template.GetNumAtoms() == 0:
        raise ValueError("selector_template must contain atoms")
    if not site:
        raise ValueError("site must be non-empty")
    return ConnectorParams()
