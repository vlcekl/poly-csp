"""Structure domain: deterministic geometric construction and transforms."""

from .backbone_builder import (
    BackboneBuildResult,
    build_backbone_heavy_coords,
    build_backbone_structure,
)
from .dihedrals import measure_dihedral_rad, set_dihedral_rad
from .hydrogens import complete_with_hydrogens
from .matrix import ScrewTransform, rotation_matrix_z
from .naming import AtomManifestEntry, MANIFEST_SCHEMA_VERSION
from .templates import build_residue_variant, load_explicit_backbone_template

__all__ = [
    "ScrewTransform",
    "BackboneBuildResult",
    "AtomManifestEntry",
    "MANIFEST_SCHEMA_VERSION",
    "build_backbone_heavy_coords",
    "build_backbone_structure",
    "build_residue_variant",
    "complete_with_hydrogens",
    "load_explicit_backbone_template",
    "measure_dihedral_rad",
    "rotation_matrix_z",
    "set_dihedral_rad",
]
