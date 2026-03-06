"""Topology domain: RDKit chemical graph construction and metadata."""

from .backbone import assign_conformer, polymerize
from .monomers import GlucoseMonomerTemplate, make_glucose_template
from .reactions import attach_selector, residue_atom_global_index
from .selectors import SelectorRegistry, SelectorTemplate
from .terminals import apply_terminal_mode

__all__ = [
    "GlucoseMonomerTemplate",
    "SelectorTemplate",
    "SelectorRegistry",
    "assign_conformer",
    "polymerize",
    "make_glucose_template",
    "attach_selector",
    "residue_atom_global_index",
    "apply_terminal_mode",
]
