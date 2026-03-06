"""Topology domain: RDKit chemical graph construction and metadata."""

from .backbone import assign_conformer, polymerize
from .monomers import GlucoseMonomerTemplate, make_glucose_template
from .reactions import attach_selector, residue_atom_global_index
from .residue_state import ResidueTemplateState, resolve_residue_template_states
from .selectors import SelectorRegistry, SelectorTemplate
from .terminals import apply_terminal_mode

__all__ = [
    "GlucoseMonomerTemplate",
    "ResidueTemplateState",
    "SelectorTemplate",
    "SelectorRegistry",
    "assign_conformer",
    "polymerize",
    "make_glucose_template",
    "attach_selector",
    "residue_atom_global_index",
    "resolve_residue_template_states",
    "apply_terminal_mode",
]
