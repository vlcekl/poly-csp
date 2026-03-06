"""Topology domain: RDKit chemical graph construction and metadata."""

from .backbone import polymerize
from .monomers import GlucoseMonomerTemplate, make_glucose_template
from .reactions import attach_selector
from .residue_state import ResidueTemplateState, resolve_residue_template_states
from .selectors import SelectorRegistry, SelectorTemplate
from .terminals import apply_terminal_mode

__all__ = [
    "GlucoseMonomerTemplate",
    "ResidueTemplateState",
    "SelectorTemplate",
    "SelectorRegistry",
    "polymerize",
    "make_glucose_template",
    "attach_selector",
    "resolve_residue_template_states",
    "apply_terminal_mode",
]
