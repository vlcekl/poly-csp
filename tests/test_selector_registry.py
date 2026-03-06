from __future__ import annotations

from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template
from poly_csp.topology.selectors import SelectorRegistry


def test_selector_registry_register_and_get() -> None:
    template = make_35_dmpc_template()
    SelectorRegistry.register(template)

    got1 = SelectorRegistry.get("35dmpc")
    got2 = SelectorRegistry.get("dmpc_35")
    assert got1.name == "35dmpc"
    assert got2.name == "35dmpc"
    assert got1.attach_atom_idx == template.attach_atom_idx
