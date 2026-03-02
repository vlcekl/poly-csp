# poly_csp/chemistry/selectors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class SelectorTemplate:
    name: str
    mol: Chem.Mol
    attach_atom_idx: int  # on selector, atom to bond from (typically carbonyl carbon)
    dihedrals: Dict[str, Tuple[int, int, int, int]]  # selector-local indices
    donors: Tuple[int, ...] = ()
    acceptors: Tuple[int, ...] = ()
    attach_dummy_idx: int | None = None  # optional [*] to be replaced at attachment time


class SelectorRegistry:
    _reg: Dict[str, SelectorTemplate] = {}

    @classmethod
    def _norm(cls, name: str) -> str:
        return name.strip().lower()

    @classmethod
    def _register_builtins(cls) -> None:
        tpl = cls._reg.get("35dmpc")
        if tpl is None:
            from poly_csp.chemistry.selector_library.dmpc_35 import make_35_dmpc_template

            tpl = make_35_dmpc_template()
            cls.register(tpl)
            cls._reg["35dmpc"] = tpl
        # Common alias used by configs.
        cls._reg["dmpc_35"] = tpl

    @classmethod
    def register(cls, template: SelectorTemplate) -> None:
        key = cls._norm(template.name)
        existing = cls._reg.get(key)
        if existing is not None and existing is not template:
            raise ValueError(f"Selector {template.name!r} is already registered.")
        cls._reg[key] = template

    @classmethod
    def get(cls, name: str) -> SelectorTemplate:
        cls._register_builtins()
        key = cls._norm(name)
        if key not in cls._reg:
            available = ", ".join(sorted(cls._reg.keys()))
            raise KeyError(f"Unknown selector {name!r}. Available: [{available}]")
        return cls._reg[key]
