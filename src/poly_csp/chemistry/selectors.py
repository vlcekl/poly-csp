# poly_csp/chemistry/selectors.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from rdkit import Chem


def infer_donor_acceptor_atoms(mol: Chem.Mol) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Lightweight donor/acceptor inference for selector plugins.
    """
    donors: list[int] = []
    acceptors: list[int] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        if z == 7:
            has_h = any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())
            if has_h and charge <= 0:
                donors.append(idx)
            if charge <= 0 and not atom.GetIsAromatic():
                acceptors.append(idx)
        elif z == 8:
            if charge <= 0:
                acceptors.append(idx)
        elif z == 16 and charge <= 0:
            acceptors.append(idx)
    return tuple(donors), tuple(acceptors)


@dataclass(frozen=True)
class SelectorTemplate:
    name: str
    mol: Chem.Mol
    attach_atom_idx: int  # atom to bond from (typically carbonyl carbon)
    dihedrals: Dict[str, Tuple[int, int, int, int]]  # selector-local indices
    donors: Tuple[int, ...] = ()
    acceptors: Tuple[int, ...] = ()
    attach_dummy_idx: int | None = None  # optional [*] replaced at attachment
    features: Dict[str, Tuple[int, ...]] = field(default_factory=dict)


def selector_from_smiles(
    name: str,
    smiles: str,
    attach_atom_idx: int,
    dihedrals: Dict[str, Tuple[int, int, int, int]],
    attach_dummy_idx: int | None = None,
    auto_detect_hbond: bool = True,
) -> SelectorTemplate:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid selector SMILES for {name!r}.")
    Chem.SanitizeMol(mol)

    donors, acceptors = infer_donor_acceptor_atoms(mol) if auto_detect_hbond else ((), ())
    return SelectorTemplate(
        name=name,
        mol=mol,
        attach_atom_idx=int(attach_atom_idx),
        attach_dummy_idx=attach_dummy_idx,
        dihedrals=dict(dihedrals),
        donors=tuple(donors),
        acceptors=tuple(acceptors),
        features={"donors": tuple(donors), "acceptors": tuple(acceptors)},
    )


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
