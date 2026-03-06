# poly_csp/topology/monomers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from rdkit import Chem
from rdkit.Chem import AllChem

PolymerKind = Literal["amylose", "cellulose"]
MonomerRepresentation = Literal["anhydro", "natural_oh"]


@dataclass(frozen=True)
class GlucoseMonomerTemplate:
    mol: Chem.Mol
    atom_idx: Dict[str, int]
    site_idx: Dict[str, int]  # C2/C3/C6 and O2/O3/O6 site labels.
    polymer: PolymerKind
    representation: MonomerRepresentation


_AMYLOSE_ALPHA_ANHYDRO_MAPPED_SMILES = (
    "[C@@H:1]1[C@H:2]([OH:8])[C@@H:3]([OH:9])[C@H:4]([OH:10])"
    "[C@@H:5]([CH2:6][OH:11])[O:7]1"
)

_CELLULOSE_BETA_ANHYDRO_MAPPED_SMILES = (
    "[C@H:1]1[C@H:2]([OH:8])[C@@H:3]([OH:9])[C@H:4]([OH:10])"
    "[C@@H:5]([CH2:6][OH:11])[O:7]1"
)

_AMYLOSE_ALPHA_NATURAL_MAPPED_SMILES = (
    "[C@@H:1]1([OH:12])[C@H:2]([OH:8])[C@@H:3]([OH:9])[C@H:4]([OH:10])"
    "[C@@H:5]([CH2:6][OH:11])[O:7]1"
)

_CELLULOSE_BETA_NATURAL_MAPPED_SMILES = (
    "[C@H:1]1([OH:12])[C@H:2]([OH:8])[C@@H:3]([OH:9])[C@H:4]([OH:10])"
    "[C@@H:5]([CH2:6][OH:11])[O:7]1"
)

_MAPNUM_TO_LABEL_ANHYDRO = {
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "O5",
    8: "O2",
    9: "O3",
    10: "O4",
    11: "O6",
}

_MAPNUM_TO_LABEL_NATURAL = {
    **_MAPNUM_TO_LABEL_ANHYDRO,
    12: "O1",
}

_HYDROXYL_MAPNUMS_ANHYDRO = frozenset((8, 9, 10, 11))
_HYDROXYL_MAPNUMS_NATURAL = frozenset((8, 9, 10, 11, 12))


def _hydroxyl_mapnums(
    representation: MonomerRepresentation,
) -> frozenset[int]:
    return (
        _HYDROXYL_MAPNUMS_ANHYDRO
        if representation == "anhydro"
        else _HYDROXYL_MAPNUMS_NATURAL
    )


def _restore_implicit_hydroxyl_hydrogens(
    mol: Chem.Mol,
    representation: MonomerRepresentation,
) -> Chem.Mol:
    """Convert parsed [OH] atoms into implicit-H hydroxyl oxygens."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.GetAtomMapNum() not in _hydroxyl_mapnums(representation):
            continue
        if atom.GetAtomicNum() != 8:
            raise ValueError(
                f"Mapped hydroxyl atom {atom.GetAtomMapNum()} is not oxygen."
            )
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(out)
    return out


def _embed_mol_deterministic(mol: Chem.Mol, seed: int) -> Chem.Mol:
    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = False

    status = AllChem.EmbedMolecule(with_h, params)
    if status != 0:
        status = AllChem.EmbedMolecule(with_h, useRandomCoords=True, randomSeed=seed)
    if status != 0:
        raise RuntimeError("RDKit failed to embed glucose template.")

    AllChem.UFFOptimizeMolecule(with_h, maxIters=250)
    out = Chem.RemoveHs(with_h, sanitize=True)
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def _make_mol_from_polymer(
    polymer: PolymerKind,
    representation: MonomerRepresentation,
) -> Chem.Mol:
    if representation == "anhydro":
        if polymer == "amylose":
            mapped_smiles = _AMYLOSE_ALPHA_ANHYDRO_MAPPED_SMILES
            seed = 1001
        elif polymer == "cellulose":
            mapped_smiles = _CELLULOSE_BETA_ANHYDRO_MAPPED_SMILES
            seed = 2001
        else:
            raise ValueError(f"Unsupported polymer kind: {polymer!r}")
    elif representation == "natural_oh":
        if polymer == "amylose":
            mapped_smiles = _AMYLOSE_ALPHA_NATURAL_MAPPED_SMILES
            seed = 1101
        elif polymer == "cellulose":
            mapped_smiles = _CELLULOSE_BETA_NATURAL_MAPPED_SMILES
            seed = 2101
        else:
            raise ValueError(f"Unsupported polymer kind: {polymer!r}")
    else:
        raise ValueError(f"Unsupported monomer representation: {representation!r}")

    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(
            f"Could not parse mapped SMILES for {polymer}/{representation}."
        )
    Chem.SanitizeMol(mol)
    mol = _restore_implicit_hydroxyl_hydrogens(mol, representation)
    return _embed_mol_deterministic(mol, seed=seed)


def _build_atom_idx(
    mol: Chem.Mol,
    representation: MonomerRepresentation,
) -> Dict[str, int]:
    label_map = (
        _MAPNUM_TO_LABEL_ANHYDRO
        if representation == "anhydro"
        else _MAPNUM_TO_LABEL_NATURAL
    )
    atom_idx: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num in label_map:
            atom_idx[label_map[map_num]] = atom.GetIdx()
        atom.SetAtomMapNum(0)

    missing = set(label_map.values()) - set(atom_idx.keys())
    if missing:
        raise ValueError(f"Missing required atom labels: {sorted(missing)}")
    return atom_idx


def _build_site_idx(atom_idx: Dict[str, int]) -> Dict[str, int]:
    return {
        "C2": atom_idx["C2"],
        "C3": atom_idx["C3"],
        "C6": atom_idx["C6"],
        "O2": atom_idx["O2"],
        "O3": atom_idx["O3"],
        "O6": atom_idx["O6"],
    }


def make_glucose_template(
    polymer: PolymerKind,
    monomer_representation: MonomerRepresentation = "anhydro",
) -> GlucoseMonomerTemplate:
    """Return a deterministic monomer graph + label mapping."""
    mol = _make_mol_from_polymer(polymer, monomer_representation)
    atom_idx = _build_atom_idx(mol, monomer_representation)
    site_idx = _build_site_idx(atom_idx)
    return GlucoseMonomerTemplate(
        mol=mol,
        atom_idx=atom_idx,
        site_idx=site_idx,
        polymer=polymer,
        representation=monomer_representation,
    )
