from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from poly_csp.chemistry.selectors import SelectorTemplate


_DMPC_35_MAPPED_SMILES = (
    "[*:1][C:2](=[O:3])[N:4][c:5]1[cH:6][c:7]([CH3:8])[cH:9][c:10]([CH3:11])[cH:12]1"
)


def _idx_from_mapnum(mol: Chem.Mol, map_num: int) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"Map number {map_num} not found.")


def _embed_if_needed(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() > 0:
        return mol

    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 3501
    status = AllChem.EmbedMolecule(with_h, params)
    if status != 0:
        status = AllChem.EmbedMolecule(with_h, useRandomCoords=True, randomSeed=3501)
    if status != 0:
        raise RuntimeError("RDKit failed to embed 3,5-DMPC selector template.")
    if all(atom.GetAtomicNum() > 0 for atom in with_h.GetAtoms()):
        AllChem.UFFOptimizeMolecule(with_h, maxIters=250)
    return Chem.RemoveHs(with_h, sanitize=True)


def make_35_dmpc_template() -> SelectorTemplate:
    mol = Chem.MolFromSmiles(_DMPC_35_MAPPED_SMILES)
    if mol is None:
        raise ValueError("Could not parse 3,5-DMPC mapped SMILES.")
    Chem.SanitizeMol(mol)
    mol = _embed_if_needed(mol)

    dummy = _idx_from_mapnum(mol, 1)
    carbonyl_c = _idx_from_mapnum(mol, 2)
    carbonyl_o = _idx_from_mapnum(mol, 3)
    amide_n = _idx_from_mapnum(mol, 4)
    c_ipso = _idx_from_mapnum(mol, 5)
    c_ortho = _idx_from_mapnum(mol, 6)
    c_meta = _idx_from_mapnum(mol, 7)
    c_para = _idx_from_mapnum(mol, 9)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    dihedrals = {
        "tau_link": (dummy, carbonyl_c, amide_n, c_ipso),
        "tau_ar": (carbonyl_c, amide_n, c_ipso, c_ortho),
        "tau_ring": (amide_n, c_ipso, c_meta, c_para),
    }

    return SelectorTemplate(
        name="35dmpc",
        mol=mol,
        attach_atom_idx=carbonyl_c,
        attach_dummy_idx=dummy,
        dihedrals=dihedrals,
        donors=(amide_n,),
        acceptors=(carbonyl_o,),
    )
