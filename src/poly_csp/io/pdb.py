from __future__ import annotations
from pathlib import Path
from rdkit import Chem

def write_pdb_from_rdkit(mol: Chem.Mol, path: str | Path) -> None:
    path = Path(path)
    pdb = Chem.MolToPDBBlock(mol)
    path.write_text(pdb, encoding="utf-8")