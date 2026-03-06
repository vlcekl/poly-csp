from __future__ import annotations

from pathlib import Path

from openmm import XmlSerializer, app as mmapp


def write_openmm_xml(system, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(XmlSerializer.serialize(system), encoding="utf-8")


def write_pdb_from_openmm(topology, positions, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        mmapp.PDBFile.writeFile(topology, positions, handle)
