from __future__ import annotations

import openmm as mm
from openmm import Vec3, app as mmapp, unit

from poly_csp.io.openmm_io import write_openmm_xml, write_pdb_from_openmm


def test_write_openmm_xml_writes_serialized_system(tmp_path) -> None:
    system = mm.System()
    system.addParticle(12.0)

    path = tmp_path / "system.xml"
    write_openmm_xml(system, path)

    text = path.read_text(encoding="utf-8")
    assert "<System" in text
    assert "<Particles>" in text


def test_write_pdb_from_openmm_writes_atoms(tmp_path) -> None:
    top = mmapp.Topology()
    chain = top.addChain("A")
    res = top.addResidue("MOL", chain)
    top.addAtom("C1", mmapp.element.carbon, res)

    positions = [Vec3(0.0, 0.0, 0.0)] * unit.nanometer

    path = tmp_path / "model.pdb"
    write_pdb_from_openmm(top, positions, path)

    text = path.read_text(encoding="utf-8")
    assert ("ATOM" in text) or ("HETATM" in text)
    assert "END" in text
