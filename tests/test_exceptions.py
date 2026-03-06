from __future__ import annotations

import openmm as mm
from openmm import unit

from poly_csp.forcefield.exceptions import apply_mixing_rules
from poly_csp.topology.atom_mapping import ComponentTag


def test_apply_mixing_rules_updates_exceptions_by_component() -> None:
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)

    nonbonded = mm.NonbondedForce()
    for _ in range(4):
        nonbonded.addParticle(0.0, 0.2, 0.1)

    qprod = 1.0 * unit.elementary_charge**2
    sigma = 0.2 * unit.nanometer
    eps = 0.5 * unit.kilojoule_per_mole

    idx_bb = nonbonded.addException(0, 1, qprod, sigma, eps)
    idx_sel = nonbonded.addException(2, 3, qprod, sigma, eps)
    idx_cross = nonbonded.addException(1, 2, qprod, sigma, eps)
    system.addForce(nonbonded)

    atom_map = {
        0: ComponentTag.BACKBONE,
        1: ComponentTag.BACKBONE,
        2: ComponentTag.SELECTOR,
        3: ComponentTag.SELECTOR,
    }
    summary = apply_mixing_rules(system, atom_map)
    assert summary["exceptions_seen"] == 3
    assert summary["exceptions_patched"] == 3

    _, _, q_bb, _, e_bb = nonbonded.getExceptionParameters(idx_bb)
    _, _, q_sel, _, e_sel = nonbonded.getExceptionParameters(idx_sel)
    _, _, q_cross, _, e_cross = nonbonded.getExceptionParameters(idx_cross)

    # backbone/cross use GLYCAM scaling (1.0/1.0), selector-selector keeps GAFF (1.2/2.0)
    assert abs(q_bb.value_in_unit(unit.elementary_charge**2) - 1.2) < 1e-12
    assert abs(q_sel.value_in_unit(unit.elementary_charge**2) - 1.0) < 1e-12
    assert abs(q_cross.value_in_unit(unit.elementary_charge**2) - 1.2) < 1e-12

    assert abs(e_bb.value_in_unit(unit.kilojoule_per_mole) - 1.0) < 1e-12
    assert abs(e_sel.value_in_unit(unit.kilojoule_per_mole) - 0.5) < 1e-12
    assert abs(e_cross.value_in_unit(unit.kilojoule_per_mole) - 1.0) < 1e-12
