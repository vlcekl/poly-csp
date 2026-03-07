from __future__ import annotations

import openmm as mm
from openmm import unit
from rdkit import Chem

from poly_csp.forcefield.exceptions import apply_mixing_rules


def _linear_component_mol(sources: list[str]) -> Chem.Mol:
    mol = Chem.RWMol()
    for source in sources:
        atom_idx = mol.AddAtom(Chem.Atom(6))
        mol.GetAtomWithIdx(atom_idx).SetProp("_poly_csp_manifest_source", source)
    for atom_idx in range(len(sources) - 1):
        mol.AddBond(atom_idx, atom_idx + 1, Chem.BondType.SINGLE)
    return mol.GetMol()


def _nonbonded_with_default_exceptions(mol: Chem.Mol) -> mm.NonbondedForce:
    nonbonded = mm.NonbondedForce()
    for atom_idx in range(mol.GetNumAtoms()):
        charge = 0.3 if atom_idx % 2 == 0 else -0.3
        nonbonded.addParticle(charge, 0.2, 0.5)
    bonds = [
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    ]
    nonbonded.createExceptionsFromBonds(bonds, 1.0, 1.0)
    return nonbonded


def test_apply_mixing_rules_patches_only_true_14_pairs_and_reports_classes() -> None:
    mol = _linear_component_mol(
        ["backbone", "backbone", "connector", "selector", "selector", "selector", "selector"]
    )
    nonbonded = _nonbonded_with_default_exceptions(mol)

    summary = apply_mixing_rules(nonbonded, mol)

    assert summary["baseline_scee"] == 1.0
    assert summary["baseline_scnb"] == 1.0
    assert summary["expected_14_pairs"] == 4
    assert summary["found_14_pairs"] == 4
    assert summary["patched_14_pairs"] == 1
    assert summary["counts_by_rule_bucket"] == {
        "backbone_backbone": 0,
        "selector_selector": 1,
        "cross_boundary": 3,
    }
    assert summary["counts_by_fine_pair_class"] == {
        "backbone_backbone": 0,
        "selector_selector": 1,
        "backbone_selector": 2,
        "backbone_connector": 0,
        "selector_connector": 1,
        "connector_connector": 0,
    }
    assert summary["connector_involving_pairs"] == 1

    exception_params = {
        tuple(sorted((int(a), int(b)))): (q, e)
        for a, b, q, _, e in (
            nonbonded.getExceptionParameters(i)
            for i in range(nonbonded.getNumExceptions())
        )
    }

    q_sel, e_sel = exception_params[(3, 6)]
    q_cross, e_cross = exception_params[(0, 3)]
    q_12, e_12 = exception_params[(0, 1)]
    q_13, e_13 = exception_params[(0, 2)]

    assert abs(q_sel.value_in_unit(unit.elementary_charge**2) + 0.075) < 1e-12
    assert abs(e_sel.value_in_unit(unit.kilojoule_per_mole) - 0.25) < 1e-12
    assert abs(q_cross.value_in_unit(unit.elementary_charge**2) + 0.09) < 1e-12
    assert abs(e_cross.value_in_unit(unit.kilojoule_per_mole) - 0.5) < 1e-12
    assert abs(q_12.value_in_unit(unit.elementary_charge**2)) < 1e-12
    assert abs(e_12.value_in_unit(unit.kilojoule_per_mole)) < 1e-12
    assert abs(q_13.value_in_unit(unit.elementary_charge**2)) < 1e-12
    assert abs(e_13.value_in_unit(unit.kilojoule_per_mole)) < 1e-12


def test_apply_mixing_rules_rejects_missing_manifest_metadata() -> None:
    mol = _linear_component_mol(["backbone", "backbone", "backbone", "selector"])
    mol.GetAtomWithIdx(3).ClearProp("_poly_csp_manifest_source")
    nonbonded = _nonbonded_with_default_exceptions(mol)

    try:
        apply_mixing_rules(nonbonded, mol)
    except ValueError as exc:
        assert "_poly_csp_manifest_source" in str(exc)
    else:
        raise AssertionError("Expected missing manifest metadata to fail.")


def test_apply_mixing_rules_rejects_missing_expected_14_exception() -> None:
    mol = _linear_component_mol(["backbone", "backbone", "backbone", "selector"])
    nonbonded = _nonbonded_with_default_exceptions(mol)
    missing_idx = None
    for exception_idx in range(nonbonded.getNumExceptions()):
        atom_a, atom_b, _, _, _ = nonbonded.getExceptionParameters(exception_idx)
        if tuple(sorted((int(atom_a), int(atom_b)))) == (0, 3):
            missing_idx = exception_idx
            break
    assert missing_idx is not None
    nonbonded.setExceptionParameters(
        missing_idx,
        0,
        0,
        0.0 * unit.elementary_charge**2,
        0.2 * unit.nanometer,
        0.0 * unit.kilojoule_per_mole,
    )

    try:
        apply_mixing_rules(nonbonded, mol)
    except ValueError as exc:
        assert "missing expected 1-4 exceptions" in str(exc)
    else:
        raise AssertionError("Expected missing 1-4 exception to fail.")
