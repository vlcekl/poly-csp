"""Tests for split-system force assembly (AMBER backbone + generic selectors)."""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from poly_csp.forcefield.system_builder import (  # noqa: E402
    build_selector_bonded_forces,
)


def _make_tagged_mol() -> Chem.Mol:
    """Build a small molecule with backbone and 'selector' atoms tagged.

    We use ethanol (CC-O) as backbone and attach a methyl-like group as
    a fake selector for testing.  Total: propanol CH3-CH2-CH2-OH.
    """
    mol = Chem.MolFromSmiles("CCCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)

    # Tag atom 0 (first carbon) as a selector atom.
    atom = mol.GetAtomWithIdx(0)
    atom.SetIntProp("_poly_csp_selector_instance", 0)
    atom.SetIntProp("_poly_csp_selector_local_idx", 0)
    # Also tag its hydrogens.
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            nbr.SetIntProp("_poly_csp_selector_instance", 0)
            nbr.SetIntProp("_poly_csp_selector_local_idx", nbr.GetIdx())

    return mol


class TestSelectorBondedForces:
    """Verify that build_selector_bonded_forces filters correctly."""

    def test_only_selector_bonds_included(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_selector_instance")
        }
        bond_force, angle_force = build_selector_bonded_forces(mol, selector_indices)

        # Should include bonds involving selector atoms but NOT pure backbone bonds.
        assert bond_force.getNumBonds() > 0
        assert bond_force.getNumBonds() < mol.GetNumBonds()

    def test_junction_bonds_included(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_selector_instance")
        }
        bond_force, _ = build_selector_bonded_forces(mol, selector_indices)

        # The bond connecting selector atom 0 to backbone atom 1 should be present.
        found_junction = False
        for bi in range(bond_force.getNumBonds()):
            p1, p2, _, _ = bond_force.getBondParameters(bi)
            if (p1 in selector_indices) != (p2 in selector_indices):
                found_junction = True
                break
        assert found_junction, "Junction bond between backbone and selector not found"

    def test_angles_positive(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_selector_instance")
        }
        _, angle_force = build_selector_bonded_forces(mol, selector_indices)
        assert angle_force.getNumAngles() > 0

    def test_no_pure_backbone_bonds(self) -> None:
        mol = _make_tagged_mol()
        selector_indices = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp("_poly_csp_selector_instance")
        }
        bond_force, _ = build_selector_bonded_forces(mol, selector_indices)

        for bi in range(bond_force.getNumBonds()):
            p1, p2, _, _ = bond_force.getBondParameters(bi)
            assert p1 in selector_indices or p2 in selector_indices, (
                f"Pure backbone bond ({p1}, {p2}) should not be in selector forces"
            )


class TestBackboneFreezing:
    """Verify backbone freezing via mass=0."""

    def test_frozen_backbone_does_not_move(self) -> None:
        """Run dynamics with backbone masses set to 0; verify they don't move."""
        from poly_csp.forcefield.system_builder import (
            build_bonded_relaxation_system,
            _atomic_mass_dalton,
        )

        mol = _make_tagged_mol()
        result = build_bonded_relaxation_system(mol)
        system = result.system

        # Identify backbone atoms (no selector tag).
        backbone_indices = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if not atom.HasProp("_poly_csp_selector_instance")
        ]

        # Save initial backbone positions.
        init_xyz = np.asarray(
            result.positions_nm.value_in_unit(unit.nanometer)
        )
        init_backbone = init_xyz[backbone_indices].copy()

        # Freeze backbone.
        for idx in backbone_indices:
            system.setParticleMass(idx, 0.0)

        integrator = openmm.LangevinIntegrator(
            350.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        context = openmm.Context(system, integrator)
        context.setPositions(result.positions_nm)

        integrator.step(200)

        state = context.getState(getPositions=True)
        final_xyz = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        final_backbone = final_xyz[backbone_indices]

        # Backbone atoms should not have moved at all.
        drift = np.max(np.abs(final_backbone - init_backbone))
        assert drift < 1e-6, f"Backbone drifted by {drift} nm despite mass=0"
