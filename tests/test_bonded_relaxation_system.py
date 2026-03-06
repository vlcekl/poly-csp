"""Tests for build_bonded_relaxation_system — generic bonded forces from RDKit."""

from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from poly_csp.mm.openmm_system import (  # noqa: E402
    build_bonded_relaxation_system,
    build_relaxation_system,
)


def _make_ethanol() -> Chem.Mol:
    """Return an RDKit ethanol molecule with 3-D coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _make_small_amide() -> Chem.Mol:
    """Return N-methylacetamide (sp2 centre) with 3-D coordinates."""
    mol = Chem.MolFromSmiles("CNC(C)=O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


# ---- Force composition tests ------------------------------------------------

class TestForceComposition:
    """Verify the returned system has the right forces and counts."""

    def test_has_bond_angle_repulsion(self) -> None:
        mol = _make_ethanol()
        result = build_bonded_relaxation_system(mol)
        system = result.system

        force_types = [
            type(system.getForce(i)).__name__
            for i in range(system.getNumForces())
        ]
        assert "HarmonicBondForce" in force_types
        assert "HarmonicAngleForce" in force_types
        assert "CustomNonbondedForce" in force_types

    def test_bond_count_matches_rdkit(self) -> None:
        mol = _make_ethanol()
        result = build_bonded_relaxation_system(mol)
        bond_force = None
        for i in range(result.system.getNumForces()):
            f = result.system.getForce(i)
            if isinstance(f, openmm.HarmonicBondForce):
                bond_force = f
                break
        assert bond_force is not None
        assert bond_force.getNumBonds() == mol.GetNumBonds()

    def test_angle_count_positive(self) -> None:
        mol = _make_ethanol()
        result = build_bonded_relaxation_system(mol)
        angle_force = None
        for i in range(result.system.getNumForces()):
            f = result.system.getForce(i)
            if isinstance(f, openmm.HarmonicAngleForce):
                angle_force = f
                break
        assert angle_force is not None
        assert angle_force.getNumAngles() > 0

    def test_particle_count_matches(self) -> None:
        mol = _make_ethanol()
        result = build_bonded_relaxation_system(mol)
        assert result.system.getNumParticles() == mol.GetNumAtoms()

    def test_more_forces_than_unbonded_system(self) -> None:
        mol = _make_ethanol()
        bonded = build_bonded_relaxation_system(mol)
        unbonded = build_relaxation_system(mol)
        assert bonded.system.getNumForces() > unbonded.system.getNumForces()


# ---- Stability test ----------------------------------------------------------

class TestStability:
    """Run a short Langevin simulation and confirm the structure stays intact."""

    def test_short_langevin_does_not_explode(self) -> None:
        mol = _make_ethanol()
        result = build_bonded_relaxation_system(mol)

        integrator = openmm.LangevinIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        context = openmm.Context(result.system, integrator)
        context.setPositions(result.positions_nm)

        # Run 100 steps of dynamics.
        integrator.step(100)

        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        xyz_A = np.asarray(positions.value_in_unit(unit.nanometer)) * 10.0
        span = xyz_A.max(axis=0) - xyz_A.min(axis=0)

        # Ethanol should stay within a 200 Å box.
        assert np.all(span < 200.0), f"Structure exploded: span = {span}"

    def test_amide_sp2_angles_survive_dynamics(self) -> None:
        mol = _make_small_amide()
        result = build_bonded_relaxation_system(mol)

        integrator = openmm.LangevinIntegrator(
            350.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        context = openmm.Context(result.system, integrator)
        context.setPositions(result.positions_nm)

        integrator.step(200)

        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        xyz_A = np.asarray(positions.value_in_unit(unit.nanometer)) * 10.0
        span = xyz_A.max(axis=0) - xyz_A.min(axis=0)
        assert np.all(span < 200.0), f"Amide structure exploded: span = {span}"


# ---- Edge cases --------------------------------------------------------------

class TestEdgeCases:
    def test_raises_without_conformer(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        with pytest.raises(ValueError, match="coordinates"):
            build_bonded_relaxation_system(mol)
