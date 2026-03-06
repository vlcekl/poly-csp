from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import openmm as mm
from openmm import unit


def add_positional_restraints(
    system: mm.System,
    atom_indices: Iterable[int],
    reference_positions_nm: unit.Quantity,
    k_kj_per_mol_nm2: float,
) -> mm.CustomExternalForce:
    force = mm.CustomExternalForce("0.5*k_pos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k_pos", float(k_kj_per_mol_nm2))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    ref = reference_positions_nm.value_in_unit(unit.nanometer)
    for idx in atom_indices:
        x0, y0, z0 = ref[int(idx)]
        force.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(force)
    return force


def add_dihedral_restraints(
    system: mm.System,
    dihedrals: Sequence[tuple[int, int, int, int, float]],
    k_kj_per_mol: float,
) -> mm.CustomTorsionForce:
    force = mm.CustomTorsionForce("k_tors*(1-cos(theta-theta0))")
    force.addGlobalParameter("k_tors", float(k_kj_per_mol))
    force.addPerTorsionParameter("theta0")
    for a, b, c, d, theta0 in dihedrals:
        force.addTorsion(int(a), int(b), int(c), int(d), [float(theta0)])
    system.addForce(force)
    return force


def add_hbond_distance_restraints(
    system: mm.System,
    pairs: Sequence[tuple[int, int, float]],
    k_kj_per_mol_nm2: float,
) -> mm.CustomBondForce:
    force = mm.CustomBondForce("0.5*k_hb*(r-r0)^2")
    force.addGlobalParameter("k_hb", float(k_kj_per_mol_nm2))
    force.addPerBondParameter("r0")
    for a, b, r0 in pairs:
        force.addBond(int(a), int(b), [float(r0)])
    system.addForce(force)
    return force
