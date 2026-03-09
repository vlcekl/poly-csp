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


def add_explicit_positional_restraints(
    system: mm.System,
    atom_indices: Sequence[int],
    reference_positions_A: Sequence[Sequence[float]],
    k_kj_per_mol_nm2: float,
    *,
    parameter_name: str,
) -> mm.CustomExternalForce:
    ref_A = np.asarray(reference_positions_A, dtype=float)
    if ref_A.ndim != 2 or ref_A.shape[1] != 3:
        raise ValueError(
            "Explicit positional restraint references must have shape (N, 3)."
        )
    if ref_A.shape[0] != len(atom_indices):
        raise ValueError(
            "Explicit positional restraint atom/reference count mismatch: "
            f"{len(atom_indices)} atoms vs {ref_A.shape[0]} reference positions."
        )
    if not parameter_name or not parameter_name.replace("_", "").isalnum():
        raise ValueError(
            f"Explicit positional restraint parameter_name {parameter_name!r} is invalid."
        )

    force = mm.CustomExternalForce(
        f"0.5*{parameter_name}*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    force.addGlobalParameter(parameter_name, float(k_kj_per_mol_nm2))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    ref_nm = ref_A / 10.0
    for idx, (x0, y0, z0) in zip(atom_indices, ref_nm, strict=True):
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
