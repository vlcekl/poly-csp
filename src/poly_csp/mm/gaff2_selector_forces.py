# poly_csp/mm/gaff2_selector_forces.py
"""Transfer GAFF2 bonded forces from a selector prmtop to the full molecule.

This module solves the per-fragment mapping problem: each selector instance
in the polymer is an identical copy of the template, so we load the prmtop
once and replicate its forces for every instance, remapping atom indices
through the ``_poly_csp_selector_local_idx`` property.

Dummy atom handling
-------------------
The selector template SMILES includes a ``[*]`` dummy atom at
``attach_dummy_idx``.  During GAFF2 parameterisation the dummy is replaced
with H, so the prmtop has the *same atom count* as the original template.
After attachment to the polymer the dummy atom is **removed**, so the
corresponding ``local_idx`` is absent from the per-instance mapping.
Any prmtop force term that involves the dummy index is skipped.
"""
from __future__ import annotations

import logging
from typing import Dict

from rdkit import Chem

import openmm as mm
from openmm import app as mmapp

from poly_csp.chemistry.selectors import SelectorTemplate
from poly_csp.mm.openmm_system import _covalent_bond_length_nm, _equilibrium_angle_rad

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-instance mapping helpers
# ---------------------------------------------------------------------------

def _selector_mappings(mol: Chem.Mol) -> Dict[int, Dict[int, int]]:
    """Return {instance_id → {local_idx → global_idx}} for every selector instance."""
    mappings: Dict[int, Dict[int, int]] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("_poly_csp_selector_instance"):
            continue
        inst = int(atom.GetIntProp("_poly_csp_selector_instance"))
        local = int(atom.GetIntProp("_poly_csp_selector_local_idx"))
        mappings.setdefault(inst, {})[local] = atom.GetIdx()
    return mappings


def _selector_all_indices(mol: Chem.Mol) -> set[int]:
    """Set of all selector atom indices in the polymer."""
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("_poly_csp_selector_instance")
    }


# ---------------------------------------------------------------------------
# Load GAFF2 forces from prmtop and remap to molecule indices
# ---------------------------------------------------------------------------

def load_gaff2_selector_forces(
    selector_prmtop_path: str,
    mol: Chem.Mol,
    selector_template: SelectorTemplate,
) -> list[mm.Force]:
    """Load GAFF2 bonded forces from the selector prmtop and remap indices.

    Returns a list of OpenMM Force objects (HarmonicBondForce,
    HarmonicAngleForce, PeriodicTorsionForce) whose particle indices
    reference the full polymer molecule.

    Parameters
    ----------
    selector_prmtop_path
        Path to an AMBER prmtop file for the standalone selector fragment
        (with dummy atom replaced by H).
    mol
        The full polymer RDKit molecule with selector atoms annotated via
        ``_poly_csp_selector_instance`` and ``_poly_csp_selector_local_idx``.
    selector_template
        The SelectorTemplate that defines ``attach_dummy_idx``.

    Returns
    -------
    List of OpenMM Force objects ready to be added to a System.  Typically
    contains one HarmonicBondForce, one HarmonicAngleForce, and one or two
    PeriodicTorsionForce objects (proper torsions + impropers).
    """
    dummy_idx = selector_template.attach_dummy_idx  # may be None

    # 1. Load the prmtop and create an OpenMM system to extract the forces.
    prmtop = mmapp.AmberPrmtopFile(selector_prmtop_path)
    ref_system = prmtop.createSystem()

    # 2. Extract per-instance mappings: {instance_id → {local_idx → global_idx}}.
    #    local_idx == prmtop atom index (except for the dummy, which is absent).
    mappings = _selector_mappings(mol)
    if not mappings:
        log.warning("No selector instances found in molecule; returning empty forces.")
        return []

    n_instances = len(mappings)
    log.info(
        "Transferring GAFF2 forces from %s to %d selector instance(s).",
        selector_prmtop_path, n_instances,
    )

    # 3. For each force type in the reference system, create a corresponding
    #    force for the full molecule and replicate every term for each instance.
    out_forces: list[mm.Force] = []

    for force_idx in range(ref_system.getNumForces()):
        ref_force = ref_system.getForce(force_idx)

        if isinstance(ref_force, mm.HarmonicBondForce):
            out_bond = mm.HarmonicBondForce()
            for inst_map in mappings.values():
                _transfer_bonds(ref_force, inst_map, dummy_idx, out_bond)
            out_forces.append(out_bond)
            log.info("  HarmonicBondForce: %d bonds", out_bond.getNumBonds())

        elif isinstance(ref_force, mm.HarmonicAngleForce):
            out_angle = mm.HarmonicAngleForce()
            for inst_map in mappings.values():
                _transfer_angles(ref_force, inst_map, dummy_idx, out_angle)
            out_forces.append(out_angle)
            log.info("  HarmonicAngleForce: %d angles", out_angle.getNumAngles())

        elif isinstance(ref_force, mm.PeriodicTorsionForce):
            out_torsion = mm.PeriodicTorsionForce()
            for inst_map in mappings.values():
                _transfer_torsions(ref_force, inst_map, dummy_idx, out_torsion)
            out_forces.append(out_torsion)
            log.info("  PeriodicTorsionForce: %d torsions", out_torsion.getNumTorsions())

        # NonbondedForce, CMMotionRemover, etc. are intentionally skipped —
        # we use soft repulsion instead.

    return out_forces


def _remap_idx(prmtop_idx: int, inst_map: Dict[int, int], dummy_idx: int | None) -> int | None:
    """Map a prmtop atom index to a global molecule index.

    Returns None if the atom is the dummy (not present in the polymer).
    The mapping is straightforward: prmtop indices correspond 1:1 to
    template local_idx values, and the instance mapping gives us global_idx.
    """
    if dummy_idx is not None and prmtop_idx == dummy_idx:
        return None
    local_idx = prmtop_idx
    return inst_map.get(local_idx)


def _transfer_bonds(
    ref_force: mm.HarmonicBondForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.HarmonicBondForce,
) -> None:
    """Copy bond terms from the reference force, remapping indices."""
    for bi in range(ref_force.getNumBonds()):
        p1, p2, r0, k = ref_force.getBondParameters(bi)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        if g1 is None or g2 is None:
            continue  # involves dummy atom → skip
        out_force.addBond(g1, g2, r0, k)


def _transfer_angles(
    ref_force: mm.HarmonicAngleForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.HarmonicAngleForce,
) -> None:
    """Copy angle terms from the reference force, remapping indices."""
    for ai in range(ref_force.getNumAngles()):
        p1, p2, p3, theta0, k = ref_force.getAngleParameters(ai)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        g3 = _remap_idx(p3, inst_map, dummy_idx)
        if g1 is None or g2 is None or g3 is None:
            continue
        out_force.addAngle(g1, g2, g3, theta0, k)


def _transfer_torsions(
    ref_force: mm.PeriodicTorsionForce,
    inst_map: Dict[int, int],
    dummy_idx: int | None,
    out_force: mm.PeriodicTorsionForce,
) -> None:
    """Copy torsion terms from the reference force, remapping indices."""
    for ti in range(ref_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = ref_force.getTorsionParameters(ti)
        g1 = _remap_idx(p1, inst_map, dummy_idx)
        g2 = _remap_idx(p2, inst_map, dummy_idx)
        g3 = _remap_idx(p3, inst_map, dummy_idx)
        g4 = _remap_idx(p4, inst_map, dummy_idx)
        if g1 is None or g2 is None or g3 is None or g4 is None:
            continue
        out_force.addTorsion(g1, g2, g3, g4, periodicity, phase, k)


# ---------------------------------------------------------------------------
# Junction forces: bonds/angles crossing the backbone-selector boundary
# ---------------------------------------------------------------------------

def build_junction_forces(
    mol: Chem.Mol,
    selector_indices: set[int],
    bond_k: float = 200_000.0,
    angle_k: float = 500.0,
) -> tuple[mm.HarmonicBondForce, mm.HarmonicAngleForce]:
    """Build bonded forces for bonds/angles crossing the backbone↔selector boundary.

    These are the attachment bonds (e.g. sugar-O—carbonyl-C) and angles
    where atoms span both sides.  Generic covalent-radius parameters are
    used since the GAFF2 prmtop does not cover these cross-boundary terms.

    Returns
    -------
    (bond_force, angle_force) covering only the junction terms.
    """
    bond_force = mm.HarmonicBondForce()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        i_sel = i in selector_indices
        j_sel = j in selector_indices
        if i_sel == j_sel:
            continue  # both backbone or both selector → not a junction
        z1 = mol.GetAtomWithIdx(i).GetAtomicNum()
        z2 = mol.GetAtomWithIdx(j).GetAtomicNum()
        r0 = _covalent_bond_length_nm(z1, z2)
        bond_force.addBond(i, j, r0, bond_k)

    # Build adjacency list for angle enumeration.
    n = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    angle_force = mm.HarmonicAngleForce()
    for j_atom in range(n):
        nbrs = adj[j_atom]
        theta0 = _equilibrium_angle_rad(mol.GetAtomWithIdx(j_atom))
        j_sel = j_atom in selector_indices
        for ii in range(len(nbrs)):
            for jj in range(ii + 1, len(nbrs)):
                a, b = nbrs[ii], nbrs[jj]
                a_sel = a in selector_indices
                b_sel = b in selector_indices
                # Include only if atoms span the boundary.
                sides = {a_sel, j_sel, b_sel}
                if len(sides) < 2:
                    continue  # all on same side → skip
                angle_force.addAngle(a, j_atom, b, theta0, angle_k)

    return bond_force, angle_force
