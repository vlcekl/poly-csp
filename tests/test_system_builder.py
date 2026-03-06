from __future__ import annotations

from poly_csp.config.schema import HelixSpec
from poly_csp.forcefield.system_builder import create_system
from poly_csp.structure.build_helix import build_backbone_coords
from poly_csp.topology.atom_mapping import build_atom_map
from poly_csp.topology.backbone import assign_conformer, polymerize
from poly_csp.topology.monomers import make_glucose_template
from poly_csp.topology.reactions import attach_selector
from poly_csp.topology.selector_library.dmpc_35 import make_35_dmpc_template


def _helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def test_create_system_particle_count_matches_molecule() -> None:
    template = make_glucose_template("amylose")
    selector = make_35_dmpc_template()

    mol = polymerize(template=template, dp=2, linkage="1-4", anomer="alpha")
    mol = assign_conformer(mol, build_backbone_coords(template, _helix(), dp=2))
    mol = attach_selector(
        mol_polymer=mol,
        template=template,
        residue_index=0,
        site="C6",
        selector=selector,
    )

    atom_map = build_atom_map(mol)
    system = create_system(mol, atom_map=atom_map)

    assert system.getNumParticles() == mol.GetNumAtoms()
    assert system.getNumForces() > 0
