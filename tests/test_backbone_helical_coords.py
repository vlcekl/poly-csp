from __future__ import annotations

import numpy as np

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.config.schema import HelixSpec
from poly_csp.geometry.transform import ScrewTransform


def _test_helix() -> HelixSpec:
    return HelixSpec(
        name="test_helix",
        theta_rad=-3.0 * np.pi / 2.0,
        rise_A=3.7,
        repeat_residues=4,
        repeat_turns=3,
        residues_per_turn=4.0 / 3.0,
        pitch_A=3.7 * (4.0 / 3.0),
        handedness="left",
    )


def test_build_backbone_coords_shape_and_symmetry() -> None:
    template = make_glucose_template("amylose")
    helix = _test_helix()
    dp = 6

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    n = template.mol.GetNumAtoms()
    assert coords.shape == (dp * n, 3)

    screw = ScrewTransform(theta_rad=helix.theta_rad, rise_A=helix.rise_A)
    res0 = coords[:n]
    for i in range(dp):
        resi = coords[i * n : (i + 1) * n]
        pred = screw.apply(res0, i)
        rmsd = np.sqrt(np.mean(np.sum((resi - pred) ** 2, axis=1)))
        assert rmsd < 1e-9


def test_ring_centroid_radius_is_constant_across_residues() -> None:
    template = make_glucose_template("amylose")
    helix = _test_helix()
    dp = 8

    coords = build_backbone_coords(template=template, helix=helix, dp=dp)
    n = template.mol.GetNumAtoms()
    ring_idx = [
        template.atom_idx["C1"],
        template.atom_idx["C2"],
        template.atom_idx["C3"],
        template.atom_idx["C4"],
        template.atom_idx["C5"],
        template.atom_idx["O5"],
    ]

    radii = []
    for i in range(dp):
        block = coords[i * n : (i + 1) * n]
        centroid = block[ring_idx].mean(axis=0)
        radii.append(float(np.linalg.norm(centroid[:2])))

    assert max(radii) - min(radii) < 1e-9
