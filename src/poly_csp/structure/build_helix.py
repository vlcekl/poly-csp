# poly_csp/structure/build_helix.py
from __future__ import annotations

import numpy as np

from poly_csp.topology.monomers import GlucoseMonomerTemplate
from poly_csp.config.schema import HelixSpec
from poly_csp.structure.matrix import ScrewTransform


def _template_coords(template: GlucoseMonomerTemplate) -> np.ndarray:
    if template.mol.GetNumConformers() == 0:
        raise ValueError("Template molecule has no conformer coordinates.")

    conf = template.mol.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).reshape((-1, 3))

    ring_labels = ("C1", "C2", "C3", "C4", "C5", "O5")
    ring_idx = [template.atom_idx[name] for name in ring_labels]
    ring_centroid = coords[ring_idx].mean(axis=0)

    # Move residue 0 ring centroid to a fixed radial offset from the helix axis.
    coords0 = coords - ring_centroid
    coords0 += np.array([3.0, 0.0, 0.0], dtype=float)
    return coords0


def build_backbone_coords(
    template: GlucoseMonomerTemplate,
    helix: HelixSpec,
    dp: int,
) -> np.ndarray:
    """
    Deterministically build concatenated coords for dp residues by screw replication.
    Returns shape (dp*n_atoms, 3).
    """
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")

    coords0 = _template_coords(template)
    screw = ScrewTransform(theta_rad=helix.theta_rad, rise_A=helix.rise_A)
    blocks = [screw.apply(coords0, i) for i in range(dp)]
    return np.concatenate(blocks, axis=0)
