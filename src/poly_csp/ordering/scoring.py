from __future__ import annotations
import numpy as np
from poly_csp.config.schema import HelixSpec
from poly_csp.geometry.transform import ScrewTransform

def min_interatomic_distance(coords: np.ndarray, heavy_mask: np.ndarray) -> float:
    idx = np.where(heavy_mask)[0]
    X = coords[idx]
    # O(N^2) naive for now; fine for small DP smoke tests
    dmin = float("inf")
    for i in range(len(X)):
        diffs = X[i+1:] - X[i]
        d2 = np.sum(diffs * diffs, axis=1)
        if d2.size:
            dmin = min(dmin, float(np.sqrt(np.min(d2))))
    return dmin

def screw_symmetry_rmsd(coords: np.ndarray, residue_atom_count: int, helix: HelixSpec, k: int = 1) -> float:
    """
    Compare residue 0 to residue k mapped back by inverse screw, RMSD on atoms.
    This is a minimal gate; later you’ll likely compare many residues and average.
    """
    if coords.shape[0] < (k + 1) * residue_atom_count:
        return 0.0

    screw = ScrewTransform(theta_rad=helix.theta_rad, rise_A=helix.rise_A)

    res0 = coords[0:residue_atom_count]
    resk = coords[k*residue_atom_count:(k+1)*residue_atom_count]

    # Map residue k back to residue 0 by applying inverse screw k steps
    # Inverse: rotate by -theta, translate by -rise
    inv = ScrewTransform(theta_rad=-helix.theta_rad, rise_A=-helix.rise_A)
    resk_mapped = inv.apply(resk, k)

    diff = res0 - resk_mapped
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))