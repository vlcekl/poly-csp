# poly_csp/geometry/transform.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple

def rotation_matrix_z(theta_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotation about +z."""
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigid alignment mapping P -> Q (least squares).
    Returns (R, t) such that P @ R.T + t ~ Q
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P{P.shape} vs Q{Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected (N,3) arrays, got P{P.shape}")
    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points for rigid alignment.")

    p_centroid = P.mean(axis=0)
    q_centroid = Q.mean(axis=0)

    p_centered = P - p_centroid
    q_centered = Q - q_centroid

    h = p_centered.T @ q_centered
    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T

    # Enforce a proper rotation (det=+1) and avoid reflections.
    if np.linalg.det(r) < 0:
        vh[-1, :] *= -1.0
        r = vh.T @ u.T

    t = q_centroid - p_centroid @ r.T
    return r, t

@dataclass(frozen=True)
class ScrewTransform:
    theta_rad: float
    rise_A: float

    def matrix(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) for the i-th screw application."""
        step = int(i)
        r = rotation_matrix_z(step * self.theta_rad)
        t = np.array([0.0, 0.0, step * self.rise_A], dtype=float)
        return r, t

    def apply(self, points: np.ndarray, i: int) -> np.ndarray:
        """Apply i screw steps to (N,3) points."""
        x = np.asarray(points, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected (N,3) points, got {x.shape}")
        r, t = self.matrix(i)
        return x @ r.T + t
