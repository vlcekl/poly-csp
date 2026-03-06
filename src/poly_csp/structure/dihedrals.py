# poly_csp/structure/dihedrals.py
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    u = _normalize(axis)
    ux, uy, uz = u
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    oc = 1.0 - c
    return np.array(
        [
            [c + ux * ux * oc, ux * uy * oc - uz * s, ux * uz * oc + uy * s],
            [uy * ux * oc + uz * s, c + uy * uy * oc, uy * uz * oc - ux * s],
            [uz * ux * oc - uy * s, uz * uy * oc + ux * s, c + uz * uz * oc],
        ],
        dtype=float,
    )


def measure_dihedral_rad(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3), got {xyz.shape}")

    p0 = xyz[a]
    p1 = xyz[b]
    p2 = xyz[c]
    p3 = xyz[d]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_u = _normalize(b1)
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return float(np.arctan2(y, x))


def set_dihedral_rad(
    coords: np.ndarray,
    a: int,
    b: int,
    c: int,
    d: int,
    target_angle_rad: float,
    rotate_mask: np.ndarray,
) -> np.ndarray:
    """
    Rotate atoms in rotate_mask around bond (b,c) to achieve target dihedral (a,b,c,d).
    rotate_mask: boolean array length N marking atoms to rotate.
    """
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N,3), got {xyz.shape}")

    mask = np.asarray(rotate_mask, dtype=bool)
    if mask.shape != (xyz.shape[0],):
        raise ValueError(
            f"rotate_mask must have shape ({xyz.shape[0]},), got {mask.shape}"
        )
    if not mask[d]:
        raise ValueError("rotate_mask must include atom d on rotated side of bond.")

    current = measure_dihedral_rad(xyz, a, b, c, d)
    delta = _wrap_to_pi(float(target_angle_rad) - current)
    if abs(delta) < 1e-14:
        return xyz.copy()

    axis = xyz[c] - xyz[b]
    r = _rotation_matrix_axis_angle(axis, delta)
    origin = xyz[c]

    out = xyz.copy()
    idx = np.where(mask)[0]
    shifted = out[idx] - origin
    out[idx] = shifted @ r.T + origin
    return out
