# poly_csp/geometry/local_frames.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from poly_csp.config.schema import SelectorPoseSpec


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Degenerate vector with near-zero norm.")
    return v / n


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_u = _normalize(a)
    b_u = _normalize(b)
    c = float(np.dot(a_u, b_u))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180-degree rotation around arbitrary axis orthogonal to a.
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, a_u))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize(np.cross(a_u, trial))
        x, y, z = axis
        return np.array(
            [
                [2 * x * x - 1, 2 * x * y, 2 * x * z],
                [2 * x * y, 2 * y * y - 1, 2 * y * z],
                [2 * x * z, 2 * y * z, 2 * z * z - 1],
            ],
            dtype=float,
        )

    v = np.cross(a_u, b_u)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _selector_basis(centered: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    e0, e1, _ = vh

    centroid_vec = centered.mean(axis=0)
    if float(np.dot(e0, centroid_vec)) < 0.0:
        e0 = -e0

    far_idx = int(np.argmax(np.linalg.norm(centered, axis=1)))
    ref = centered[far_idx]
    if float(np.dot(e1, ref)) < 0.0:
        e1 = -e1

    e2 = _normalize(np.cross(e0, e1))
    e1 = _normalize(np.cross(e2, e0))
    return np.vstack([_normalize(e0), e1, e2])


def compute_residue_local_frame(
    coords_res: np.ndarray,
    labels: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) mapping local->global:
      global_point = local_point @ R.T + t

    Local axes:
    - x: C1 -> O4 (or C1 -> C4 fallback)
    - z: ring normal from x cross (C1 -> C2) (fallback to C3)
    - y: z cross x
    """
    xyz = np.asarray(coords_res, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected coords_res shape (N,3); got {xyz.shape}")

    c1 = xyz[labels["C1"]]
    x_ref = xyz[labels["O4"]] if "O4" in labels else xyz[labels["C4"]]
    v_plane = xyz[labels["C2"]] - c1
    x_axis = _normalize(x_ref - c1)
    z_trial = np.cross(x_axis, v_plane)
    if np.linalg.norm(z_trial) < 1e-8 and "C3" in labels:
        z_trial = np.cross(x_axis, xyz[labels["C3"]] - c1)
    z_axis = _normalize(z_trial)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))

    r = np.vstack([x_axis, y_axis, z_axis])
    t = c1.copy()
    return r, t


def pose_selector_in_frame(
    selector_coords: np.ndarray,
    pose: SelectorPoseSpec,
    r_res: np.ndarray,
    t_res: np.ndarray,
    attach_atom_idx: int,
) -> np.ndarray:
    """
    Place selector coordinates in residue-local frame.
    Deterministic placement is derived from selector PCA basis and optional
    pose directional hint (carbonyl_dir_local).
    """
    xyz = np.asarray(selector_coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected selector_coords shape (N,3); got {xyz.shape}")
    if not (0 <= attach_atom_idx < xyz.shape[0]):
        raise ValueError(f"attach_atom_idx {attach_atom_idx} out of range.")

    centered = xyz - xyz[attach_atom_idx]
    basis = _selector_basis(centered)  # rows define selector-local axes in global selector frame
    local = centered @ basis.T

    if pose.carbonyl_dir_local is not None:
        desired = _normalize(np.array(pose.carbonyl_dir_local, dtype=float))
        current = _normalize(local.mean(axis=0))
        rot = _rotation_from_a_to_b(current, desired)
        local = local @ rot.T

    return local @ r_res.T + t_res
