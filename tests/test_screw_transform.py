from __future__ import annotations

import numpy as np

from poly_csp.structure.matrix import ScrewTransform, kabsch_align, rotation_matrix_z


def test_rotation_matrix_z_orthonormality() -> None:
    theta = 1.234
    r = rotation_matrix_z(theta)
    identity = np.eye(3)
    assert np.allclose(r.T @ r, identity, atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_screw_group_property() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 3))

    screw = ScrewTransform(theta_rad=0.713, rise_A=3.7)
    i = 4
    j = -3

    lhs = screw.apply(x, i + j)
    rhs = screw.apply(screw.apply(x, i), j)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_screw_translation_along_z_matches_i_times_rise() -> None:
    screw = ScrewTransform(theta_rad=0.8, rise_A=5.15)
    r, t = screw.matrix(6)

    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.allclose(t[:2], np.array([0.0, 0.0]), atol=1e-12)
    assert np.isclose(t[2], 6.0 * 5.15, atol=1e-12)


def test_kabsch_align_recovers_rigid_transform() -> None:
    rng = np.random.default_rng(19)
    p = rng.standard_normal((20, 3))
    r_true = rotation_matrix_z(0.37)
    t_true = np.array([1.5, -0.8, 2.1])
    q = p @ r_true.T + t_true

    r_fit, t_fit = kabsch_align(p, q)
    q_fit = p @ r_fit.T + t_fit

    assert np.allclose(q_fit, q, atol=1e-10)
    assert np.allclose(r_fit.T @ r_fit, np.eye(3), atol=1e-12)
