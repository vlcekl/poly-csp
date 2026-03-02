"""Reference helix presets.

This module is currently reference-only and is not the runtime source of truth
for pipeline defaults. Active defaults are managed by Hydra YAML files in
`conf/helix/`.
"""

from __future__ import annotations
import math
from poly_csp.config.schema import HelixSpec

def _make_from_residues_per_turn(
    name: str,
    residues_per_turn: float,
    rise_A: float,
    handedness: str = "right",
) -> HelixSpec:
    # rotation per residue (magnitude)
    theta = 2.0 * math.pi / residues_per_turn
    # encode handedness by sign convention on theta:
    # right-handed: +theta, left-handed: -theta
    theta = theta if handedness == "right" else -theta

    pitch_A = rise_A * residues_per_turn
    return HelixSpec(
        name=name,
        theta_rad=theta,
        rise_A=rise_A,
        residues_per_turn=residues_per_turn,
        pitch_A=pitch_A,
        handedness=handedness,  # informational
    )

# --- Cellulose: 2_1 screw (180° per residue), ~10.3 Å repeat per 2 residues.
cellulose_i_2_1 = _make_from_residues_per_turn(
    name="cellulose_I_2_1",
    residues_per_turn=2.0,
    rise_A=5.15,
    handedness="right",  # 2_1 is effectively non-chiral; choose a sign convention and stick to it.
)

# --- Amylose V-helix: 6 residues per turn, pitch ~7.8–7.9 Å.
amylose_v6_1 = _make_from_residues_per_turn(
    name="amylose_V_6_1",
    residues_per_turn=6.0,
    rise_A=7.8 / 6.0,  # ≈1.30 Å per residue
    handedness="left",
)

# --- Amylose tris(3,5-DMPC): left-handed 4/3 helix starting guess.
# The exact pitch/rise is solvent- and substitution-dependent; use as a tunable *starting guess*.
# If helix is "4/3", interpret as 4 residues per 3 turns -> rotation per residue = 360*(3/4)=270° = 3π/2.
# This produces a strong per-residue twist (often effectively -90° depending on convention); treat as experimental-calibrated later.
amylose_admpc_4_3_guess = HelixSpec(
    name="amylose_ADMPC_4_3_guess",
    theta_rad=-3.0 * math.pi / 2.0,  # left-handed sign convention
    rise_A=2.6,                      # placeholder; tune to match target density/packing
    residues_per_turn=4.0 / 3.0,     # turns per 360° is inverted; keep for metadata consistency if you want
    pitch_A=2.6 * (4.0 / 3.0),
    handedness="left",
)

def admcp_chiralpak_ad_default() -> HelixSpec:
    """
    Amylose tris(3,5-dimethylphenylcarbamate) (ADMPC), Chiralpak AD-like.
    - Left-handed 4/3 helix (repeat: 4 residues, 3 turns).
    - Rise/contour length per residue h ~ 0.36–0.38 nm => ~3.6–3.8 Å.
    """
    repeat_residues = 4
    repeat_turns = 3

    # Left-handed sign convention: negative theta about +z as z increases.
    theta_rad = -2.0 * math.pi * (repeat_turns / repeat_residues)  # = -3π/2 per residue

    # Choose mid-point of reported h-range as a default; expose for tuning.
    rise_A = 3.7  # Å per residue (0.37 nm)

    residues_per_turn = repeat_residues / repeat_turns  # 4/3
    pitch_A = rise_A * residues_per_turn                # ~4.93 Å per 360°

    return HelixSpec(
        name="amylose_ADMPC_ChiralpakAD_4over3_default",
        theta_rad=theta_rad,
        rise_A=rise_A,
        repeat_residues=repeat_residues,
        repeat_turns=repeat_turns,
        residues_per_turn=residues_per_turn,
        pitch_A=pitch_A,
        handedness="left",
    )
