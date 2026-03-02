# poly_csp/config/schema.py
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field, PositiveInt, confloat


PolymerKind = Literal["amylose", "cellulose"]
Site = Literal["C2", "C3", "C6"]
Handedness = Literal["left", "right"]
MonomerRepresentation = Literal["anhydro", "natural_oh"]
EndMode = Literal["open", "capped", "periodic"]


class HelixSpec(BaseModel):
    """
    Defines a helix by a screw operation applied once per residue:
      - rotate about +z by theta_rad
      - translate along +z by rise_A
    """
    name: str

    # Screw parameters (per residue)
    theta_rad: float = Field(..., description="Rotation per residue about +z (radians).")
    rise_A: float = Field(..., description="Translation per residue along +z (angstrom).")

    # For tight helices like 4/3, store the rational form explicitly.
    repeat_residues: Optional[PositiveInt] = Field(None, description="Residues in helical repeat (e.g., 4).")
    repeat_turns: Optional[PositiveInt] = Field(None, description="Turns in helical repeat (e.g., 3).")

    # Informational/derived convenience fields
    residues_per_turn: confloat(gt=0) = Field(..., description="n = residues per 360° turn.")
    pitch_A: float = Field(..., description="Pitch (angstrom) per 360° turn.")

    handedness: Handedness = "right"
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Optional: torsion targets to keep polymerizable geometry
    # Use degrees here for human readability; convert internally.
    # (You’ll refine these once you lock your monomer atom labels.)
    glycosidic_phi_deg: Optional[float] = None
    glycosidic_psi_deg: Optional[float] = None
    glycosidic_omega_deg: Optional[float] = None


class BackboneSpec(BaseModel):
    polymer: PolymerKind
    dp: PositiveInt
    monomer_representation: MonomerRepresentation = "anhydro"
    end_mode: EndMode = "open"
    end_caps: Dict[str, str] = Field(default_factory=dict)
    helix: HelixSpec
    # ring pucker defaults to 4C1; use later if you add internal coordinate enforcement
    ring_pucker: Literal["4C1"] = "4C1"


class SelectorPoseSpec(BaseModel):
    """
    Deterministic initial pose rules for a selector in the residue-local frame.
    Keep minimal here; expand later.
    """
    # Example: initial dihedrals to apply after bonding (degrees)
    dihedral_targets_deg: Dict[str, float] = Field(default_factory=dict)

    # Optional directional preferences in residue-local frame
    # (unit vectors in local frame; use later if needed)
    carbonyl_dir_local: Optional[Tuple[float, float, float]] = None
    aromatic_normal_local: Optional[Tuple[float, float, float]] = None
