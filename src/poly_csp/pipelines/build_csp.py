# src/poly_csp/pipelines/build_csp.py
"""
Minimal working CSP build pipeline skeleton.

- Uses Hydra configs from /conf at repo root.
- Builds a helically symmetric amylose/cellulose backbone (coords only -> RDKit mol).
- Optionally attaches a selector (placeholder wiring; chemistry details live in functionalization.py).
- Writes PDB + JSON build report.

Run (from repo root):
  python -m poly_csp.pipelines.build_csp
  python -m poly_csp.pipelines.build_csp polymer.dp=24 helix=admcp_chiralpak_ad selector.enabled=true
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from poly_csp.chemistry.backbone_build import build_backbone_coords
from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.chemistry.polymerize import assign_conformer, polymerize
from poly_csp.chemistry.terminals import apply_terminal_mode
from poly_csp.config.schema import (
    BackboneSpec,
    HelixSpec,
    MonomerRepresentation,
    PolymerKind,
    SelectorPoseSpec,
    Site,
)
from poly_csp.io.pdb import write_pdb_from_rdkit
from poly_csp.ordering.scoring import min_interatomic_distance, screw_symmetry_rmsd

# Optional selector imports (keep pipeline runnable even before selector is implemented)
try:
    from poly_csp.chemistry.selectors import SelectorRegistry
    from poly_csp.chemistry.functionalization import (
        apply_selector_pose_dihedrals,
        attach_selector,
    )
except Exception:  # noqa: BLE001
    SelectorRegistry = None  # type: ignore
    apply_selector_pose_dihedrals = None  # type: ignore
    attach_selector = None  # type: ignore


@dataclass
class BuildReport:
    polymer: str
    dp: int
    monomer_representation: str
    end_mode: str
    end_caps: dict[str, str]
    helix_name: str
    theta_rad: float
    rise_A: float
    residues_per_turn: float
    pitch_A: float
    selector_enabled: bool
    selector_name: Optional[str]
    selector_sites: List[str]
    selector_pose_dihedral_targets_deg: dict[str, float]
    qc_min_heavy_distance_A: float
    qc_screw_symmetry_rmsd_A: float
    output_export_formats: List[str]


def _cfg_to_helixspec(cfg: DictConfig) -> HelixSpec:
    # Expect hydra config at cfg.helix.* fields.
    # This keeps the pipeline independent of how you structure YAML, as long as keys match.
    return HelixSpec(
        name=str(cfg.helix.name),
        theta_rad=float(cfg.helix.theta_rad),
        rise_A=float(cfg.helix.rise_A),
        repeat_residues=int(cfg.helix.repeat_residues) if "repeat_residues" in cfg.helix else None,
        repeat_turns=int(cfg.helix.repeat_turns) if "repeat_turns" in cfg.helix else None,
        residues_per_turn=float(cfg.helix.residues_per_turn),
        pitch_A=float(cfg.helix.pitch_A),
        handedness=str(cfg.helix.handedness) if "handedness" in cfg.helix else "right",
    )


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _heavy_atom_mask_from_rdkit(mol) -> np.ndarray:
    # Simple heavy atom mask: atomic number > 1
    mask = np.zeros((mol.GetNumAtoms(),), dtype=bool)
    for i, a in enumerate(mol.GetAtoms()):
        mask[i] = a.GetAtomicNum() > 1
    return mask


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    Required config fields (minimal):
      cfg.polymer.kind: "amylose"|"cellulose"
      cfg.polymer.dp: int
      cfg.helix.*: HelixSpec fields
      cfg.output.dir: str

    Optional:
      cfg.selector.enabled: bool
      cfg.selector.name: str
      cfg.selector.sites: list[str] e.g. ["C6"] or ["C2","C3","C6"]
    """
    # Resolve Hydra config to plain dict for logging/debug
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    print("=== poly_csp build_csp ===")
    print(OmegaConf.to_yaml(cfg))

    polymer_kind: PolymerKind = str(cfg.polymer.kind)  # type: ignore[assignment]
    dp: int = int(cfg.polymer.dp)
    monomer_representation: MonomerRepresentation = str(
        cfg.polymer.monomer_representation
        if "monomer_representation" in cfg.polymer
        else "anhydro"
    )  # type: ignore[assignment]
    end_mode = str(cfg.polymer.end_mode if "end_mode" in cfg.polymer else "open")
    end_caps = (
        dict(OmegaConf.to_container(cfg.polymer.end_caps, resolve=True))
        if "end_caps" in cfg.polymer and cfg.polymer.end_caps is not None
        else {}
    )
    output_export_formats = (
        [str(x) for x in cfg.output.export_formats]
        if "output" in cfg and "export_formats" in cfg.output
        else ["pdb"]
    )
    unsupported_formats = [fmt for fmt in output_export_formats if fmt not in {"pdb"}]
    if unsupported_formats:
        raise NotImplementedError(
            "Requested export formats are not implemented yet: "
            + ", ".join(sorted(set(unsupported_formats)))
        )

    helix = _cfg_to_helixspec(cfg)
    backbone = BackboneSpec(
        polymer=polymer_kind,
        dp=dp,
        monomer_representation=monomer_representation,
        end_mode=end_mode,  # type: ignore[arg-type]
        end_caps=end_caps,
        helix=helix,
    )

    # Output directory (Hydra changes cwd by default; use cfg.output.dir relative to original project root)
    # Hydra sets the run dir as current working directory. We'll write into it by default unless user sets absolute.
    outdir = _ensure_outdir(cfg.output.dir if "output" in cfg and "dir" in cfg.output else "outputs")

    # ---- Stage 1/2: backbone template + helical coords + polymerize
    template = make_glucose_template(
        backbone.polymer,
        monomer_representation=backbone.monomer_representation,
    )
    coords = build_backbone_coords(template=template, helix=backbone.helix, dp=backbone.dp)

    mol_poly = polymerize(
        template=template,
        dp=backbone.dp,
        linkage="1-4",
        anomer="alpha" if backbone.polymer == "amylose" else "beta",
    )
    mol_poly = apply_terminal_mode(
        mol=mol_poly,
        mode=backbone.end_mode,
        caps=backbone.end_caps,
        representation=backbone.monomer_representation,
    )
    removed_old = (
        json.loads(mol_poly.GetProp("_poly_csp_removed_old_indices_json"))
        if mol_poly.HasProp("_poly_csp_removed_old_indices_json")
        else []
    )
    coords_for_mol = coords
    if removed_old:
        keep_mask = np.ones((coords.shape[0],), dtype=bool)
        keep_mask[np.asarray(removed_old, dtype=int)] = False
        coords_for_mol = coords[keep_mask]
    mol_poly = assign_conformer(mol_poly, coords_for_mol)

    # ---- Stage 3 (optional): attach selectors
    selector_enabled = bool(getattr(cfg, "selector", {}).get("enabled", False)) if isinstance(cfg_resolved, dict) else False
    selector_name = None
    selector_sites: List[Site] = []
    selector_pose = SelectorPoseSpec()

    if selector_enabled:
        if (
            SelectorRegistry is None
            or attach_selector is None
            or apply_selector_pose_dihedrals is None
        ):
            raise RuntimeError(
                "Selector attachment requested but selector modules are not available. "
                "Implement chemistry/selectors.py and chemistry/functionalization.py."
            )

        selector_name = str(cfg.selector.name)
        selector_sites = [str(s) for s in cfg.selector.sites]  # type: ignore[assignment]
        selector = SelectorRegistry.get(selector_name)
        if "pose" in cfg.selector and cfg.selector.pose is not None:
            pose_payload = OmegaConf.to_container(cfg.selector.pose, resolve=True)
            if isinstance(pose_payload, dict):
                selector_pose = SelectorPoseSpec(**pose_payload)

        # Attach across all residues/sites and optionally apply requested pose dihedrals.
        for res_i in range(backbone.dp):
            for site in selector_sites:
                mol_poly = attach_selector(
                    mol_polymer=mol_poly,
                    template=template,
                    residue_index=res_i,
                    site=site,  # type: ignore[arg-type]
                    selector=selector,
                    mode="bond_from_OH_oxygen",
                )
                if selector_pose.dihedral_targets_deg:
                    mol_poly = apply_selector_pose_dihedrals(
                        mol=mol_poly,
                        residue_index=res_i,
                        site=site,  # type: ignore[arg-type]
                        pose_spec=selector_pose,
                        selector=selector,
                    )

    # ---- QC metrics
    conf = mol_poly.GetConformer(0)
    xyz = np.array(conf.GetPositions(), dtype=float)  # OpenMM Vec3-like -> numpy array; may be (N,3) already depending on RDKit build
    xyz = np.asarray(xyz).reshape((-1, 3))

    heavy_mask = _heavy_atom_mask_from_rdkit(mol_poly)
    qc_min_dist = float(min_interatomic_distance(xyz, heavy_mask))

    # screw symmetry RMSD for backbone: compare residue 0 vs residue k mapped by screw (k = repeat_residues if present else 1)
    n_atoms_per_res = template.mol.GetNumAtoms()
    k = int(helix.repeat_residues) if helix.repeat_residues else 1
    # Compute symmetry RMSD on the backbone-only coordinates so selector atoms
    # (which are appended after polymer atoms) do not pollute residue indexing.
    qc_sym_rmsd = float(
        screw_symmetry_rmsd(
            coords,
            residue_atom_count=n_atoms_per_res,
            helix=helix,
            k=k,
        )
    )

    report = BuildReport(
        polymer=str(backbone.polymer),
        dp=int(backbone.dp),
        monomer_representation=str(backbone.monomer_representation),
        end_mode=str(backbone.end_mode),
        end_caps=dict(backbone.end_caps),
        helix_name=str(helix.name),
        theta_rad=float(helix.theta_rad),
        rise_A=float(helix.rise_A),
        residues_per_turn=float(helix.residues_per_turn),
        pitch_A=float(helix.pitch_A),
        selector_enabled=bool(selector_enabled),
        selector_name=selector_name,
        selector_sites=[str(s) for s in selector_sites],
        selector_pose_dihedral_targets_deg=dict(selector_pose.dihedral_targets_deg),
        qc_min_heavy_distance_A=qc_min_dist,
        qc_screw_symmetry_rmsd_A=qc_sym_rmsd,
        output_export_formats=output_export_formats,
    )

    # ---- Write outputs
    pdb_path = outdir / "model.pdb"
    json_path = outdir / "build_report.json"
    cfg_path = outdir / "resolved_config.yaml"

    if "pdb" in output_export_formats:
        write_pdb_from_rdkit(mol_poly, pdb_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"\nWrote:\n  {pdb_path}\n  {json_path}\n  {cfg_path}")
    print("\nQC:")
    print(f"  min heavy-atom distance (Å): {qc_min_dist:.3f}")
    print(f"  screw symmetry RMSD (Å):     {qc_sym_rmsd:.3f}")


if __name__ == "__main__":
    main()
