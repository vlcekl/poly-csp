from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rdkit import Chem

from poly_csp.io.pdb import write_pdb_from_rdkit


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def export_amber_artifacts(
    mol: Chem.Mol,
    outdir: str | Path,
    model_name: str = "model",
    charge_model: str = "bcc",
    net_charge: int | str | None = "auto",
    polymer: str = "amylose",
    dp: int | None = None,
    selector_mol: Chem.Mol | None = None,
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> Dict[str, object]:
    """Export AMBER artifacts using residue-aware assembly.

    Uses GLYCAM06 backbone + GAFF2 selectors assembled residue-by-residue.
    Requires AmberTools with GLYCAM06j installed.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    return _export_residue_aware(
        mol=mol,
        out=out,
        model_name=model_name,
        charge_model=charge_model,
        net_charge=net_charge,
        polymer=polymer,
        dp=dp,
        selector_mol=selector_mol,
        periodic=periodic,
        box_vectors_A=box_vectors_A,
    )


def _export_residue_aware(
    mol: Chem.Mol,
    out: Path,
    model_name: str,
    charge_model: str,
    net_charge: int | str | None,
    polymer: str,
    dp: int | None,
    selector_mol: Chem.Mol | None,
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> Dict[str, object]:
    """Assemble AMBER topology using GLYCAM06 backbone + GAFF2 selectors."""
    from poly_csp.io.glycam_assembly import (
        build_linkage_frcmod,
        build_selector_prmtop,
        build_tleap_script,
        parameterize_selector_fragment,
        run_tleap_assembly,
    )

    if dp is None:
        if mol.HasProp("_poly_csp_dp"):
            dp = int(mol.GetIntProp("_poly_csp_dp"))
        else:
            raise ValueError(
                "residue_aware backend requires dp (degree of polymerization). "
                "Pass dp= or ensure the molecule has _poly_csp_dp metadata."
            )

    # Optionally parameterize the selector fragment
    selector_lib_path = None
    selector_frcmod_path = None
    selector_prmtop_path = None
    if selector_mol is not None:
        sel_dir = out / "selector_params"
        sel_artifacts = parameterize_selector_fragment(
            selector_mol=selector_mol,
            charge_model=charge_model,
            work_dir=sel_dir,
        )
        selector_lib_path = sel_artifacts["lib"]
        selector_frcmod_path = sel_artifacts["frcmod"]
        # Build standalone prmtop for GAFF2 force extraction.
        selector_prmtop_path = build_selector_prmtop(
            mol2_path=sel_artifacts["mol2"],
            frcmod_path=sel_artifacts["frcmod"],
            work_dir=sel_dir,
        )

    # Generate supplementary frcmod for missing GLYCAM06j linkage torsions.
    linkage_frcmod = build_linkage_frcmod(out)
    linkage_frcmod_path = str(linkage_frcmod.resolve())

    # Write the PDB for coordinate reference
    pdb_path = out / f"{model_name}.pdb"
    write_pdb_from_rdkit(mol, pdb_path)

    # Generate and run tleap script
    script = build_tleap_script(
        polymer=polymer,
        dp=dp,
        selector_lib_path=selector_lib_path,
        selector_frcmod_path=selector_frcmod_path,
        linkage_frcmod_path=linkage_frcmod_path,
        model_name=model_name,
        periodic=periodic,
        box_vectors_A=box_vectors_A,
    )
    assembly_result = run_tleap_assembly(
        tleap_script=script,
        outdir=out,
        model_name=model_name,
    )

    manifest_path = out / "amber_export.json"
    summary: Dict[str, object] = {
        "enabled": True,
        "parameterized": True,
        "charge_model": charge_model,
        "parameter_backend": "residue_aware",
        "polymer": polymer,
        "dp": dp,
        "files": {
            "pdb": str(pdb_path),
            "prmtop": assembly_result["prmtop"],
            "inpcrd": assembly_result["inpcrd"],
            "tleap_input": assembly_result["tleap_input"],
            "tleap_log": assembly_result["tleap_log"],
        },
        "notes": [
            "Assembled with GLYCAM06j backbone + GAFF2 selectors.",
            "Charges derived per fragment and replicated for symmetry.",
        ],
        "periodic": bool(periodic),
    }
    if box_vectors_A is not None:
        summary["box_vectors_A"] = list(box_vectors_A)
    if selector_lib_path:
        summary["files"]["selector_lib"] = selector_lib_path  # type: ignore[index]
        summary["files"]["selector_frcmod"] = selector_frcmod_path  # type: ignore[index]
    if selector_prmtop_path:
        summary["files"]["selector_prmtop"] = selector_prmtop_path  # type: ignore[index]
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary
