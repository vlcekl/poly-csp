# poly_csp/forcefield/glycam.py
"""Residue-aware tleap assembly using GLYCAM06 + GAFF2.

Builds the AMBER topology by:
1. Loading GLYCAM06j for polysaccharide backbone residues.
2. Loading GAFF2 for selector fragments.
3. Assembling the system residue-by-residue with proper linkage records.

This replaces the monolithic antechamber approach from the original
``ambertools`` backend and ensures:
- Every glucose unit has identical GLYCAM06-derived parameters.
- Every selector has identical GAFF2-derived parameters.
- Charges are derived once per fragment and replicated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from rdkit import Chem

from poly_csp.io.pdb import write_pdb_from_rdkit


# GLYCAM06j residue codes for glucose units in 1->4 linked chains.
# Convention: first char = linkage to next residue, rest = sugar code.
# See GLYCAM06 parameter documentation.
GLYCAM_RESIDUE_NAMES = {
    ("amylose", "terminal_nonreducing"): "0GA",   # α-D-Glc, no downstream linkage
    ("amylose", "internal"):             "4GA",   # α-D-Glc, 1->4 linked
    ("amylose", "terminal_reducing"):    "4GA",   # reducing end (same type; capped separately)
    ("cellulose", "terminal_nonreducing"): "0GB",  # β-D-Glc
    ("cellulose", "internal"):             "4GB",  # β-D-Glc, 1->4 linked
    ("cellulose", "terminal_reducing"):    "4GB",
}


def _ensure_required_tools(tools: Sequence[str]) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            "GLYCAM assembly requires executables not found on PATH: "
            + ", ".join(missing)
            + ". Install AmberTools with GLYCAM06 support."
        )


def _run_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(
        list(cmd), cwd=str(cwd), text=True, capture_output=True, check=False,
    )
    combined = (
        f"$ {' '.join(cmd)}\n\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}. "
            f"See: {log_path}"
        )


def build_glycam_sequence(
    polymer: str,
    dp: int,
) -> List[str]:
    """Build a sequence of GLYCAM residue codes for a linear 1->4 chain.

    Returns a list of residue names, one per glucose unit:
    - First residue: terminal (non-reducing end)
    - Middle residues: internal
    - Last residue: terminal (reducing end)
    """
    if dp < 1:
        raise ValueError(f"dp must be >= 1, got {dp}")

    key_base = polymer.lower()
    if dp == 1:
        return [GLYCAM_RESIDUE_NAMES[(key_base, "terminal_nonreducing")]]

    seq: List[str] = []
    seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "terminal_nonreducing")])
    for _ in range(dp - 2):
        seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "internal")])
    seq.append(GLYCAM_RESIDUE_NAMES[(key_base, "terminal_reducing")])
    return seq


def load_glycam_params(
    polymer: str,
    dp: int,
    work_dir: Path | None = None,
) -> Dict[str, object]:
    """Prepare backbone GLYCAM metadata used by forcefield assembly steps."""
    sequence = build_glycam_sequence(polymer=polymer, dp=dp)
    return {
        "polymer": polymer,
        "dp": dp,
        "sequence": sequence,
        "work_dir": None if work_dir is None else str(Path(work_dir)),
    }


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
    """Export AMBER artifacts using residue-aware GLYCAM + GAFF assembly."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if dp is None:
        if mol.HasProp("_poly_csp_dp"):
            dp = int(mol.GetIntProp("_poly_csp_dp"))
        else:
            raise ValueError(
                "residue_aware backend requires dp (degree of polymerization). "
                "Pass dp= or ensure the molecule has _poly_csp_dp metadata."
            )

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
        selector_prmtop_path = build_selector_prmtop(
            mol2_path=sel_artifacts["mol2"],
            frcmod_path=sel_artifacts["frcmod"],
            work_dir=sel_dir,
        )

    linkage_frcmod = build_linkage_frcmod(out)
    linkage_frcmod_path = str(linkage_frcmod.resolve())

    pdb_path = out / f"{model_name}.pdb"
    write_pdb_from_rdkit(mol, pdb_path)

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


def build_tleap_script(
    polymer: str,
    dp: int,
    selector_lib_path: str | None = None,
    selector_frcmod_path: str | None = None,
    linkage_frcmod_path: str | None = None,
    model_name: str = "model",
    prmtop_name: str = "model.prmtop",
    inpcrd_name: str = "model.inpcrd",
    periodic: bool = False,
    box_vectors_A: tuple[float, float, float] | None = None,
) -> str:
    """Generate a tleap input script for residue-aware assembly.

    The script:
    1. Sources GLYCAM06j for backbone residues.
    2. Optionally loads GAFF2 selector library/frcmod.
    3. Assembles the polymer chain from residue codes.
    4. Optionally defines a periodic head-to-tail bond and box.
    5. Saves prmtop/inpcrd.
    """
    lines: List[str] = [
        "# poly_csp residue-aware GLYCAM06 + GAFF2 assembly",
        "source leaprc.GLYCAM_06j-1",
    ]

    if selector_lib_path or selector_frcmod_path:
        lines.append("source leaprc.gaff2")
    if selector_frcmod_path:
        lines.append(f"loadamberparams {selector_frcmod_path}")
    if selector_lib_path:
        lines.append(f"loadoff {selector_lib_path}")
    if linkage_frcmod_path:
        lines.append(f"loadamberparams {linkage_frcmod_path}")

    seq = build_glycam_sequence(polymer, dp)
    seq_str = " ".join(seq)
    lines.append(f"mol = sequence {{ {seq_str} }}")

    if periodic and dp > 1:
        # Create head-to-tail bond across the periodic boundary.
        # tleap uses 1-based residue indexing.
        lines.append(f"bond mol.1.C1 mol.{dp}.O4")

    if periodic and box_vectors_A is not None:
        Lx, Ly, Lz = box_vectors_A
        lines.append(f"setBox mol centers {{ {Lx:.4f} {Ly:.4f} {Lz:.4f} }}")

    lines.extend([
        f"saveamberparm mol {prmtop_name} {inpcrd_name}",
        "quit",
    ])
    return "\n".join(lines) + "\n"


def run_tleap_assembly(
    tleap_script: str,
    outdir: Path,
    model_name: str = "model",
) -> Dict[str, object]:
    """Execute tleap with the given script and return artifact paths.

    Parameters
    ----------
    tleap_script : str
        Contents of the tleap input file.
    outdir : Path
        Directory for all output files.
    model_name : str
        Base name for prmtop/inpcrd.

    Returns
    -------
    Dict with keys: prmtop, inpcrd, tleap_input, tleap_log, parameterized.
    """
    _ensure_required_tools(("tleap",))

    outdir.mkdir(parents=True, exist_ok=True)
    tleap_path = outdir / "tleap.in"
    tleap_log = outdir / "tleap.log"
    prmtop_path = outdir / f"{model_name}.prmtop"
    inpcrd_path = outdir / f"{model_name}.inpcrd"

    tleap_path.write_text(tleap_script, encoding="utf-8")
    _run_command(["tleap", "-f", tleap_path.name], cwd=outdir, log_path=tleap_log)

    missing = [
        str(p) for p in (prmtop_path, inpcrd_path)
        if not p.exists() or p.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "tleap assembly completed but expected outputs were not generated: "
            + ", ".join(missing) + f". See: {tleap_log}"
        )

    return {
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "tleap_input": str(tleap_path),
        "tleap_log": str(tleap_log),
        "parameterized": True,
        "parameter_backend": "residue_aware",
    }


def build_linkage_frcmod(outdir: Path, filename: str = "linkage.frcmod") -> Path:
    """Generate a supplementary frcmod with zero-energy torsion placeholders.

    GLYCAM06j does not parameterize torsions across the periodic 0GA–4GA
    boundary (e.g. H2-Cg-Cg-H2).  These zero-barrier terms let tleap
    complete cleanly; for production MD they should be replaced with
    QM-fitted values.
    """
    # Missing torsions reported by tleap for 1→4 periodic linkage.
    missing_torsions = [
        "H2-Cg-Cg-H2",
        "H2-Cg-Cg-H1",
        "H1-Cg-Cg-H1",
        "H2-Cg-Cg-Oh",
        "H1-Cg-Cg-Oh",
        "Oh-Cg-Cg-Oh",
        "H2-Cg-Cg-Os",
        "H1-Cg-Cg-Os",
        "Oh-Cg-Cg-Os",
        "Os-Cg-Cg-Os",
        "Cg-Cg-Cg-H2",
        "Cg-Cg-Cg-H1",
        "Cg-Cg-Cg-Oh",
        "Cg-Cg-Cg-Os",
        "Cg-Cg-Cg-Cg",
    ]
    lines = [
        "Supplementary frcmod for periodic glycosidic linkage (poly_csp)",
        "MASS",
        "",
        "BOND",
        "",
        "ANGLE",
        "",
        "DIHE",
    ]
    for torsion in missing_torsions:
        # 1 term, 0.0 barrier, 0.0 phase, periodicity 3.0
        lines.append(f"{torsion}   1    0.000         0.0             3.0")
    lines.extend(["", "IMPROPER", "", "NONBON", "", ""])

    outdir.mkdir(parents=True, exist_ok=True)
    frcmod_path = outdir / filename
    frcmod_path.write_text("\n".join(lines), encoding="utf-8")
    return frcmod_path


def parameterize_selector_fragment(
    selector_mol: Chem.Mol,
    charge_model: str = "bcc",
    net_charge: int = 0,
    work_dir: Path | None = None,
) -> Dict[str, str]:
    """Run Antechamber/Parmchk2 on a selector fragment via GAFF helpers."""
    from poly_csp.forcefield.gaff import parameterize_gaff_fragment

    return parameterize_gaff_fragment(
        fragment_mol=selector_mol,
        charge_model=charge_model,
        net_charge=net_charge,
        residue_name="SEL",
        pdb_name="selector.pdb",
        mol2_name="selector.mol2",
        frcmod_name="selector.frcmod",
        lib_name="selector.lib",
        work_dir=work_dir,
        ensure_tools_fn=_ensure_required_tools,
        run_command_fn=_run_command,
        write_pdb_fn=write_pdb_from_rdkit,
    )


def build_selector_prmtop(
    mol2_path: str | Path,
    frcmod_path: str | Path,
    work_dir: Path | None = None,
) -> str:
    """Create a standalone AMBER prmtop for the selector fragment."""
    from poly_csp.forcefield.gaff import build_fragment_prmtop

    return build_fragment_prmtop(
        mol2_path=mol2_path,
        frcmod_path=frcmod_path,
        prmtop_name="selector.prmtop",
        inpcrd_name="selector.inpcrd",
        clean_mol2_name="selector_clean.mol2",
        work_dir=work_dir,
        run_command_fn=_run_command,
    )
