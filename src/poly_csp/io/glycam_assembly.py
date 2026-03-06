# poly_csp/io/glycam_assembly.py
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
    """Run antechamber + parmchk2 on a single selector fragment.

    Returns dict with paths to mol2, frcmod, and lib files.
    """
    _ensure_required_tools(("antechamber", "parmchk2"))

    import tempfile
    if work_dir is None:
        wd = Path(tempfile.mkdtemp(prefix="polycsp_sel_"))
    else:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)

    pdb_path = wd / "selector.pdb"
    mol2_path = wd / "selector.mol2"
    frcmod_path = wd / "selector.frcmod"
    lib_path = wd / "selector.lib"

    # Replace dummy atoms ([*], atomic number 0) with hydrogen so that
    # antechamber can perform atom typing and charge derivation.
    clean_mol = Chem.RWMol(selector_mol)
    for atom in clean_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(clean_mol)

    write_pdb_from_rdkit(clean_mol, pdb_path)

    # antechamber
    _run_command(
        [
            "antechamber",
            "-i", pdb_path.name, "-fi", "pdb",
            "-o", mol2_path.name, "-fo", "mol2",
            "-at", "gaff2", "-c", charge_model,
            "-nc", str(net_charge),
            "-rn", "SEL", "-dr", "no", "-pf", "y", "-s", "2",
        ],
        cwd=wd,
        log_path=wd / "antechamber.log",
    )

    # parmchk2
    _run_command(
        [
            "parmchk2",
            "-i", mol2_path.name, "-f", "mol2",
            "-s", "gaff2", "-o", frcmod_path.name,
        ],
        cwd=wd,
        log_path=wd / "parmchk2.log",
    )

    # saveoff via tleap
    saveoff_script = "\n".join([
        "source leaprc.gaff2",
        f"loadamberparams {frcmod_path.name}",
        f"sel = loadmol2 {mol2_path.name}",
        f"saveoff sel {lib_path.name}",
        "quit",
    ]) + "\n"
    saveoff_in = wd / "saveoff.in"
    saveoff_in.write_text(saveoff_script, encoding="utf-8")
    _run_command(
        ["tleap", "-f", saveoff_in.name],
        cwd=wd,
        log_path=wd / "saveoff.log",
    )

    return {
        "mol2": str(mol2_path.resolve()),
        "frcmod": str(frcmod_path.resolve()),
        "lib": str(lib_path.resolve()),
    }


def build_selector_prmtop(
    mol2_path: str | Path,
    frcmod_path: str | Path,
    work_dir: Path | None = None,
) -> str:
    """Create a standalone AMBER prmtop for the selector fragment.

    Uses tleap to load the GAFF2 mol2 and frcmod produced by
    ``parameterize_selector_fragment`` and save a prmtop/inpcrd pair.
    The resulting prmtop can be loaded by OpenMM to extract bonded forces.

    The mol2 file produced by antechamber sometimes contains duplicate bonds
    (especially for aromatic systems).  This function deduplicates them
    before feeding the file to tleap.

    Parameters
    ----------
    mol2_path
        Path to the selector mol2 file (GAFF2 atom types + charges).
    frcmod_path
        Path to the selector frcmod file (additional parameters).
    work_dir
        Working directory for tleap output.  Defaults to the parent
        directory of the mol2 file.

    Returns
    -------
    Absolute path to the generated prmtop file.
    """
    mol2_path = Path(mol2_path)
    frcmod_path = Path(frcmod_path)

    if work_dir is None:
        work_dir = mol2_path.parent
    else:
        work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_mol2 = work_dir / "selector_clean.mol2"
    _deduplicate_mol2_bonds(mol2_path, clean_mol2)

    prmtop_path = work_dir / "selector.prmtop"
    inpcrd_path = work_dir / "selector.inpcrd"

    script = "\n".join([
        "source leaprc.gaff2",
        f"loadamberparams {frcmod_path.resolve()}",
        f"sel = loadmol2 {clean_mol2.resolve()}",
        f"saveamberparm sel {prmtop_path.name} {inpcrd_path.name}",
        "quit",
    ]) + "\n"

    script_path = work_dir / "build_prmtop.in"
    script_path.write_text(script, encoding="utf-8")

    _run_command(
        ["tleap", "-f", script_path.name],
        cwd=work_dir,
        log_path=work_dir / "build_prmtop.log",
    )

    if not prmtop_path.exists() or prmtop_path.stat().st_size == 0:
        raise RuntimeError(
            f"tleap failed to generate selector prmtop. "
            f"See: {work_dir / 'build_prmtop.log'}"
        )

    return str(prmtop_path.resolve())


def _deduplicate_mol2_bonds(src: Path, dst: Path) -> None:
    """Remove duplicate bonds from an AMBER mol2 file.

    Antechamber sometimes writes duplicate bond records for aromatic systems.
    This reads the mol2, deduplicates bond pairs, renumbers, and writes clean output.
    """
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines: list[str] = []
    in_bond_section = False
    past_bond_section = False
    bond_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "@<TRIPOS>BOND":
            in_bond_section = True
            out_lines.append(line)
            continue
        if in_bond_section and stripped.startswith("@<TRIPOS>"):
            # End of bond section — process and flush.
            in_bond_section = False
            past_bond_section = True
            _flush_dedup_bonds(bond_lines, out_lines)
            out_lines.append(line)
            continue
        if in_bond_section:
            bond_lines.append(line)
        else:
            out_lines.append(line)

    # If the file ends right after the bond section (no further sections).
    if in_bond_section and not past_bond_section:
        _flush_dedup_bonds(bond_lines, out_lines)

    # Fix the atom/bond counts in the MOLECULE header.
    _fix_mol2_bond_count(out_lines, bond_lines)

    dst.write_text("".join(out_lines), encoding="utf-8")


def _flush_dedup_bonds(bond_lines: list[str], out_lines: list[str]) -> None:
    """Deduplicate bond lines and append to out_lines with renumbered IDs."""
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int, str]] = []
    for raw_line in bond_lines:
        parts = raw_line.split()
        if len(parts) < 4:
            continue
        a, b = int(parts[1]), int(parts[2])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b, parts[3]))

    for idx, (a, b, btype) in enumerate(deduped, 1):
        out_lines.append(f"     {idx:>2d}    {a:>2d}    {b:>2d} {btype}\n")

    # Store deduped count for header fixup.
    bond_lines.clear()
    bond_lines.extend([f"{len(deduped)}"])  # stash the count


def _fix_mol2_bond_count(lines: list[str], count_stash: list[str]) -> None:
    """Fix the bond count in the @<TRIPOS>MOLECULE header."""
    if not count_stash:
        return
    try:
        new_count = int(count_stash[0])
    except (ValueError, IndexError):
        return

    # The counts line is the 3rd line after @<TRIPOS>MOLECULE.
    for i, line in enumerate(lines):
        if line.strip() == "@<TRIPOS>MOLECULE":
            # Line i+2 is the counts line: "  natom  nbond  ..."
            if i + 2 < len(lines):
                parts = lines[i + 2].split()
                if len(parts) >= 2:
                    parts[1] = str(new_count)
                    lines[i + 2] = "   " + "    ".join(parts) + "\n"
            break
