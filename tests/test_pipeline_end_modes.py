from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def _run_build(overrides: str) -> None:
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )


def _count_pdb_atoms(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(
        1
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )


def test_pipeline_capped_mode_runs_with_explicit_caps(tmp_path: Path) -> None:
    outdir = tmp_path / "capped_out"
    _run_build(
        "topology.backbone.dp=3 topology.selector.enabled=false "
        "topology.backbone.end_mode=capped "
        "+topology.backbone.end_caps.left=acetyl "
        "+topology.backbone.end_caps.right=methoxy "
        "forcefield.options.enabled=false amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "capped"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()


def test_pipeline_periodic_natural_oh_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_out"
    _run_build(
        "topology.backbone.dp=3 topology.selector.enabled=false "
        "topology.backbone.monomer_representation=natural_oh "
        "topology.backbone.end_mode=periodic "
        "forcefield.options.enabled=false amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "periodic"
    assert report["monomer_representation"] == "natural_oh"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()


def test_pipeline_writes_all_atom_model_by_default_and_optional_heavy_debug(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "all_atom_out"
    _run_build(
        "topology.backbone.dp=1 topology.selector.enabled=false "
        "forcefield.options.enabled=false amber.enabled=false "
        "output.write_heavy_debug=true "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["final_structure_variant"] == "all_atom"
    assert report["heavy_debug_written"] is True

    model_pdb = outdir / "model.pdb"
    heavy_pdb = outdir / "model_heavy.pdb"
    assert model_pdb.exists()
    assert heavy_pdb.exists()
    assert _count_pdb_atoms(model_pdb) > _count_pdb_atoms(heavy_pdb)
