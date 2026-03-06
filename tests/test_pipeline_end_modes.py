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


def test_pipeline_capped_mode_runs_with_explicit_caps(tmp_path: Path) -> None:
    outdir = tmp_path / "capped_out"
    _run_build(
        "polymer.dp=3 selector.enabled=false polymer.end_mode=capped "
        "+polymer.end_caps.left=acetyl +polymer.end_caps.right=methoxy "
        "relax.enabled=false amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "capped"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()


def test_pipeline_periodic_natural_oh_runs(tmp_path: Path) -> None:
    outdir = tmp_path / "periodic_out"
    _run_build(
        "polymer.dp=3 selector.enabled=false "
        "polymer.monomer_representation=natural_oh polymer.end_mode=periodic "
        "relax.enabled=false amber.enabled=false "
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "periodic"
    assert report["monomer_representation"] == "natural_oh"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()
