from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run_build(overrides: str) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        "polycsp",
        "bash",
        "-lc",
        (
            "cd /home/lukas/work/projects/poly_csp && "
            "PYTHONPATH=src python -m poly_csp.pipelines.build_csp "
            + overrides
        ),
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)


def test_pipeline_capped_mode_runs_with_explicit_caps(tmp_path: Path) -> None:
    outdir = tmp_path / "capped_out"
    _run_build(
        "polymer.dp=3 selector.enabled=false polymer.end_mode=capped "
        "+polymer.end_caps.left=acetyl +polymer.end_caps.right=methoxy "
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
        f"output.dir={outdir}"
    )

    report = json.loads((outdir / "build_report.json").read_text(encoding="utf-8"))
    assert report["end_mode"] == "periodic"
    assert report["monomer_representation"] == "natural_oh"
    assert report["qc_pass"] is True
    assert (outdir / "model.pdb").exists()
