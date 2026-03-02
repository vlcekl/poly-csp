from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run_build(overrides: str) -> subprocess.CompletedProcess[str]:
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
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def test_pipeline_relax_enabled_writes_relax_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "relax_out"
    _run_build(
        "polymer.dp=2 "
        "selector.enabled=false "
        "relax.enabled=true relax.n_stages=1 relax.max_iterations=25 "
        "relax.anneal.enabled=false "
        f"output.dir={outdir}"
    )

    report_path = outdir / "build_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["relax_enabled"] is True
    assert isinstance(data["relax_summary"], dict)
    assert data["relax_summary"]["enabled"] is True
    assert len(data["relax_summary"]["stage_energies_kj_mol"]) >= 1


def test_pipeline_amber_export_scaffold_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "amber_out"
    _run_build(
        "polymer.dp=2 "
        "selector.enabled=false "
        "output.export_formats=[pdb,amber] "
        "amber.enabled=true amber.dir=amber_artifacts "
        f"output.dir={outdir}"
    )

    report_path = outdir / "build_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["amber_enabled"] is True
    summary = data["amber_summary"]
    assert summary["enabled"] is True
    assert summary["parameterized"] is False

    amber_dir = outdir / "amber_artifacts"
    assert (amber_dir / "model.pdb").exists()
    assert (amber_dir / "model.prmtop").exists()
    assert (amber_dir / "model.inpcrd").exists()
    assert (amber_dir / "tleap.in").exists()
    assert (amber_dir / "amber_export.json").exists()
