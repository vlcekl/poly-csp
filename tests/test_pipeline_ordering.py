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


def test_pipeline_ordering_enabled_writes_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "ordered_out"
    overrides = (
        "polymer.dp=2 "
        "selector.enabled=true selector.sites=[C6] "
        "ordering.enabled=true ordering.max_candidates=8 "
        "relax.enabled=false amber.enabled=false "
        f"output.dir={outdir}"
    )
    cmd = [sys.executable, "-m", "poly_csp.pipelines.build_csp", *shlex.split(overrides)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    subprocess.run(
        cmd, check=True, text=True, capture_output=True, cwd=_ROOT, env=env
    )

    report_path = outdir / "build_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["ordering_enabled"] is True
    assert isinstance(data["ordering_summary"], dict)
    assert "final_hbond_geometric_fraction" in data["ordering_summary"]
