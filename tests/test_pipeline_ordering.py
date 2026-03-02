from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_pipeline_ordering_enabled_writes_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "ordered_out"
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
            "polymer.dp=2 "
            "selector.enabled=true selector.sites=[C6] "
            "ordering.enabled=true ordering.max_candidates=8 "
            f"output.dir={outdir}"
        ),
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)

    report_path = outdir / "build_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["ordering_enabled"] is True
    assert isinstance(data["ordering_summary"], dict)
    assert "final_hbond_fraction" in data["ordering_summary"]
