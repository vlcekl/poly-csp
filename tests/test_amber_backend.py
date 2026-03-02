from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from poly_csp.chemistry.monomers import make_glucose_template
from poly_csp.io.amber import export_amber_artifacts


def _template_mol():
    return make_glucose_template("amylose").mol


def test_export_placeholder_backend_writes_scaffold(tmp_path: Path) -> None:
    outdir = tmp_path / "amber_placeholder"
    summary = export_amber_artifacts(
        mol=_template_mol(),
        outdir=outdir,
        model_name="unit",
        charge_model="bcc",
        parameter_backend="placeholder",
    )
    assert summary["parameterized"] is False
    assert (outdir / "unit.pdb").exists()
    assert (outdir / "unit.prmtop").exists()
    assert (outdir / "unit.inpcrd").exists()
    assert (outdir / "amber_export.json").exists()


def test_export_ambertools_backend_missing_tools_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import poly_csp.io.amber as amber_mod

    monkeypatch.setattr(amber_mod.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="requires executables"):
        export_amber_artifacts(
            mol=_template_mol(),
            outdir=tmp_path / "amber_missing",
            parameter_backend="ambertools",
        )


def test_export_ambertools_backend_command_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import poly_csp.io.amber as amber_mod

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(
        cmd,
        cwd: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        work = Path(cwd)
        exe = cmd[0]
        if exe == "antechamber":
            out_name = cmd[cmd.index("-o") + 1]
            (work / out_name).write_text("@<TRIPOS>MOLECULE\nUNIT\n", encoding="utf-8")
        elif exe == "parmchk2":
            out_name = cmd[cmd.index("-o") + 1]
            (work / out_name).write_text("MASS\n", encoding="utf-8")
        elif exe == "tleap":
            (work / "model.lib").write_text("!entry.MOL.unit.atoms\n", encoding="utf-8")
            (work / "model.prmtop").write_text("%VERSION\n", encoding="utf-8")
            (work / "model.inpcrd").write_text("Mock inpcrd\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(amber_mod.shutil, "which", fake_which)
    monkeypatch.setattr(amber_mod.subprocess, "run", fake_run)

    outdir = tmp_path / "ambertools_mock"
    summary = export_amber_artifacts(
        mol=_template_mol(),
        outdir=outdir,
        parameter_backend="ambertools",
        charge_model="bcc",
    )

    assert summary["parameterized"] is True
    assert summary["parameter_backend"] == "ambertools"
    assert summary["net_charge"] == 0
    assert (outdir / "model.mol2").exists()
    assert (outdir / "model.frcmod").exists()
    assert (outdir / "model.lib").exists()
    assert (outdir / "model.prmtop").exists()
    assert (outdir / "model.inpcrd").exists()
    assert (outdir / "antechamber.log").exists()
    assert (outdir / "parmchk2.log").exists()
    assert (outdir / "tleap.log").exists()
