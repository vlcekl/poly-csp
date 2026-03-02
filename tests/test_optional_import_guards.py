from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _fresh_import_build_csp(monkeypatch: pytest.MonkeyPatch, injected_exc: Exception):
    module_name = "poly_csp.pipelines.build_csp"
    optional_name = "poly_csp.mm.minimize"
    sys.modules.pop(module_name, None)
    sys.modules.pop(optional_name, None)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == optional_name:
            raise injected_exc
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module(module_name)


def test_optional_import_missing_dependency_is_graceful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _fresh_import_build_csp(monkeypatch, ImportError("missing openmm"))
    assert mod.RelaxSpec is None
    assert mod.run_staged_relaxation is None


def test_optional_import_unexpected_error_is_re_raised_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        RuntimeError, match="Unexpected error while importing OpenMM modules."
    ):
        _fresh_import_build_csp(monkeypatch, RuntimeError("boom"))
