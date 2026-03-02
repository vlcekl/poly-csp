from __future__ import annotations


def test_package_imports() -> None:
    import poly_csp  # noqa: F401
    import poly_csp.chemistry  # noqa: F401
    import poly_csp.config  # noqa: F401
    import poly_csp.geometry  # noqa: F401
    import poly_csp.io  # noqa: F401
    import poly_csp.mm  # noqa: F401
    import poly_csp.ordering  # noqa: F401
    import poly_csp.pipelines  # noqa: F401
