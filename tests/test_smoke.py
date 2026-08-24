"""Smoke test for mini_agent package import."""

import mini_agent


def test_import_mini_agent() -> None:
    assert mini_agent.__version__ == "0.3.0"
