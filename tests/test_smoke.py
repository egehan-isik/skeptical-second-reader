"""Smoke test: the package installs and imports.

Fails first if pyproject.toml or the src/ layout breaks.
"""

import skeptical_second_reader


def test_package_imports():
    assert skeptical_second_reader.__name__ == "skeptical_second_reader"
