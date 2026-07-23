from __future__ import annotations

import os

import pytest
from jax import config


_X64_VALUES = {"0": False, "1": True}
_x64_setting = os.environ.get("GEOJAX_TEST_X64", "1")
if _x64_setting not in _X64_VALUES:
    raise RuntimeError("GEOJAX_TEST_X64 must be '0' or '1'.")

config.update("jax_enable_x64", _X64_VALUES[_x64_setting])


def pytest_report_header() -> str:
    precision = "float64" if config.jax_enable_x64 else "float32"
    return f"GeoJAX numerical precision: {precision}"


@pytest.fixture
def dtype_atol() -> float:
    """Extra absolute tolerance required only by float32 linear algebra."""
    return 0.0 if config.jax_enable_x64 else 2e-6
