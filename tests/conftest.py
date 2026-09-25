"""Configuration and fixtures for pytest."""

from __future__ import annotations

import pytest

from gwmock_newtonian.config import NewtonianScaffoldConfig


@pytest.fixture
def detector_names() -> list[str]:
    """Strain channel names shared across tests."""
    return ["ET1", "ET2"]


@pytest.fixture
def witness_names() -> list[str]:
    """Witness channel names shared across tests."""
    return ["SEIS1"]


@pytest.fixture
def base_config(detector_names: list[str], witness_names: list[str]) -> NewtonianScaffoldConfig:
    """A minimal, valid scaffold configuration."""
    return NewtonianScaffoldConfig(
        detectors=detector_names,
        witnesses=witness_names,
        sampling_frequency=64.0,
        duration=2.0,
        seed=1234,
        coupling=0.3,
    )
