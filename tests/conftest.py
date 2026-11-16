"""Configuration and fixtures for pytest."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian_noise.config import NewtonianScaffoldConfig


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


# --- Physical joint backend fixtures -------------------------------------------------
#
# Every numeric value below is UNANCHORED: an illustrative crustal medium, flat
# displacement PSD and witness deployment chosen to exercise the code (mixed
# measured quantities, a non-SI output unit, a timing offset that makes the
# strain--witness cross-spectra complex), not a representative ET site.

#: Sampling frequency and segment length of the physical-backend fixtures.
JOINT_SAMPLING_FREQUENCY = 16.0
JOINT_SEGMENT_DURATION = 2.0


def _joint_detector_config():
    """Return the detector-response fixture wrapping an isotropic flat-PSD field."""
    from gwmock_newtonian_noise.detector.geometry import equilateral_triangle
    from gwmock_newtonian_noise.detector.masses import shared_vertex_layout
    from gwmock_newtonian_noise.detector.response import DetectorResponseConfig
    from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig
    from gwmock_newtonian_noise.seismic.medium import ElasticHalfSpace
    from gwmock_newtonian_noise.seismic.spectrum import power_law_psd, uniform_directional_density

    field = RayleighWaveFieldConfig(
        medium=ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0),
        surface_displacement_psd=power_law_psd(reference_frequency=1.0, reference_psd=1.0, exponent=0.0),
        directional_density=uniform_directional_density,
        n_azimuth=16,
    )
    layout = shared_vertex_layout(equilateral_triangle(10_000.0), test_mass_depth=20.0)
    return DetectorResponseConfig(field=field, layout=layout, n_depth=48)


def make_joint_array(**overrides):
    """Return the four-sensor witness fixture; ``overrides`` replace fields of sensor ``W2``."""
    from dataclasses import replace

    from gwmock_newtonian_noise.witness.array import WitnessArray
    from gwmock_newtonian_noise.witness.sensor import SensorQuantity, WitnessSensor

    def noise(quantity: SensorQuantity):
        # Comparable to each quantity's field signal (displacement PSD ~1 m^2/Hz,
        # times omega^2 per time derivative), so dropping it is visible in every auto-spectrum.
        order = {SensorQuantity.DISPLACEMENT: 0, SensorQuantity.VELOCITY: 1, SensorQuantity.ACCELERATION: 2}[quantity]
        return lambda frequency: 0.3 * (2.0 * np.pi * frequency) ** (2 * order)

    sensors = [
        WitnessSensor(
            name="W0", position=(0.0, 0.0, 0.0), self_noise_psd=noise(SensorQuantity.DISPLACEMENT), timing_offset=0.03
        ),
        WitnessSensor(
            name="W1",
            position=(300.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0),
            quantity=SensorQuantity.VELOCITY,
            self_noise_psd=noise(SensorQuantity.VELOCITY),
        ),
        WitnessSensor(
            name="W2",
            position=(-150.0, 260.0, 0.0),
            orientation=(0.0, 1.0, 0.0),
            quantity=SensorQuantity.ACCELERATION,
            self_noise_psd=noise(SensorQuantity.ACCELERATION),
        ),
        WitnessSensor(
            name="W3",
            position=(-150.0, -260.0, 0.0),
            self_noise_psd=noise(SensorQuantity.DISPLACEMENT),
            timing_offset=-0.02,
        ),
    ]
    if overrides:
        sensors[2] = replace(sensors[2], **overrides)
    return WitnessArray(sensors=tuple(sensors))


@pytest.fixture(scope="session")
def joint_detector_config():
    """The detector-response configuration of the physical-backend fixtures."""
    return _joint_detector_config()


@pytest.fixture(scope="session")
def joint_array():
    """The witness array of the physical-backend fixtures."""
    return make_joint_array()


@pytest.fixture(scope="session")
def joint_prototype(joint_detector_config, joint_array):
    """One constructed physical backend; tests receive independent deep copies of it."""
    from gwmock_newtonian_noise.joint import NewtonianJointSimulator

    return NewtonianJointSimulator(
        joint_detector_config,
        joint_array,
        sampling_frequency=JOINT_SAMPLING_FREQUENCY,
        segment_duration=JOINT_SEGMENT_DURATION,
        seed=2026,
    )


@pytest.fixture
def joint_simulator(joint_prototype):
    """A fresh, unstarted physical backend (seed 2026) on the fixture configuration."""
    import copy

    return copy.deepcopy(joint_prototype)


@pytest.fixture(scope="session")
def joint_array_factory():
    """Return :func:`make_joint_array`, to build variants of the witness fixture."""
    return make_joint_array
