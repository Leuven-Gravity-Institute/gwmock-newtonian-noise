"""Shared builders for the field-family tests.

Every numeric value here is UNANCHORED: an illustrative crustal medium, flat
spectra, an illustrative layer and a synthetic deployment chosen to exercise the
code (three-dimensional sensor placement, mixed measured quantities, a timing
offset), not a representative Einstein Telescope site.
"""

from __future__ import annotations

import numpy as np

from gwmock_newtonian_noise.detector.geometry import equilateral_triangle
from gwmock_newtonian_noise.detector.masses import shared_vertex_layout
from gwmock_newtonian_noise.families import (
    BodyWaveFamily,
    CavityModel,
    FamilyJointModel,
    LayerOverHalfSpace,
    LoveWaveFamily,
    PointSource,
    PointSourceFamily,
    RayleighFamily,
    uniform_sphere_density,
)
from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig
from gwmock_newtonian_noise.seismic.medium import ElasticHalfSpace
from gwmock_newtonian_noise.seismic.spectrum import uniform_directional_density
from gwmock_newtonian_noise.witness.array import WitnessArray
from gwmock_newtonian_noise.witness.sensor import SensorQuantity, WitnessSensor

#: UNANCHORED illustrative medium.
MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED illustrative layer over half-space (a soft, light layer over the medium above).
LAYER = LayerOverHalfSpace(
    layer_thickness=200.0,
    layer_s_wave_speed=1500.0,
    layer_density=2200.0,
    half_space_s_wave_speed=3200.0,
    half_space_density=2700.0,
)

#: Test-mass depth and arm length of the synthetic triangle (UNANCHORED).
TEST_MASS_DEPTH = 300.0
ARM_LENGTH = 10_000.0

#: Sampling of the realization tests.
SAMPLING_FREQUENCY = 16.0
SEGMENT_DURATION = 2.0


def flat(value: float):
    """Return a flat one-sided PSD callable."""
    return lambda frequencies: np.full(np.shape(frequencies), float(value))


def body(free_surface: bool, *, p: float = 1.0, sv: float = 0.5, sh: float = 0.7, density=uniform_sphere_density, **kw):
    """Return a small body-wave family with flat, distinct per-type PSDs."""
    return BodyWaveFamily(
        medium=kw.pop("medium", MEDIUM),
        p_psd=flat(p),
        sv_psd=flat(sv),
        sh_psd=flat(sh),
        directional_density=density,
        free_surface=free_surface,
        n_polar=kw.pop("n_polar", 4),
        n_azimuth=kw.pop("n_azimuth", 8),
    )


def love(psd: float = 1.0, **kw) -> LoveWaveFamily:
    """Return a small Love-wave family."""
    return LoveWaveFamily(
        medium=kw.pop("medium", LAYER),
        surface_displacement_psd=flat(psd),
        directional_density=kw.pop("directional_density", uniform_directional_density),
        n_azimuth=kw.pop("n_azimuth", 8),
    )


def rayleigh(psd: float = 1.0, n_azimuth: int = 8) -> RayleighFamily:
    """Return a small Rayleigh family on the shared medium."""
    field = RayleighWaveFieldConfig(
        medium=MEDIUM,
        surface_displacement_psd=flat(psd),
        directional_density=uniform_directional_density,
        n_azimuth=n_azimuth,
    )
    return RayleighFamily(field=field, n_depth=48)


def point(psd: float = 1.0) -> PointSourceFamily:
    """Return two independent point sources near the first vertex, off the sensor plane."""
    return PointSourceFamily(
        medium=MEDIUM,
        sources=(
            PointSource(position=(400.0, 250.0, 450.0), displacement_psd=flat(psd), reference_distance=100.0),
            PointSource(position=(-300.0, -500.0, 150.0), displacement_psd=flat(0.5 * psd), reference_distance=100.0),
        ),
    )


def layout():
    """Return the shared-vertex triangle layout with test masses at depth."""
    return shared_vertex_layout(equilateral_triangle(ARM_LENGTH), test_mass_depth=TEST_MASS_DEPTH)


def array(*, self_noise: bool = True) -> WitnessArray:
    """Return a three-dimensional witness array: surface, borehole and cavern sensors on all axes."""
    noise = (lambda frequency: 1e-3) if self_noise else None
    sensors = (
        WitnessSensor(name="S0z", position=(0.0, 0.0, 0.0), self_noise_psd=noise, timing_offset=0.03),
        WitnessSensor(name="B1x", position=(150.0, 80.0, 120.0), orientation=(1.0, 0.0, 0.0), self_noise_psd=noise),
        WitnessSensor(
            name="C2y",
            position=(-60.0, 40.0, 290.0),
            orientation=(0.0, 1.0, 0.0),
            quantity=SensorQuantity.VELOCITY,
            self_noise_psd=noise,
        ),
        WitnessSensor(name="C3z", position=(70.0, -90.0, 310.0), self_noise_psd=noise),
    )
    return WitnessArray(sensors=sensors)


def model(family, *, cavity: float | None = 10.0, self_noise: bool = True) -> FamilyJointModel:
    """Return a joint model of ``family`` on the shared layout and array."""
    return FamilyJointModel(
        family=family,
        layout=layout(),
        array=array(self_noise=self_noise),
        cavity=None if cavity is None else CavityModel(radius=cavity),
    )


def realization_grid() -> np.ndarray:
    """Return the ``rfftfreq`` grid of the realization tests."""
    n_samples = round(SEGMENT_DURATION * SAMPLING_FREQUENCY)
    return np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_FREQUENCY)


def assert_covariance_close(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    """Assert two ``(n_freq, n, n)`` covariances agree entry-wise in units of ``sqrt(C_aa C_bb)``.

    Channels span many orders of magnitude (strain ~1e-12, displacement ~1), so a
    plain relative tolerance fails on entries that are numerically zero by
    cancellation; normalizing by the expected auto-spectra compares every entry on
    the scale that matters for a coherence.
    """
    diagonal = np.sqrt(np.abs(np.einsum("fii->fi", expected).real))
    scale = diagonal[:, :, None] * diagonal[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(scale > 0.0, np.abs(actual - expected) / scale, np.abs(actual - expected))
    assert np.max(normalized) <= tolerance, float(np.max(normalized))
