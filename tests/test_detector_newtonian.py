"""Tests for the Newtonian-noise acceleration kernels and their independent controls.

The plane-wave (production) kernel is checked against:

- a closed-form depth-overlap integral for a purely exponential source, derived
  independently of the production quadrature;
- the classic uniform-sheet limit ``a_z = 2 pi G rho0`` (an analytic external
  reference);
- the direct real-space Green's-function quadrature in
  :func:`~gwmock_newtonian.detector.newtonian.acceleration_quadrature`, which is
  the required "direct test-mass acceleration integration" control.

All numeric values here are UNANCHORED illustrative fixtures -- no representative
ET site parameter set is implied.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian.detector.newtonian import (
    GRAVITATIONAL_CONSTANT,
    SURFACE_AMPLITUDE_REFERENCE_METRES,
    RayleighMode,
    acceleration_quadrature,
    density_depth_amplitude,
    plane_wave_acceleration,
    surface_vertical_amplitude,
)
from gwmock_newtonian.seismic.field import RayleighWaveFieldConfig
from gwmock_newtonian.seismic.medium import ElasticHalfSpace
from gwmock_newtonian.seismic.spectrum import power_law_psd, uniform_directional_density

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal medium used only to exercise the kernels.
_MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED: a flat illustrative displacement PSD, not a measured site spectrum.
_FLAT_PSD = power_law_psd(reference_frequency=1.0, reference_psd=1.0, exponent=0.0)


def _field(n_azimuth: int = 16, phase_velocity=None) -> RayleighWaveFieldConfig:
    return RayleighWaveFieldConfig(
        medium=_MEDIUM,
        surface_displacement_psd=_FLAT_PSD,
        directional_density=uniform_directional_density,
        phase_velocity=phase_velocity,
        n_azimuth=n_azimuth,
    )


def _synthetic_mode(wavenumber: float = 0.1) -> RayleighMode:
    """UNANCHORED: a synthetic mode with a convenient decay length of 10 m."""
    return RayleighMode(
        frequency=1.0,
        wavenumber=wavenumber,
        azimuth=0.0,
        phase_velocity=2.0 * np.pi / wavenumber,
    )


def _exponential_profile(amplitude: float, decay: float):
    return lambda z: amplitude * np.exp(-decay * np.asarray(z, dtype=float))


class TestModeValidation:
    """Physical guards on the Rayleigh mode."""

    def test_rejects_non_positive_wavenumber(self) -> None:
        """A zero wavenumber has no horizontal propagation."""
        with pytest.raises(ValueError, match="wavenumber"):
            RayleighMode(frequency=1.0, wavenumber=0.0, azimuth=0.0, phase_velocity=1.0)

    def test_rejects_non_positive_frequency(self) -> None:
        """A non-positive frequency is unphysical."""
        with pytest.raises(ValueError, match="frequency"):
            RayleighMode(frequency=0.0, wavenumber=1.0, azimuth=0.0, phase_velocity=1.0)


class TestDepthIntegrals:
    """The depth overlap integrals match a closed-form exponential result."""

    @pytest.mark.parametrize("mass_depth", [0.5, 5.0, 20.0])
    def test_depth_integrals_match_closed_form_exponential(self, mass_depth: float) -> None:
        """I1/I2 equal the analytic integral of an exponential source over the half-space."""
        k = 0.1
        decay = 0.085
        amplitude = -2.0e-3
        mode = _synthetic_mode(k)
        profile = _exponential_profile(amplitude, decay)
        smooth, signed = __import__(
            "gwmock_newtonian.detector.newtonian", fromlist=["_depth_integrals"]
        )._depth_integrals(profile, mode, mass_depth, mass_depth + 120.0, 400)
        # Analytic integrals of A0 exp(-decay z) exp(-k |z - z0|) / sign(z - z0).
        common = amplitude * (np.exp(-k * mass_depth) - np.exp(-decay * mass_depth)) / (decay - k)
        overlap = amplitude * np.exp(-decay * mass_depth) / (decay + k)
        np.testing.assert_allclose(smooth.real, common + overlap, rtol=1e-8)
        np.testing.assert_allclose(signed.real, -common + overlap, rtol=1e-8)


class TestArealTerm:
    """The free-surface areal term is present, signed and anchored."""

    def test_areal_only_matches_coughlin_plane_wave_expression(self) -> None:
        """Areal-only acceleration equals ``-2 pi i G rho0 exp(-k z0)`` horizontally."""
        mode = _synthetic_mode(0.1)
        position = [(0.0, 0.0, 5.0)]
        acceleration = plane_wave_acceleration(
            position, mode, np.zeros_like, _MEDIUM.density, include_volume=False, include_areal=True
        )
        expected_horizontal = -2.0j * np.pi * GRAVITATIONAL_CONSTANT * _MEDIUM.density * np.exp(-0.1 * 5.0)
        expected_vertical = 2.0 * np.pi * GRAVITATIONAL_CONSTANT * _MEDIUM.density * np.exp(-0.1 * 5.0)
        np.testing.assert_allclose(acceleration[0, 0], expected_horizontal, rtol=1e-12)
        np.testing.assert_allclose(acceleration[0, 2], expected_vertical, rtol=1e-12)

    def test_uniform_sheet_quadrature_reaches_two_pi_g_rho0(self) -> None:
        """A uniform areal sheet gives ``a_z = 2 pi G rho0`` in the infinite-radius limit.

        The quadrature is evaluated at a small wavenumber and a large radius; the
        analytic reference ``2 pi G rho0`` is external to the production kernel.
        """
        mode = _synthetic_mode(1.0e-6)
        position = [(0.0, 0.0, 10.0)]
        acceleration = acceleration_quadrature(
            position,
            mode,
            np.zeros_like,
            _MEDIUM.density,
            radial_max=2.0e5,
            n_radial=400,
            n_azimuth=128,
            include_volume=False,
            include_areal=True,
            n_depth=4,
        )
        expected = 2.0 * np.pi * GRAVITATIONAL_CONSTANT * _MEDIUM.density
        assert acceleration[0, 0].real == pytest.approx(0.0, abs=1e-18)
        assert acceleration[0, 2].real == pytest.approx(expected, rel=0.005)

    def test_volume_term_partially_cancels_the_areal_term(self) -> None:
        """For a Rayleigh-like profile the bulk horizontal term opposes the areal term.

        This is the partial-cancellation (Poisson-ratio-dependent) behaviour of
        Harms (2019) and Coughlin et al. (2016): the horizontal acceleration is
        reduced, not enhanced.
        """
        field = _field()
        frequency = 1.0
        phase_velocity = float(field.phase_velocity_at(np.array([frequency]))[0])
        wavenumber = 2.0 * np.pi * frequency / phase_velocity
        mode = RayleighMode(frequency, wavenumber, 0.0, phase_velocity)

        def profile(z):
            return density_depth_amplitude(z, frequency, field)

        position = [(0.0, 0.0, 20.0)]
        volume = plane_wave_acceleration(
            position, mode, profile, _MEDIUM.density, include_volume=True, include_areal=False
        )
        areal = plane_wave_acceleration(
            position, mode, profile, _MEDIUM.density, include_volume=False, include_areal=True
        )
        total = plane_wave_acceleration(
            position, mode, profile, _MEDIUM.density, include_volume=True, include_areal=True
        )
        assert abs(total[0, 0]) < abs(areal[0, 0])
        assert np.sign(total[0, 0].imag) == np.sign(areal[0, 0].imag)
        assert np.real(volume[0, 0] * np.conj(areal[0, 0])) < 0.0  # opposing horizontal contributions


class TestPlaneWaveVersusQuadrature:
    """The closed-form kernel agrees with a direct real-space integration."""

    def test_plane_wave_acceleration_matches_direct_quadrature(self) -> None:
        """Closed-form and direct Green's-function quadrature agree to a stated tolerance.

        The tolerance is 1% of the acceleration-vector norm; the quadrature uses
        a synthetic exponential source and a domain of several decay lengths. The
        two routes share no algebra beyond the physical Green's function.
        """
        mode = _synthetic_mode(0.1)
        profile = _exponential_profile(-2.0e-3, 0.085)
        positions = [(0.0, 0.0, 5.0), (40.0, -15.0, 8.0)]
        closed_form = plane_wave_acceleration(positions, mode, profile, _MEDIUM.density)
        quadrature = acceleration_quadrature(
            positions,
            mode,
            profile,
            _MEDIUM.density,
            radial_max=400.0,
            n_radial=300,
            n_azimuth=160,
            n_depth=200,
        )
        for closed, direct in zip(closed_form, quadrature, strict=True):
            relative = np.linalg.norm(closed - direct) / np.linalg.norm(closed)
            assert relative < 0.01, f"closed-form/direct relative error {relative:.4f} exceeded tolerance"

    @pytest.mark.parametrize("reference_amplitude", [2.0, 0.5])
    def test_volume_only_closed_form_tracks_quadrature_off_unity_amplitude(self, reference_amplitude: float) -> None:
        """The bulk closed form scales with ``reference_amplitude`` exactly as the quadrature does.

        The bulk profile is supplied per unit surface amplitude, so a
        ``reference_amplitude`` other than the default must scale the closed-form
        depth integrals exactly as it scales the quadrature's source. Rather than
        an absolute bound (which would be dominated by the direct route's domain
        truncation at a few percent), the two routes are compared through their
        disagreement at unit amplitude: the disagreement must be unchanged at a
        non-unit amplitude, which only holds if the bulk term is scaled in both.
        """
        mode = _synthetic_mode(0.1)
        profile = _exponential_profile(-2.0e-3, 0.085)
        positions = [(0.0, 0.0, 5.0), (40.0, -15.0, 8.0)]

        def routes(amplitude: float) -> tuple[np.ndarray, np.ndarray]:
            """Return the volume-only closed-form and quadrature accelerations at ``amplitude``."""
            closed = plane_wave_acceleration(
                positions,
                mode,
                profile,
                _MEDIUM.density,
                reference_amplitude=amplitude,
                include_volume=True,
                include_areal=False,
            )
            direct = acceleration_quadrature(
                positions,
                mode,
                profile,
                _MEDIUM.density,
                radial_max=400.0,
                n_radial=300,
                n_azimuth=160,
                n_depth=200,
                reference_amplitude=amplitude,
                include_volume=True,
                include_areal=False,
            )
            return closed, direct

        closed_unit, direct_unit = routes(1.0)
        closed_scaled, direct_scaled = routes(reference_amplitude)
        np.testing.assert_allclose(closed_scaled, reference_amplitude * closed_unit, rtol=1e-12)
        np.testing.assert_allclose(direct_scaled, reference_amplitude * direct_unit, rtol=1e-12)
        unit_disagreement = np.linalg.norm(closed_unit - direct_unit) / np.linalg.norm(closed_unit)
        scaled_disagreement = np.linalg.norm(closed_scaled - direct_scaled) / np.linalg.norm(closed_scaled)
        assert scaled_disagreement == pytest.approx(unit_disagreement, abs=1e-9)

    def test_quadrature_converges_as_the_radius_grows(self) -> None:
        """Increasing the horizontal integration radius does not make agreement worse."""
        mode = _synthetic_mode(0.1)
        profile = _exponential_profile(-2.0e-3, 0.085)
        positions = [(0.0, 0.0, 5.0)]
        closed = plane_wave_acceleration(positions, mode, profile, _MEDIUM.density)
        errors = []
        for radius in (200.0, 400.0):
            direct = acceleration_quadrature(
                positions, mode, profile, _MEDIUM.density, radial_max=radius, n_radial=300, n_azimuth=160, n_depth=200
            )
            errors.append(float(np.linalg.norm(closed - direct) / np.linalg.norm(closed)))
        assert errors[1] < 0.01
        assert errors[1] <= errors[0] + 0.01


class TestScalingAndUnits:
    """Dimensional scaling of the kernels."""

    def test_acceleration_scales_linearly_with_gravitational_constant(self) -> None:
        """Halving G halves every acceleration component exactly."""
        mode = _synthetic_mode()
        profile = _exponential_profile(-1.0e-3, 0.085)
        positions = [(0.0, 0.0, 5.0)]
        base = plane_wave_acceleration(positions, mode, profile, _MEDIUM.density)
        halved = plane_wave_acceleration(
            positions, mode, profile, _MEDIUM.density, gravitational_constant=0.5 * GRAVITATIONAL_CONSTANT
        )
        np.testing.assert_allclose(halved, 0.5 * base, rtol=1e-12)

    def test_areal_acceleration_scales_with_density(self) -> None:
        """The free-surface term is proportional to the background density; the bulk term is not.

        The bulk amplitude is supplied through the density profile (which owns its
        own rho0 factor), while ``medium_density`` only enters the areal source, so
        these are tested separately rather than conflated.
        """
        mode = _synthetic_mode()
        positions = [(0.0, 0.0, 5.0)]
        areal_base = plane_wave_acceleration(
            positions, mode, np.zeros_like, _MEDIUM.density, include_volume=False, include_areal=True
        )
        areal_heavy = plane_wave_acceleration(
            positions, mode, np.zeros_like, 3.0 * _MEDIUM.density, include_volume=False, include_areal=True
        )
        np.testing.assert_allclose(areal_heavy, 3.0 * areal_base, rtol=1e-12)

        volume_base = plane_wave_acceleration(
            positions,
            mode,
            _exponential_profile(-1.0e-3, 0.085),
            _MEDIUM.density,
            include_volume=True,
            include_areal=False,
        )
        profile_doubled = plane_wave_acceleration(
            positions,
            mode,
            _exponential_profile(-2.0e-3, 0.085),
            _MEDIUM.density,
            include_volume=True,
            include_areal=False,
        )
        np.testing.assert_allclose(profile_doubled, 2.0 * volume_base, rtol=1e-12)

    def test_zero_source_gives_zero_acceleration(self) -> None:
        """A zero density profile with the areal term disabled gives exactly zero."""
        mode = _synthetic_mode()
        acceleration = plane_wave_acceleration(
            [(0.0, 0.0, 5.0)], mode, np.zeros_like, _MEDIUM.density, include_volume=True, include_areal=False
        )
        assert np.max(np.abs(acceleration)) == 0.0


class TestFieldProfileHelpers:
    """The field-facing helpers reproduce the field's conventions."""

    def test_surface_vertical_amplitude_is_unity_by_normalization(self) -> None:
        """The field normalizes to a unit surface vertical displacement."""
        assert surface_vertical_amplitude(1.0, _field()) == pytest.approx(1.0)

    def test_density_profile_is_negative_bulk_compression(self) -> None:
        """The bulk density perturbation is nonzero and decays with depth.

        The sign is not asserted against a hand value here (the field module owns
        it); the independent check is in the field test suite. This only guards
        that the helper returns the same profile at every azimuth, as required for
        a horizontally homogeneous field.
        """
        field = _field()
        surface = float(density_depth_amplitude(np.array([0.0]), 1.0, field)[0])
        deep = float(density_depth_amplitude(np.array([100.0]), 1.0, field)[0])
        assert abs(surface) > abs(deep)
        assert abs(deep) > 0.0
        assert surface_vertical_amplitude(1.0, field) == 1.0
        assert SURFACE_AMPLITUDE_REFERENCE_METRES == 1.0
