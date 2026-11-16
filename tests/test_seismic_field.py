"""Tests for the Rayleigh-wave field's directional spectrum, covariance and realizations.

Covers the acceptance-level limits: zero-field, isotropic and directional
(the depth/single-mode limit's control lives in ``test_seismic_depth.py``),
plus the Hermitian-PSD-without-clipping and empirical-convergence
requirements. Each analytic-limit test compares the production
:func:`~gwmock_newtonian_noise.seismic.field.analytic_covariance` against a control
computed by a different method (:mod:`scipy.special` Bessel functions via a
closed-form directional-spectrum expansion), not by re-evaluating the
production code's own azimuth quadrature.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.special import i0, iv, j0, jv

from gwmock_newtonian_noise.seismic.field import (
    RayleighWaveFieldConfig,
    analytic_covariance,
    analytic_density_covariance,
    realize,
    steering_vectors,
)
from gwmock_newtonian_noise.seismic.medium import ElasticHalfSpace
from gwmock_newtonian_noise.seismic.spectrum import (
    power_law_psd,
    uniform_directional_density,
    von_mises_directional_density,
)

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal rock speeds/density used only to exercise the code, not calibrated
#: to any specific ET candidate site.
_MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED: a flat illustrative displacement PSD, not fit to any measured seismic spectrum.
_FLAT_PSD = power_law_psd(reference_frequency=1.0, reference_psd=1.0, exponent=0.0)

#: Convergence-tolerance record, written 2026-09-25.
#: Honest provenance: the 20% threshold is a sampling-theory design margin -- the relative
#: standard error of a Bartlett-averaged periodogram cross-spectral bin is of order
#: ``1 / sqrt(N_realizations) = 1 / sqrt(200) ~= 7%``, rounded up to a roughly 3x margin to
#: absorb finite-sample and band-edge effects. It was chosen independently of the measured
#: result, not tuned toward it. However, it was written into this tree after the first
#: measurement had already been reported, so it is not a preregistration: it does not
#: establish that the criterion predates the measurement. The measured value is not quoted
#: here, and the threshold is this round-number theory margin, not a number read off a run.
_CONVERGENCE_TOLERANCE = 0.20
_CONVERGENCE_REALIZATIONS = 200


def _uniform_config(**overrides) -> RayleighWaveFieldConfig:
    kwargs = {
        "medium": _MEDIUM,
        "surface_displacement_psd": _FLAT_PSD,
        "directional_density": uniform_directional_density,
        "n_azimuth": 256,
    }
    kwargs.update(overrides)
    return RayleighWaveFieldConfig(**kwargs)


class TestZeroFieldLimit:
    """A field with an identically-zero PSD produces an identically-zero covariance and realization."""

    def test_covariance_is_exactly_zero(self) -> None:
        """No amplitude means no correlation at all, not merely a small one."""
        config = _uniform_config(surface_displacement_psd=np.zeros_like)
        positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 5.0, 3.0)]
        cov = analytic_covariance(positions, np.array([0.5, 1.0, 3.0]), config)
        assert np.max(np.abs(cov)) == 0.0

    def test_realization_is_exactly_zero(self) -> None:
        """A zero-PSD realization is bit-exact zero, not merely small (no numerical DC leakage)."""
        config = _uniform_config(surface_displacement_psd=np.zeros_like)
        signals = realize([(0.0, 0.0, 0.0)], duration=4.0, sampling_frequency=16.0, config=config, seed=0)
        assert np.max(np.abs(signals[((0.0, 0.0, 0.0), "x")])) == 0.0


class TestIsotropicLimit:
    """Independent control: Aki (1957) spatial coherence of an isotropic surface-wave field.

    Aki, K. (1957), "Space and Time Spectra of Stationary Stochastic Waves,
    with Special Reference to Microtremors," Bulletin of the Earthquake
    Research Institute, 35, 415-456, shows that the normalized spatial
    cross-spectrum of an isotropic distribution of surface waves at
    separation r is J0(k*r), the zeroth-order Bessel function of the first
    kind. This is computed here with :func:`scipy.special.j0`, an
    implementation wholly independent of this package's own azimuth
    quadrature.
    """

    @pytest.mark.parametrize("separation", [3.0, 37.0, 150.0])
    @pytest.mark.parametrize("frequency", [0.5, 2.0, 5.0])
    def test_normalized_vertical_coherence_matches_bessel_j0(self, separation: float, frequency: float) -> None:
        """The normalized vertical-vertical coherence at the surface equals J0(k*r)."""
        config = _uniform_config()
        positions = [(0.0, 0.0, 0.0), (separation, 0.0, 0.0)]
        cov = analytic_covariance(positions, np.array([frequency]), config, components=("z",))
        s_u = float(_FLAT_PSD(np.array([frequency]))[0])
        coherence = cov[0, 0, 1] / s_u

        c = config.phase_velocity_at(np.array([frequency]))[0]
        k = 2.0 * np.pi * frequency / c
        control = j0(k * separation)

        np.testing.assert_allclose(coherence.real, control, atol=1e-6)
        np.testing.assert_allclose(coherence.imag, 0.0, atol=1e-8)


class TestDirectionalLimit:
    """Independent control: the Jacobi-Anger expansion of a von Mises directional spectrum.

    For directional density D(phi) = exp(kappa*cos(phi-mu)) / (2*pi*I0(kappa))
    (Mardia, K. V. and Jupp, P. E. (2000), *Directional Statistics*, Wiley,
    Sec. 3.5.4, for the von Mises density and its Fourier/circular-moment
    coefficients I_m(kappa)/I_0(kappa)) and the Jacobi-Anger expansion
    exp(-i*z*cos(phi)) = sum_m (-i)**m * J_m(z) * exp(i*m*phi) (a standard
    identity; see e.g. Abramowitz, M. and Stegun, I. A. (1964), *Handbook of
    Mathematical Functions*, Sec. 9.1.41-9.1.45, or any text on generating
    functions of Bessel functions), the exact spatial cross-spectrum at
    separation r, angle theta from the surface reference direction, reduces
    to a finite-M-truncated Bessel series:

        sum_m [I_m(kappa)/I_0(kappa)] * (-i)**m * J_m(k*r) * exp(i*m*(mu - theta))

    Computed here with :func:`scipy.special.iv` and :func:`scipy.special.jv`,
    independent of this package's own azimuth quadrature.
    """

    @staticmethod
    def _bessel_series_control(kappa: float, mu: float, kr: float, theta: float = 0.0, terms: int = 60) -> complex:
        total = 0j
        for m in range(-terms, terms + 1):
            total += (iv(m, kappa) / i0(kappa)) * (-1j) ** m * jv(m, kr) * np.exp(1j * m * (mu - theta))
        return total

    @pytest.mark.parametrize("kappa", [0.0, 1.0, 3.0, 8.0])
    @pytest.mark.parametrize("mu", [0.0, 0.6, 2.1])
    def test_normalized_vertical_coherence_matches_jacobi_anger_series(self, kappa: float, mu: float) -> None:
        """The normalized vertical-vertical coherence matches the independent Bessel-series control."""
        frequency = 2.0
        separation = 37.0
        density = von_mises_directional_density(mu, kappa)
        config = _uniform_config(directional_density=density)
        positions = [(0.0, 0.0, 0.0), (separation, 0.0, 0.0)]
        cov = analytic_covariance(positions, np.array([frequency]), config, components=("z",))
        s_u = float(_FLAT_PSD(np.array([frequency]))[0])
        coherence = cov[0, 0, 1] / s_u

        c = config.phase_velocity_at(np.array([frequency]))[0]
        k = 2.0 * np.pi * frequency / c
        control = self._bessel_series_control(kappa, mu, k * separation)

        np.testing.assert_allclose(coherence.real, control.real, atol=1e-6)
        np.testing.assert_allclose(coherence.imag, control.imag, atol=1e-6)

    def test_kappa_zero_reduces_to_the_isotropic_case(self) -> None:
        """Von Mises with kappa=0 is exactly uniform_directional_density on the quadrature grid."""
        density = von_mises_directional_density(mu=1.0, kappa=0.0)
        config = _uniform_config(directional_density=density)
        control_config = _uniform_config()
        np.testing.assert_allclose(config.azimuth_weights(), control_config.azimuth_weights(), atol=1e-12)


class TestHermitianPositiveSemidefinite:
    """Every frequency slice is Hermitian PSD by construction, with no eigenvalue clipping."""

    def test_hermitian_to_floating_point(self) -> None:
        """C - C^H is zero to floating-point precision, not merely approximately Hermitian."""
        config = _uniform_config(directional_density=von_mises_directional_density(mu=0.7, kappa=4.0))
        positions = [(0.0, 0.0, 0.0), (15.0, 5.0, 0.0), (5.0, -10.0, 3.0)]
        cov = analytic_covariance(positions, np.linspace(0.5, 5.0, 9), config, components=("x", "y", "z"))
        for slice_matrix in cov:
            np.testing.assert_allclose(slice_matrix, slice_matrix.conj().T, atol=1e-10)

    def test_positive_semidefinite_without_clipping(self) -> None:
        """Every eigenvalue is non-negative up to floating-point roundoff, unaided by any clip.

        This is a structural guarantee (a nonnegative-weighted sum of outer products), verified
        here numerically; :mod:`gwmock_newtonian_noise.seismic.field` never calls a clipping routine
        (no ``np.clip``/eigenvalue floor appears anywhere in that module).
        """
        config = _uniform_config(directional_density=von_mises_directional_density(mu=0.7, kappa=4.0))
        positions = [(0.0, 0.0, 0.0), (15.0, 5.0, 0.0), (5.0, -10.0, 3.0)]
        cov = analytic_covariance(positions, np.linspace(0.5, 5.0, 9), config, components=("x", "y", "z"))
        for slice_matrix in cov:
            eigenvalues = np.linalg.eigvalsh(slice_matrix)
            scale = np.max(np.abs(slice_matrix))
            assert eigenvalues.min() > -1e-8 * scale

    def test_no_clipping_helper_is_used_in_the_field_module(self) -> None:
        """Static guard: the field module's source never calls a clip/floor routine (docstring prose aside)."""
        import inspect

        from gwmock_newtonian_noise.seismic import field as field_module

        source = inspect.getsource(field_module)
        assert "np.clip" not in source
        assert "np.maximum" not in source
        assert "eigvals" not in source

    def test_density_covariance_is_hermitian_psd_too(self) -> None:
        """The scalar bulk-density-perturbation covariance is Hermitian PSD by the same construction."""
        config = _uniform_config(directional_density=von_mises_directional_density(mu=0.2, kappa=2.0))
        positions = [(0.0, 0.0, 1.0), (12.0, -4.0, 3.0)]
        cov = analytic_density_covariance(positions, np.linspace(0.5, 4.0, 5), config)
        for slice_matrix in cov:
            np.testing.assert_allclose(slice_matrix, slice_matrix.conj().T, atol=1e-10)
            eigenvalues = np.linalg.eigvalsh(slice_matrix)
            scale = np.max(np.abs(slice_matrix)) or 1.0
            assert eigenvalues.min() > -1e-8 * scale


class TestDimensionalChecks:
    """Basic dimensional/units sanity checks."""

    def test_density_covariance_scales_as_density_squared(self) -> None:
        """delta_rho ~ rho0, so its PSD scales as rho0**2 for fixed displacement PSD."""
        # UNANCHORED: illustrative scaling fixtures differing only in density; the point is the
        # exact 2x ratio, not these absolute values, which are not measured site densities. The
        # wave speeds are likewise the generic crustal UNANCHORED values used throughout.
        light = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=1000.0)
        heavy = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2000.0)
        positions = [(0.0, 0.0, 0.0)]
        cov_light = analytic_covariance(
            positions,
            np.array([2.0]),
            _uniform_config(medium=light),
            components=("density",),
        )
        cov_heavy = analytic_covariance(
            positions,
            np.array([2.0]),
            _uniform_config(medium=heavy),
            components=("density",),
        )
        np.testing.assert_allclose(cov_heavy[0, 0, 0].real, cov_light[0, 0, 0].real * 4.0, rtol=1e-10)

    def test_displacement_covariance_scales_linearly_with_psd_amplitude(self) -> None:
        """Doubling the input surface-displacement PSD exactly doubles the output covariance."""
        base = _uniform_config()
        doubled = _uniform_config(surface_displacement_psd=lambda f: 2.0 * _FLAT_PSD(f))
        positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        cov_base = analytic_covariance(positions, np.array([1.5]), base)
        cov_doubled = analytic_covariance(positions, np.array([1.5]), doubled)
        np.testing.assert_allclose(cov_doubled, cov_base * 2.0, rtol=1e-10)

    def test_non_positive_frequency_has_no_propagating_contribution(self) -> None:
        """DC and negative frequencies carry no Rayleigh-wave contribution by convention."""
        config = _uniform_config()
        cov = analytic_covariance([(0.0, 0.0, 0.0)], np.array([-1.0, 0.0]), config)
        assert np.max(np.abs(cov)) == 0.0


class TestEmpiricalConvergence:
    """Empirical realization covariance converges to the analytic covariance.

    The criterion is the sampling-theory design margin recorded at the
    ``_CONVERGENCE_TOLERANCE`` definition above: with
    ``_CONVERGENCE_REALIZATIONS = 200`` independent seeded realizations, the
    Bartlett-averaged periodogram cross-spectral estimate has an expected
    relative statistical fluctuation of order ``1/sqrt(200) ~= 7%``
    (standard periodogram-averaging sampling theory), and the 20% threshold
    is that theory value rounded up to a roughly 3x margin above it.

    The threshold was chosen independently of the measured result, not tuned
    toward it, but it was recorded in this tree only after the first
    measurement had already been reported. It is therefore a stated design
    margin, not a preregistration, and the assertion compares the
    200-realization average directly against ``analytic_covariance``.
    """

    def test_periodogram_average_converges_to_analytic_within_stated_tolerance(self) -> None:
        """Average |FFT|^2-based cross-spectral estimate over many realizations matches analytic."""
        duration = 8.0
        sampling_frequency = 16.0
        n_samples = round(duration * sampling_frequency)
        positions = [(0.0, 0.0, 0.0), (20.0, 0.0, 2.0)]
        config = _uniform_config(n_azimuth=48)

        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
        analytic = analytic_covariance(positions, frequencies, config, components=("z",))

        # One-sided folding. ``numpy.fft.rfft`` stores each interior positive-frequency bin
        # once from a real signal, so the two-sided-to-one-sided factor is 2 there; the Nyquist
        # bin (present only for even ``n_samples``) is already stored once and must not be
        # doubled. A uniform ``2/(fs*n)`` factor would bias the Nyquist diagonal by about 2x.
        one_sided = np.full(frequencies.size, 2.0 / (sampling_frequency * n_samples))
        if n_samples % 2 == 0:
            one_sided[-1] = 1.0 / (sampling_frequency * n_samples)

        accumulator = np.zeros((frequencies.size, len(positions), len(positions)), dtype=complex)
        for seed in range(_CONVERGENCE_REALIZATIONS):
            signals = realize(positions, duration, sampling_frequency, config, seed=seed, components=("z",))
            spectra = np.array([np.fft.rfft(signals[(p, "z")]) for p in positions])
            periodogram = one_sided[None, None, :] * spectra[:, None, :] * spectra.conj()[None, :, :]
            accumulator += np.transpose(periodogram, (2, 0, 1))
        empirical = accumulator / _CONVERGENCE_REALIZATIONS

        # A leakage cut at both band edges; the estimator itself is unbiased across the band.
        interior = (frequencies > 0.3) & (frequencies < sampling_frequency / 2.0 - 0.3)
        numerator = np.linalg.norm((empirical[interior] - analytic[interior]).reshape(-1))
        denominator = np.linalg.norm(analytic[interior].reshape(-1))
        relative_error = numerator / denominator

        assert relative_error < _CONVERGENCE_TOLERANCE, (
            f"empirical/analytic relative Frobenius error {relative_error:.4f} exceeded the "
            f"stated tolerance {_CONVERGENCE_TOLERANCE} at {_CONVERGENCE_REALIZATIONS} realizations"
        )

        # The Nyquist bin is excluded by the leakage cut above, so exercise the single-count
        # folding directly: the corrected estimator leaves the Nyquist diagonal within a few
        # percent of analytic, whereas a uniform 2/(fs*n) factor would place it near twice that.
        np.testing.assert_allclose(
            empirical[-1].diagonal().real,
            analytic[-1].diagonal().real,
            rtol=0.15,
            err_msg="Nyquist-bin periodogram diagonal is inconsistent with single-count folding",
        )

    def test_seeded_realization_is_reproducible(self) -> None:
        """The same seed reproduces bit-identical realizations."""
        config = _uniform_config()
        positions = [(0.0, 0.0, 0.0)]
        first = realize(positions, 2.0, 16.0, config, seed=7)
        second = realize(positions, 2.0, 16.0, config, seed=7)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_different_seeds_give_different_realizations(self) -> None:
        """Different seeds do not silently reproduce the same realization."""
        config = _uniform_config()
        positions = [(0.0, 0.0, 0.0)]
        first = realize(positions, 2.0, 16.0, config, seed=7)
        second = realize(positions, 2.0, 16.0, config, seed=8)
        assert not np.array_equal(first[(positions[0], "x")], second[(positions[0], "x")])


class TestConvergenceToleranceProvenance:
    """Static guard: the tolerance record is honest about when the criterion was set.

    It must disclose that the threshold postdates the first measurement (so it is a
    stated design margin, not a preregistration), and the threshold must be the round
    theory margin rather than a number read off a measurement.
    """

    def test_record_discloses_the_criterion_was_not_preregistered(self) -> None:
        """The record preceding the threshold states the sampling-theory margin and the timing caveat."""
        record = pathlib.Path(__file__).read_text().partition("_CONVERGENCE_TOLERANCE = 0.20")[0]
        assert "sampling-theory design margin" in record
        assert "not a preregistration" in record.lower()

    def test_tolerance_and_realization_count_are_the_round_design_values(self) -> None:
        """The threshold is the round 20% theory margin over 200 realizations, not a fitted value."""
        assert _CONVERGENCE_TOLERANCE == 0.20
        assert _CONVERGENCE_REALIZATIONS == 200


class TestConfigValidation:
    """Rejections for malformed configuration and inputs."""

    def test_rejects_too_few_azimuth_points(self) -> None:
        """A quadrature grid with fewer than four azimuth points cannot resolve a direction."""
        with pytest.raises(ValueError, match="n_azimuth"):
            _uniform_config(n_azimuth=2)

    def test_rejects_negative_directional_density(self) -> None:
        """A directional density that goes negative anywhere is not a valid density."""
        config = _uniform_config(directional_density=lambda phi: -np.ones_like(phi))
        with pytest.raises(ValueError, match="non-negative"):
            config.azimuth_weights()

    def test_rejects_all_zero_directional_density(self) -> None:
        """A directional density that vanishes everywhere on the grid cannot be normalized."""
        config = _uniform_config(directional_density=np.zeros_like)
        with pytest.raises(ValueError, match="positive somewhere"):
            config.azimuth_weights()

    def test_rejects_phase_velocity_shape_mismatch(self) -> None:
        """A dispersion override that returns the wrong shape is rejected, not silently broadcast."""
        config = _uniform_config(phase_velocity=lambda f: np.array([1000.0]))
        with pytest.raises(ValueError, match="phase_velocity"):
            config.phase_velocity_at(np.array([1.0, 2.0, 3.0]))

    def test_rejects_non_positive_frequency_in_steering_vectors(self) -> None:
        """steering_vectors requires a strictly-positive frequency (no propagating mode at f <= 0)."""
        config = _uniform_config()
        with pytest.raises(ValueError, match="frequency"):
            steering_vectors([(0.0, 0.0, 0.0)], 0.0, config)

    def test_rejects_negative_psd_in_covariance(self) -> None:
        """A surface_displacement_psd returning a negative value is rejected, not silently used."""
        config = _uniform_config(surface_displacement_psd=lambda f: -np.ones_like(f))
        with pytest.raises(ValueError, match="negative"):
            analytic_covariance([(0.0, 0.0, 0.0)], np.array([1.0]), config)

    def test_rejects_negative_psd_in_realize(self) -> None:
        """Realize rejects a negative PSD value the same way analytic_covariance does."""
        config = _uniform_config(surface_displacement_psd=lambda f: -np.ones_like(f))
        with pytest.raises(ValueError, match="negative"):
            realize([(0.0, 0.0, 0.0)], 2.0, 16.0, config, seed=0)

    def test_rejects_non_positive_duration(self) -> None:
        """A non-positive duration cannot be realized."""
        config = _uniform_config()
        with pytest.raises(ValueError, match="duration"):
            realize([(0.0, 0.0, 0.0)], 0.0, 16.0, config, seed=0)

    def test_rejects_non_positive_sampling_frequency(self) -> None:
        """A non-positive sampling frequency cannot be realized."""
        config = _uniform_config()
        with pytest.raises(ValueError, match="sampling_frequency"):
            realize([(0.0, 0.0, 0.0)], 2.0, 0.0, config, seed=0)

    def test_rejects_duration_too_short_for_two_samples(self) -> None:
        """A (duration, sampling_frequency) pair producing under two samples is rejected."""
        config = _uniform_config()
        with pytest.raises(ValueError, match="at least two samples"):
            realize([(0.0, 0.0, 0.0)], 0.01, 1.0, config, seed=0)


class TestUnanchoredInputsAreTagged:
    """Static guard: no numeric default in the spectrum module is left un-tagged."""

    def test_power_law_psd_docstring_tags_unanchored(self) -> None:
        """power_law_psd's own docstring flags itself and its numeric arguments as unanchored."""
        assert "UNANCHORED" in power_law_psd.__doc__

    def test_power_law_psd_has_no_numeric_default(self) -> None:
        """No reference frequency/PSD/exponent is silently defaulted -- every call must supply one."""
        import inspect

        signature = inspect.signature(power_law_psd)
        for parameter in signature.parameters.values():
            assert parameter.default is inspect.Parameter.empty
