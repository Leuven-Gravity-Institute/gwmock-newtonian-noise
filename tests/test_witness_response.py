"""Tests for the joint strain--witness response, covariance and realizations.

The production :func:`~gwmock_newtonian.witness.response.joint_covariance`
assembles one shared field into the three detector strains and every active
witness. These tests control it independently:

- the strain block is asserted *equal* to the unmodified
  :func:`~gwmock_newtonian.detector.response.strain_covariance`;
- the witness per-mode response is recomputed from the field's own
  :func:`~gwmock_newtonian.seismic.field.steering_vectors` primitive, not by
  calling the production witness helper again;
- analytic strain--witness and witness--witness covariances are compared with a
  seeded Monte-Carlo average of realizations, at a tolerance stated from
  sampling theory before it was measured;
- perfect / noisy / infinite-noise / missing-sensor limits, sensor permutations
  and displacement/velocity/acceleration unit conversions are checked as exact
  algebraic identities.

Every numeric fixture is UNANCHORED: illustrative values chosen to exercise the
code, not a representative ET site or sensor deployment.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian.detector.geometry import TriangleGeometry, equilateral_triangle
from gwmock_newtonian.detector.masses import shared_vertex_layout
from gwmock_newtonian.detector.response import DetectorResponseConfig, strain_covariance
from gwmock_newtonian.seismic.field import RayleighWaveFieldConfig, steering_vectors
from gwmock_newtonian.seismic.medium import ElasticHalfSpace
from gwmock_newtonian.seismic.spectrum import power_law_psd, uniform_directional_density
from gwmock_newtonian.witness.array import CorrelatedNoise, ExplicitNoise, WitnessArray, nested_disk_array
from gwmock_newtonian.witness.response import (
    joint_covariance,
    joint_mode_responses,
    joint_strain_covariance_block,
    realize_joint,
    synthesize_joint,
    witness_covariance,
    witness_mode_responses,
)
from gwmock_newtonian.witness.sensor import SensorQuantity, WitnessSensor, constant_response, quantity_transfer

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal medium used only to exercise the response pipeline.
_MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED: illustrative arm length and test-mass depth in metres.
_ARM_LENGTH = 10_000.0
_TEST_MASS_DEPTH = 20.0

#: UNANCHORED: a flat illustrative displacement PSD, not a measured site spectrum.
_FLAT_PSD = power_law_psd(reference_frequency=1.0, reference_psd=1.0, exponent=0.0)

#: Monte-Carlo convergence record. The three covariance blocks are scored
#: separately, because a single pooled Frobenius norm is dominated by the
#: witness block and cannot see the strain--witness block (an earlier
#: measurement reproduced a pooled 0.124 alongside a cross-block 0.300).
_MC_REALIZATIONS = 2000

#: Relative-Frobenius tolerance for the auto blocks (strain--strain and
#: witness--witness). The per-bin periodogram relative standard error is of
#: order ``1 / sqrt(2000) ~= 2.2%``; 0.20 is a roughly 9x sampling-theory
#: margin, chosen independently of the measured result.
_MC_AUTO_TOLERANCE = 0.20

#: Sampling sigmas allowed on the strain--witness block. The cross-spectrum
#: relative error has width ``~ 1 / (gamma_rms * sqrt(N))`` with ``gamma_rms``
#: the power-weighted RMS strain--witness coherence of the fixture, so the
#: tolerance is ``_MC_CROSS_SIGMAS / (gamma_rms * sqrt(N))``. This is the
#: coherence-based cross tolerance the cross-covariance claim needs, not the
#: auto-block ``1 / sqrt(N)`` margin.
_MC_CROSS_SIGMAS = 2.5

#: Bound on the normalized cross residual ratio
#: ``sum |emp - analytic|^2 / sum(C_ss C_ww / N)``. Its expectation is ~1
#: whatever the coherence, and it uses the auto-spectra in the denominator, so
#: zeroing or doubling the cross block drives it to ``~ N * gamma_rms**2``
#: even if the analytic cross were also zeroed.
_MC_CROSS_RATIO_MAX = 3.0

_SAMPLING_FREQUENCY = 32.0
_DURATION = 4.0

#: A relative eigenvalue floor for PSD checks.
_PSD_FLOOR = 1e-10


def _field(**overrides) -> RayleighWaveFieldConfig:
    """Return an isotropic field fixture on a generic crustal half-space."""
    kwargs = {
        "medium": _MEDIUM,
        "surface_displacement_psd": _FLAT_PSD,
        "directional_density": uniform_directional_density,
        "n_azimuth": 16,
    }
    kwargs.update(overrides)
    return RayleighWaveFieldConfig(**kwargs)


def _geometry() -> TriangleGeometry:
    """Return the canonical equilateral triangle at the fixture arm length."""
    return equilateral_triangle(_ARM_LENGTH)


def _detector_config(**overrides) -> DetectorResponseConfig:
    """Return a detector-response configuration wrapping the shared fixture field."""
    kwargs = {
        "field": _field(),
        "layout": shared_vertex_layout(_geometry(), test_mass_depth=_TEST_MASS_DEPTH),
        "n_depth": 48,
    }
    kwargs.update(overrides)
    return DetectorResponseConfig(**kwargs)


def _array(count: int = 4, **overrides) -> WitnessArray:
    """Return a nested UNANCHORED witness array."""
    kwargs = {"aperture": 400.0, "max_sensors": max(count, 4), "mode": "fixed_aperture"}
    kwargs.update(overrides)
    return nested_disk_array(count, **kwargs)


def _frequency_grid() -> np.ndarray:
    """Return the realization's rfft frequency grid."""
    n_samples = round(_DURATION * _SAMPLING_FREQUENCY)
    return np.fft.rfftfreq(n_samples, d=1.0 / _SAMPLING_FREQUENCY)


class TestPerModeResponses:
    """The per-mode responses are built from the field's own primitives."""

    def test_witness_response_matches_the_steering_primitive(self) -> None:
        """The witness response is the sensor transfer times the oriented field steering.

        The expected value is recomputed from
        :func:`~gwmock_newtonian.seismic.field.steering_vectors` directly, so a
        sign, axis or transfer error in the production helper fails this test.
        """
        field = _field()
        array = _array(3, self_noise_psd=None)
        frequency = 2.0
        responses = witness_mode_responses(np.array([frequency]), field, array)
        assert responses.shape == (1, field.azimuth_grid().size, 3)
        positions = [sensor.position for sensor in array.active_sensors]
        steering = steering_vectors(positions, frequency, field)
        for index, sensor in enumerate(array.active_sensors):
            axis = sensor.unit_orientation
            expected = np.zeros(field.azimuth_grid().size, dtype=complex)
            for component_index, component in enumerate(("x", "y", "z")):
                expected += axis[component_index] * steering[component][:, index]
            expected *= sensor.transfer(frequency)
            np.testing.assert_allclose(responses[0, :, index], expected, rtol=1e-12, atol=0.0)

    def test_witness_response_is_zero_for_a_zero_psd_field(self) -> None:
        """A field with no power excites no witness mode."""
        field = _field(surface_displacement_psd=np.zeros_like)
        responses = witness_mode_responses(np.array([0.5, 2.0]), field, _array(3))
        assert np.max(np.abs(responses)) == 0.0

    def test_joint_response_places_the_m6_strain_first(self) -> None:
        """The joint response's strain block is the detector response and the rest the witnesses."""
        from gwmock_newtonian.detector.response import _plane_wave_kernel, mode_strain

        config = _detector_config()
        array = _array(3)
        frequency = 2.0
        responses = joint_mode_responses(np.array([frequency]), config, array)
        assert responses.shape == (1, config.field.azimuth_grid().size, 6)
        kernel = _plane_wave_kernel(config, frequency)
        for azimuth_index, azimuth in enumerate(config.field.azimuth_grid()):
            np.testing.assert_allclose(
                responses[0, azimuth_index, :3],
                mode_strain(config, frequency, float(azimuth), kernel),
                rtol=1e-12,
                atol=0.0,
            )
        witness = witness_mode_responses(np.array([frequency]), config.field, array)
        np.testing.assert_allclose(responses[:, :, 3:], witness, rtol=1e-12, atol=0.0)


class TestFieldAnchoredCovariance:
    """Independent anchors for the witness and cross covariances."""

    def test_vertical_witness_covariance_equals_the_field_z_covariance(self) -> None:
        """Vertical displacement witnesses reproduce the field's own z-z analytic covariance exactly."""
        from gwmock_newtonian.seismic.field import analytic_covariance

        field = _field()
        array = _array(4, self_noise_psd=None, orientation=(0.0, 0.0, 1.0))
        frequencies = np.array([0.5, 2.0, 4.0])
        positions = [sensor.position for sensor in array.active_sensors]
        anchor = analytic_covariance(positions, frequencies, field, components=("z",))
        covariance = witness_covariance(frequencies, field, array)
        np.testing.assert_allclose(covariance.matrix, anchor, rtol=1e-12, atol=0.0)

    def test_strain_witness_cross_block_reproduces_the_shared_mode_sum(self) -> None:
        """The cross block equals the explicit shared-mode outer-product sum from primitives."""
        from gwmock_newtonian.detector.response import _plane_wave_kernel, mode_strain

        config = _detector_config()
        array = _array(3, self_noise_psd=None)
        field = config.field
        frequencies = np.array([2.0])
        weights = field.azimuth_weights()
        witness = witness_mode_responses(frequencies, field, array)
        expected = np.zeros((1, 3, 3), dtype=complex)
        for index, frequency in enumerate(frequencies):
            surface_psd = float(np.asarray(field.surface_displacement_psd(np.array([frequency])))[0])
            kernel = _plane_wave_kernel(config, float(frequency))
            for azimuth_index, azimuth in enumerate(field.azimuth_grid()):
                strain = mode_strain(config, float(frequency), float(azimuth), kernel)
                expected[index] += (
                    surface_psd * weights[azimuth_index] * np.outer(strain, witness[index, azimuth_index].conj())
                )
        joint = joint_covariance(frequencies, config, array)
        np.testing.assert_allclose(joint.cross_block(), expected, rtol=1e-12, atol=0.0)


class TestJointCovarianceStructure:
    """The joint covariance reuses the detector strain operator and is Hermitian PSD."""

    def test_strain_block_equals_the_m6_covariance(self) -> None:
        """The strain block is exactly the unmodified detector strain covariance."""
        config = _detector_config()
        frequencies = np.array([0.5, 2.0, 5.0])
        joint = joint_covariance(frequencies, config, _array(4))
        expected = strain_covariance(frequencies, config).matrix
        np.testing.assert_allclose(joint.strain_block(), expected, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(joint_strain_covariance_block(frequencies, config), expected, rtol=1e-12, atol=0.0)

    def test_channel_order_is_detectors_then_witnesses(self) -> None:
        """Channel order labels the matrix axes as three strains then the active witnesses."""
        joint = joint_covariance(np.array([2.0]), _detector_config(), _array(3))
        assert joint.channel_order == ("ET1", "ET2", "ET3", "W0", "W1", "W2")
        assert joint.witness_channel_order == ("W0", "W1", "W2")

    def test_every_slice_is_hermitian_positive_semidefinite(self) -> None:
        """A non-negative weighted sum of outer products is Hermitian PSD by construction."""
        array = _array(4, self_noise_psd=lambda frequency: 0.1, noise=CorrelatedNoise(300.0))
        joint = joint_covariance(np.array([0.5, 2.0, 5.0]), _detector_config(), array)
        for slice_matrix in joint.matrix:
            np.testing.assert_allclose(slice_matrix, slice_matrix.conj().T, atol=0.0)
            eigenvalues = np.linalg.eigvalsh(slice_matrix)
            assert np.min(eigenvalues) >= -_PSD_FLOOR * max(1.0, float(np.max(np.abs(eigenvalues))))

    def test_witness_block_equals_the_standalone_witness_covariance(self) -> None:
        """The witness block matches :func:`witness_covariance` on the same field and array."""
        config = _detector_config()
        array = _array(4, self_noise_psd=lambda frequency: 0.2, noise=CorrelatedNoise(250.0))
        frequencies = np.array([0.5, 2.0])
        joint = joint_covariance(frequencies, config, array)
        standalone = witness_covariance(frequencies, config.field, array)
        np.testing.assert_allclose(joint.witness_block(), standalone.matrix, rtol=1e-12, atol=0.0)
        assert standalone.channel_order == array.channel_names

    def test_rejects_overlapping_channel_names(self) -> None:
        """A witness named like a detector channel is rejected, not silently aliased."""
        array = WitnessArray(sensors=(WitnessSensor(name="ET1", position=(0.0, 0.0, 0.0)),))
        with pytest.raises(ValueError, match="disjoint"):
            joint_covariance(np.array([2.0]), _detector_config(), array)


class TestMonteCarloConvergence:
    """Analytic cross-covariances agree with sampled realizations, block by block.

    The estimator is the one-sided folded periodogram ``conj(X_a) X_b`` of a
    seeded spectral-representation realization, averaged over
    ``_MC_REALIZATIONS`` seeds. The strain--strain, strain--witness and
    witness--witness blocks are scored and asserted separately; the pooled norm
    is reported only as a supplemental summary. The strain--witness threshold is derived from the
    fixture's coherence (see ``_MC_CROSS_SIGMAS``), and a normalized residual
    ratio guards the cross block independently of that threshold.
    """

    def _empirical(self, config, array, responses, seeds: int) -> np.ndarray:
        """Return the seed-averaged one-sided periodogram covariance of the joint channels."""
        n_samples = round(_DURATION * _SAMPLING_FREQUENCY)
        frequencies = _frequency_grid()
        one_sided = np.full(frequencies.size, 2.0 / (_SAMPLING_FREQUENCY * n_samples))
        if n_samples % 2 == 0:
            one_sided[-1] = 1.0 / (_SAMPLING_FREQUENCY * n_samples)
        accumulator = np.zeros((frequencies.size, len(array.channel_names) + 3, len(array.channel_names) + 3), complex)
        for seed in range(seeds):
            signals = synthesize_joint(responses, config, array, _DURATION, _SAMPLING_FREQUENCY, seed)
            order = (*config.layout.geometry.names, *array.channel_names)
            spectra = np.array([np.fft.rfft(signals[name]) for name in order])
            # conj(X_a) X_b: a channel is Re(c exp(-i omega t)), so its rfft is
            # proportional to conj(c) and C_ab = E[c_a conj(c_b)] (scipy.signal.csd order).
            periodogram = one_sided[None, None, :] * spectra.conj()[:, None, :] * spectra[None, :, :]
            accumulator += np.transpose(periodogram, (2, 0, 1))
        return accumulator / seeds

    def _interior(self) -> np.ndarray:
        """Return the leakage-cut interior frequency mask."""
        frequencies = _frequency_grid()
        return (frequencies > 0.3) & (frequencies < _SAMPLING_FREQUENCY / 2.0 - 0.3)

    @staticmethod
    def _relative_error(analytic_block: np.ndarray, empirical_block: np.ndarray, interior: np.ndarray) -> float:
        """Return the relative Frobenius error of one covariance block on the interior bins."""
        return float(
            np.linalg.norm((empirical_block[interior] - analytic_block[interior]).reshape(-1))
            / np.linalg.norm(analytic_block[interior].reshape(-1))
        )

    @staticmethod
    def _power_weighted_coherence(joint, interior: np.ndarray) -> float:
        """Return the power-weighted RMS strain--witness coherence of the analytic covariance."""
        strain_psd = np.real(np.diagonal(joint.strain_block(), axis1=1, axis2=2))
        witness_psd = np.real(np.diagonal(joint.witness_block(), axis1=1, axis2=2))
        denominator = np.sqrt(strain_psd[:, :, None] * witness_psd[:, None, :])
        safe = np.where(denominator > 0.0, denominator, 1.0)
        coherence_squared = np.where(denominator > 0.0, np.abs(joint.cross_block()) ** 2 / safe**2, 0.0)
        weight = np.abs(joint.cross_block()) ** 2
        return float(np.sqrt(np.sum(weight[interior] * coherence_squared[interior]) / np.sum(weight[interior])))

    @staticmethod
    def _cross_residual_ratio(joint, empirical_cross: np.ndarray, seeds: int, interior: np.ndarray) -> float:
        """Return the cross-block residual normalized by its sampling variance from the auto-spectra."""
        strain_psd = np.real(np.diagonal(joint.strain_block(), axis1=1, axis2=2))
        witness_psd = np.real(np.diagonal(joint.witness_block(), axis1=1, axis2=2))
        variance = strain_psd[:, :, None] * witness_psd[:, None, :] / seeds
        return float(
            np.sum(np.abs(empirical_cross[interior] - joint.cross_block()[interior]) ** 2) / np.sum(variance[interior])
        )

    def test_blocks_converge_separately(self) -> None:
        """Each noiseless covariance block is reproduced within its own sampling tolerance."""
        config = _detector_config()
        array = _array(4, self_noise_psd=None)
        frequencies = _frequency_grid()
        responses = joint_mode_responses(frequencies, config, array)
        joint = joint_covariance(frequencies, config, array)
        empirical = self._empirical(config, array, responses, _MC_REALIZATIONS)
        interior = self._interior()

        strain_error = self._relative_error(joint.strain_block(), empirical[:, :3, :3], interior)
        cross_error = self._relative_error(joint.cross_block(), empirical[:, :3, 3:], interior)
        witness_error = self._relative_error(joint.witness_block(), empirical[:, 3:, 3:], interior)
        pooled_error = self._relative_error(joint.matrix, empirical, interior)  # supplemental only
        coherence = self._power_weighted_coherence(joint, interior)
        if coherence <= 0.0:
            raise AssertionError("analytic strain-witness coherence is zero; the cross block is not testable")
        cross_tolerance = _MC_CROSS_SIGMAS / (coherence * np.sqrt(_MC_REALIZATIONS))
        cross_ratio = self._cross_residual_ratio(joint, empirical[:, :3, 3:], _MC_REALIZATIONS, interior)

        context = (
            f"strain-strain={strain_error:.4f}, strain-witness={cross_error:.4f}, "
            f"witness-witness={witness_error:.4f}, pooled(supplemental)={pooled_error:.4f}, "
            f"gamma_rms={coherence:.3f}, cross tolerance={cross_tolerance:.4f}, cross ratio={cross_ratio:.3f}"
        )
        assert strain_error < _MC_AUTO_TOLERANCE, context
        assert witness_error < _MC_AUTO_TOLERANCE, context
        assert cross_error < cross_tolerance, context
        assert cross_ratio < _MC_CROSS_RATIO_MAX, context

    def test_cross_block_phases_converge_with_a_timing_offset(self) -> None:
        """A timing offset makes the strain--witness block complex; its phase must still converge.

        With untimed sensors in an isotropic field the cross block is real, so
        an estimator that conjugates it (``X_a conj(X_b)`` instead of
        ``conj(X_a) X_b``) scores identically and the convergence check above
        cannot see the sign convention. A 50 ms offset gives the cross block an
        imaginary part of ~8% of its norm, which the conjugated estimator turns
        into a residual far outside the sampling bound.
        """
        config = _detector_config()
        array = _array(4, self_noise_psd=None, timing_offset=0.05)
        frequencies = _frequency_grid()
        responses = joint_mode_responses(frequencies, config, array)
        joint = joint_covariance(frequencies, config, array)
        interior = self._interior()
        cross = joint.cross_block()[interior]
        assert np.linalg.norm(cross.imag) > 0.05 * np.linalg.norm(cross)
        empirical = self._empirical(config, array, responses, _MC_REALIZATIONS)
        coherence = self._power_weighted_coherence(joint, interior)
        cross_error = self._relative_error(joint.cross_block(), empirical[:, :3, 3:], interior)
        cross_tolerance = _MC_CROSS_SIGMAS / (coherence * np.sqrt(_MC_REALIZATIONS))
        cross_ratio = self._cross_residual_ratio(joint, empirical[:, :3, 3:], _MC_REALIZATIONS, interior)
        context = f"strain-witness={cross_error:.4f} (tolerance {cross_tolerance:.4f}), cross ratio={cross_ratio:.3f}"
        assert cross_error < cross_tolerance, context
        assert cross_ratio < _MC_CROSS_RATIO_MAX, context

    def test_correlated_sensor_noise_converges(self) -> None:
        """A correlated-noise-only ensemble is reproduced by the covariance model."""
        config = _detector_config(
            field=_field(surface_displacement_psd=np.zeros_like),
        )
        array = _array(
            4,
            self_noise_psd=lambda frequency: 1.0,
            noise=CorrelatedNoise(correlation_length=250.0),
        )
        frequencies = _frequency_grid()
        responses = joint_mode_responses(frequencies, config, array)
        analytic = joint_covariance(frequencies, config, array)
        empirical = self._empirical(config, array, responses, _MC_REALIZATIONS)
        interior = self._interior()
        witness_error = self._relative_error(analytic.witness_block(), empirical[:, 3:, 3:], interior)
        # With a zero field the strain and cross blocks are exactly zero in both.
        assert np.max(np.abs(empirical[:, :3, :])) == 0.0
        assert np.max(np.abs(analytic.matrix[:, :3, :])) == 0.0
        assert witness_error < _MC_AUTO_TOLERANCE, f"correlated-noise witness relative error {witness_error:.4f}"

    def test_realize_joint_is_seed_reproducible(self) -> None:
        """The same seed reproduces bit-identical joint realizations."""
        config = _detector_config()
        array = _array(3, self_noise_psd=lambda frequency: 0.1)
        first = realize_joint(config, array, _DURATION, _SAMPLING_FREQUENCY, 11)
        second = realize_joint(config, array, _DURATION, _SAMPLING_FREQUENCY, 11)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


class TestLimits:
    """Perfect, noisy, infinite-noise and missing-sensor limits."""

    def test_perfect_sensors_equal_a_zero_noise_model(self) -> None:
        """Noiseless sensors give exactly the field-only covariance."""
        config = _detector_config()
        array = _array(4, self_noise_psd=None)
        frequencies = np.array([0.5, 2.0])
        implicit = joint_covariance(frequencies, config, array).witness_block()
        explicit = joint_covariance(
            frequencies,
            config,
            _array(4, self_noise_psd=lambda frequency: 0.0),
        ).witness_block()
        np.testing.assert_allclose(implicit, explicit, atol=0.0)

    def test_noise_adds_exactly_to_the_witness_block(self) -> None:
        """Self-noise adds its diagonal to the witness block and leaves the strains and cross terms alone."""
        config = _detector_config()
        array = _array(4, self_noise_psd=None)
        noisy = _array(4, self_noise_psd=lambda frequency: 3.0)
        frequencies = np.array([0.5, 2.0])
        clean, dirty = joint_covariance(frequencies, config, array), joint_covariance(frequencies, config, noisy)
        np.testing.assert_allclose(dirty.strain_block(), clean.strain_block(), atol=0.0)
        np.testing.assert_allclose(dirty.cross_block(), clean.cross_block(), atol=0.0)
        np.testing.assert_allclose(
            dirty.witness_block() - clean.witness_block(),
            3.0 * np.broadcast_to(np.eye(4), (frequencies.size, 4, 4)),
            atol=0.0,
        )

    def test_infinite_noise_drives_the_strain_witness_coherence_to_zero(self) -> None:
        """As self-noise diverges, the normalized strain--witness coherence vanishes."""
        config = _detector_config()
        frequency = np.array([2.0])

        def coherence(self_noise_psd: float | None) -> float:
            noise = None if self_noise_psd is None else (lambda f: self_noise_psd)
            covariance = joint_covariance(frequency, config, _array(1, self_noise_psd=noise))
            strain_psd = float(covariance.strain_block()[0, 0, 0].real)
            witness_psd = float(covariance.witness_block()[0, 0, 0].real)
            cross = complex(covariance.cross_block()[0, 0, 0])
            return abs(cross) / np.sqrt(strain_psd * witness_psd)

        # The limit is only meaningful if the noiseless witness is coherent with the strain.
        assert coherence(None) > 1e-2
        ladder = [coherence(level) for level in (1.0, 1.0e6, 1.0e24)]
        assert coherence(None) > ladder[0] > ladder[1] > ladder[2]
        assert ladder[2] < 1e-10

    def test_missing_channels_select_a_submatrix(self) -> None:
        """Dropping sensors takes the corresponding rows and columns of the full covariance."""
        config = _detector_config()
        full_array = _array(4, self_noise_psd=lambda frequency: 0.5)
        masked_array = full_array.with_missing(("W1",))
        frequencies = np.array([0.5, 2.0])
        full = joint_covariance(frequencies, config, full_array).matrix
        masked = joint_covariance(frequencies, config, masked_array).matrix
        keep = [0, 1, 2, 3, 5, 6]
        np.testing.assert_allclose(masked, full[:, keep][:, :, keep], atol=0.0)

    def test_all_channels_missing_reduces_to_the_strain_covariance(self) -> None:
        """A fully masked array leaves only the three detector strain channels."""
        config = _detector_config()
        frequencies = np.array([0.5, 2.0])
        masked = joint_covariance(frequencies, config, _array(3).with_missing(("W0", "W1", "W2")))
        assert masked.matrix.shape == (2, 3, 3)
        np.testing.assert_allclose(masked.matrix, strain_covariance(frequencies, config).matrix, rtol=1e-12, atol=0.0)


class TestInvariances:
    """Sensor permutations and displacement/velocity/acceleration unit conversions."""

    def test_permuted_sensors_permute_the_joint_covariance(self) -> None:
        """Reordering sensors permutes rows and columns by the same order."""
        config = _detector_config()
        array = _array(4, self_noise_psd=lambda frequency: 0.3)
        order = (2, 0, 3, 1)
        frequencies = np.array([0.5, 2.0])
        original = joint_covariance(frequencies, config, array)
        permuted = joint_covariance(frequencies, config, array.permuted(order))
        selection = [0, 1, 2, *(3 + index for index in order)]
        assert permuted.channel_order == tuple(original.channel_order[index] for index in selection)
        np.testing.assert_allclose(permuted.matrix, original.matrix[:, selection][:, :, selection], atol=0.0)

    def test_velocity_and_acceleration_scale_by_the_transfer(self) -> None:
        """The covariance transforms by the exact quantity transfer under a unit change.

        With ``V -> T(f) V`` on the witness side, the witness block scales by
        ``|T|**2`` and the strain--witness cross block by ``conj(T)``; the
        transfer is the displacement-to-quantity factor of
        :func:`~gwmock_newtonian.witness.sensor.quantity_transfer`.
        """
        config = _detector_config()
        frequencies = np.array([0.5, 2.0, 4.0])
        displacement = joint_covariance(
            frequencies, config, _array(3, self_noise_psd=None, quantity=SensorQuantity.DISPLACEMENT)
        )
        for quantity, transfer in (
            (SensorQuantity.VELOCITY, [quantity_transfer(SensorQuantity.VELOCITY, f) for f in frequencies]),
            (SensorQuantity.ACCELERATION, [quantity_transfer(SensorQuantity.ACCELERATION, f) for f in frequencies]),
        ):
            converted = joint_covariance(frequencies, config, _array(3, self_noise_psd=None, quantity=quantity))
            factor = np.asarray(transfer, dtype=complex)[:, None, None]
            np.testing.assert_allclose(converted.strain_block(), displacement.strain_block(), atol=0.0)
            np.testing.assert_allclose(
                converted.witness_block(), displacement.witness_block() * np.abs(factor) ** 2, rtol=1e-12, atol=0.0
            )
            np.testing.assert_allclose(
                converted.cross_block(), displacement.cross_block() * np.conj(factor), rtol=1e-12, atol=0.0
            )

    def test_response_and_timing_scale_the_witness_block(self) -> None:
        """A constant complex calibration and response scale the witness power by its squared magnitude."""
        config = _detector_config()
        frequencies = np.array([0.5, 2.0])
        base = joint_covariance(frequencies, config, _array(3, self_noise_psd=None))
        scaled = joint_covariance(
            frequencies,
            config,
            _array(3, self_noise_psd=None, calibration=2.0 + 1.0j, response=constant_response(0.5 - 0.25j)),
        )
        gain = abs((2.0 + 1.0j) * (0.5 - 0.25j)) ** 2
        np.testing.assert_allclose(scaled.witness_block(), base.witness_block() * gain, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(scaled.strain_block(), base.strain_block(), atol=0.0)


class TestManifestIntegration:
    """The joint covariance carries the array manifest."""

    def test_joint_covariance_embeds_the_array_manifest(self) -> None:
        """The embedded manifest equals the array's own manifest."""
        array = _array(4, self_noise_psd=lambda frequency: 2.0)
        config = _detector_config()
        joint = joint_covariance(np.array([2.0]), config, array, manifest_frequency=3.0)
        manifest = array.manifest(reference_frequency=3.0)
        assert joint.array_manifest.to_dict() == manifest.to_dict()


class TestExplicitNoiseIntegration:
    """An explicit non-diagonal noise covariance is honoured end to end.

    Regression cases for the sensor-identity fix: a non-diagonal
    matrix must follow a sensor permutation and must reduce to the active
    principal submatrix under a missing mask.
    """

    def test_explicit_rank_deficient_noise_is_reported(self) -> None:
        """A rank-one explicit noise gives noise effective rank one while the raw count is four."""
        matrix = np.ones((4, 4), dtype=complex)

        def covariance(frequency: float) -> np.ndarray:
            del frequency
            return matrix

        array = _array(
            4,
            self_noise_psd=None,
            noise=ExplicitNoise(channel_order=("W0", "W1", "W2", "W3"), covariance_callable=covariance),
        )
        manifest = array.manifest(reference_frequency=2.0)
        assert manifest.raw_sensor_count == 4
        assert manifest.noise_effective_rank == 1

    def test_noiseless_array_noise_rank_differs_from_covariance_rank(self) -> None:
        """A noiseless array has noise rank zero but full field-plus-noise rank.

        The manifest ranks the noise matrix alone; the covariance ranks the
        field plus noise. On the same noiseless four-sensor array at 2 Hz the
        two must stay distinct, so neither can be read as the other.
        """
        array = _array(4, self_noise_psd=None)
        covariance = witness_covariance(np.array([2.0]), _field(), array, manifest_frequency=2.0)
        assert covariance.array_manifest.noise_effective_rank == 0
        assert covariance.effective_rank().tolist() == [4]
        assert not hasattr(covariance.array_manifest, "effective_rank")

    def test_noiseless_array_noise_participation_ratio_differs_from_covariance_ratio(self) -> None:
        """A noiseless array has noise participation ratio zero but a field-plus-noise ratio above one.

        The manifest's ratio covers the noise matrix alone; the covariance's
        covers the field plus noise. On the same noiseless four-sensor array at
        2 Hz the two must stay distinct, so neither can be read as the other.
        """
        array = _array(4, self_noise_psd=None)
        covariance = witness_covariance(np.array([2.0]), _field(), array, manifest_frequency=2.0)
        assert covariance.array_manifest.noise_participation_ratio == 0.0
        (field_ratio,) = covariance.participation_ratio().tolist()
        assert 1.0 < field_ratio <= 4.0
        assert not hasattr(covariance.array_manifest, "participation_ratio")

    def test_explicit_noise_follows_permutation_and_missing_in_the_joint_covariance(self) -> None:
        """The witness noise block permutes and masks like the array channels."""
        matrix = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=complex)

        def covariance(frequency: float) -> np.ndarray:
            del frequency
            return matrix

        sensors = (
            WitnessSensor(name="A", position=(0.0, 0.0, 0.0)),
            WitnessSensor(name="B", position=(40.0, 0.0, 0.0)),
        )
        array = WitnessArray(
            sensors=sensors,
            noise=ExplicitNoise(channel_order=("A", "B"), covariance_callable=covariance),
        )
        config = _detector_config(field=_field(surface_displacement_psd=np.zeros_like))
        frequencies = np.array([2.0])
        original = joint_covariance(frequencies, config, array).witness_block()
        np.testing.assert_allclose(original[0], matrix, atol=0.0)

        permuted = joint_covariance(frequencies, config, array.permuted((1, 0))).witness_block()
        np.testing.assert_allclose(permuted[0], matrix[::-1, ::-1], atol=0.0)

        masked = joint_covariance(frequencies, config, array.with_missing(("B",))).witness_block()
        np.testing.assert_allclose(masked[0], [[2.0]], atol=0.0)
