"""Tests for witness sensors, arrays, noise models, manifests and nested families.

These tests control the sensor model against the field's own steering primitive
and against exact algebra (quantity transfers, timing phases, correlated-noise
ranks), and they pin the nested-array invariants the design requires: the
aperture and every per-sensor property are fixed across ``N``, and the manifest
reports the raw sensor count and the effective rank separately.

All numeric fixtures are UNANCHORED illustrative values chosen only to exercise
the code, not a representative ET sensor deployment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gwmock_newtonian.witness.array import (
    CorrelatedNoise,
    DiagonalNoise,
    ExplicitNoise,
    WitnessArray,
    effective_rank,
    nested_disk_array,
    participation_ratio,
    sunflower_positions,
)
from gwmock_newtonian.witness.sensor import (
    SensorQuantity,
    WitnessSensor,
    constant_response,
    quantity_transfer,
    unit_response,
)

pytestmark = pytest.mark.unit


def _flat_psd(frequency: float) -> float:
    """Return an UNANCHORED unit self-noise PSD, so every frequency has noise."""
    del frequency
    return 1.0


class TestQuantityTransfer:
    """The displacement-to-quantity transfer follows the field's exp(-i w t) convention."""

    def test_displacement_is_identity(self) -> None:
        """The displacement transfer is exactly one."""
        assert quantity_transfer(SensorQuantity.DISPLACEMENT, 3.0) == 1.0 + 0.0j

    def test_velocity_is_minus_i_omega(self) -> None:
        """Velocity is the first time derivative, ``-i * omega``."""
        frequency = 3.0
        assert quantity_transfer(SensorQuantity.VELOCITY, frequency) == pytest.approx(-1j * 2.0 * np.pi * frequency)

    def test_acceleration_is_minus_omega_squared(self) -> None:
        """Acceleration is the second time derivative, ``-omega**2``."""
        frequency = 3.0
        assert quantity_transfer(SensorQuantity.ACCELERATION, frequency) == pytest.approx(
            -((2.0 * np.pi * frequency) ** 2)
        )

    def test_velocity_over_displacement_is_i_omega(self) -> None:
        """The ratio between the velocity and displacement transfers is the derivative factor."""
        frequency = 2.5
        ratio = quantity_transfer(SensorQuantity.VELOCITY, frequency) / quantity_transfer(
            SensorQuantity.DISPLACEMENT, frequency
        )
        assert ratio == pytest.approx(-1j * 2.0 * np.pi * frequency)

    def test_rejects_non_positive_frequency(self) -> None:
        """A non-positive frequency is unphysical."""
        with pytest.raises(ValueError, match="frequency"):
            quantity_transfer(SensorQuantity.DISPLACEMENT, 0.0)


class TestSensorModel:
    """Sensor validation and the assembled complex transfer."""

    def test_orientation_is_normalized(self) -> None:
        """The sensing axis is stored raw but exposed normalized."""
        sensor = WitnessSensor(name="W", position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 3.0))
        np.testing.assert_allclose(sensor.unit_orientation, [0.0, 0.0, 1.0], atol=0.0)

    def test_rejects_a_zero_orientation(self) -> None:
        """A zero sensing axis cannot define a projection."""
        with pytest.raises(ValueError, match="non-zero"):
            WitnessSensor(name="W", position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0))

    def test_rejects_empty_name_or_wrong_position(self) -> None:
        """The identifier and position are validated."""
        with pytest.raises(ValueError, match="name"):
            WitnessSensor(name="", position=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="three components"):
            WitnessSensor(name="W", position=(0.0, 0.0))

    def test_rejects_a_non_callable_response(self) -> None:
        """The instrument response must be callable."""
        with pytest.raises(TypeError, match="response"):
            WitnessSensor(name="W", position=(0.0, 0.0, 0.0), response=1.0)  # type: ignore[arg-type]

    def test_default_response_is_identity(self) -> None:
        """The default instrument response is the identity."""
        sensor = WitnessSensor(name="W", position=(0.0, 0.0, 0.0))
        assert sensor.response(5.0) == 1.0
        assert unit_response(5.0) == 1.0
        assert constant_response(2.0 - 1.0j)(7.0) == 2.0 - 1.0j

    def test_calibration_scales_the_transfer(self) -> None:
        """Calibration multiplies the transfer by a complex scalar."""
        sensor = WitnessSensor(name="W", position=(0.0, 0.0, 0.0), calibration=2.0 * np.exp(0.3j))
        assert sensor.transfer(1.5) == pytest.approx(2.0 * np.exp(0.3j))

    def test_timing_offset_is_a_pure_phase(self) -> None:
        """A timing offset contributes ``exp(+i * omega * tau)`` and no magnitude."""
        tau = 0.004
        frequency = 5.0
        sensor = WitnessSensor(name="W", position=(0.0, 0.0, 0.0), timing_offset=tau)
        transfer = sensor.transfer(frequency)
        assert abs(transfer) == pytest.approx(1.0)
        assert np.angle(transfer) == pytest.approx(2.0 * np.pi * frequency * tau)

    def test_transfer_composes_quantity_response_and_timing(self) -> None:
        """The full transfer is calibration * response * quantity transfer * timing."""
        frequency = 4.0
        tau = 0.002
        sensor = WitnessSensor(
            name="W",
            position=(0.0, 0.0, 0.0),
            quantity=SensorQuantity.ACCELERATION,
            response=constant_response(3.0 + 0.0j),
            calibration=0.5 + 0.5j,
            timing_offset=tau,
        )
        response, derivative = 3.0 + 0j, -((2.0 * np.pi * frequency) ** 2)
        expected = (0.5 + 0.5j) * response * derivative * np.exp(1j * 2.0 * np.pi * frequency * tau)
        assert sensor.transfer(frequency) == pytest.approx(expected)

    def test_self_noise_defaults_to_zero_and_rejects_negative(self) -> None:
        """A sensor without a PSD is noiseless; a negative PSD is rejected."""
        assert WitnessSensor(name="W", position=(0.0, 0.0, 0.0)).self_noise(2.0) == 0.0
        noisy = WitnessSensor(name="W", position=(0.0, 0.0, 0.0), self_noise_psd=_flat_psd)
        assert noisy.self_noise(2.0) == 1.0
        bad = WitnessSensor(name="W", position=(0.0, 0.0, 0.0), self_noise_psd=lambda frequency: -1.0)
        with pytest.raises(ValueError, match="negative"):
            bad.self_noise(2.0)

    def test_rejects_non_finite_timing_offset(self) -> None:
        """A non-finite timing offset is rejected."""
        with pytest.raises(ValueError, match="finite"):
            WitnessSensor(name="W", position=(0.0, 0.0, 0.0), timing_offset=np.inf)


class TestNoiseModels:
    """Diagonal, correlated and explicit noise covariances and their ranks."""

    def _sensors(self, count: int, spacing: float, psd: float = 1.0) -> tuple[WitnessSensor, ...]:
        """Return ``count`` collinear sensors with a constant UNANCHORED self-noise PSD."""
        return tuple(
            WitnessSensor(
                name=f"W{i}",
                position=(i * spacing, 0.0, 0.0),
                orientation=(0.0, 0.0, 1.0),
                self_noise_psd=lambda frequency, value=psd: value,
            )
            for i in range(count)
        )

    def test_diagonal_noise_is_the_self_noise_psd(self) -> None:
        """The diagonal model returns each sensor's own self-noise PSD on the diagonal."""
        sensors = self._sensors(4, 10.0)
        matrix = DiagonalNoise().covariance(2.0, sensors)
        np.testing.assert_allclose(matrix, np.eye(4), atol=0.0)

    def test_correlated_noise_is_rank_one_for_a_long_correlation_length(self) -> None:
        """A correlation length far exceeding the aperture gives a rank-one covariance."""
        sensors = self._sensors(6, 50.0)
        matrix = CorrelatedNoise(correlation_length=1.0e9).covariance(2.0, sensors)
        assert effective_rank(matrix) == 1
        assert participation_ratio(matrix) == pytest.approx(1.0, rel=1e-6)

    def test_correlated_noise_is_diagonal_for_a_short_correlation_length(self) -> None:
        """A correlation length far below the spacing gives an effectively diagonal matrix."""
        sensors = self._sensors(4, 100.0)
        matrix = CorrelatedNoise(correlation_length=1.0e-6).covariance(2.0, sensors)
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-300)

    def test_correlated_noise_is_hermitian_psd(self) -> None:
        """A Gaussian spatial kernel is a valid covariance for any point set."""
        sensors = self._sensors(5, 30.0)
        matrix = CorrelatedNoise(correlation_length=40.0).covariance(2.0, sensors)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12

    def test_explicit_noise_follows_sensor_identity(self) -> None:
        """The declared matrix is selected by sensor name under permutation and masking."""
        names = ("A", "B")
        matrix = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=complex)
        noise = ExplicitNoise(channel_order=names, covariance_callable=lambda frequency: matrix)
        sensors = (
            WitnessSensor(name="A", position=(0.0, 0.0, 0.0)),
            WitnessSensor(name="B", position=(1.0, 0.0, 0.0)),
        )
        np.testing.assert_allclose(noise.covariance(1.0, sensors), matrix, atol=0.0)
        flipped = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=complex)
        np.testing.assert_allclose(noise.covariance(1.0, (sensors[1], sensors[0])), flipped, atol=0.0)
        np.testing.assert_allclose(noise.covariance(1.0, (sensors[0],)), [[2.0]], atol=0.0)
        np.testing.assert_allclose(noise.covariance(1.0, (sensors[1],)), [[3.0]], atol=0.0)

    def test_explicit_noise_validates_shape_hermiticity_and_names(self) -> None:
        """The explicit model rejects a wrong shape, a non-Hermitian matrix, bad names and an empty order."""
        names = ("A", "B", "C")
        sensors = tuple(WitnessSensor(name=name, position=(index * 10.0, 0.0, 0.0)) for index, name in enumerate(names))
        good = ExplicitNoise(channel_order=names, covariance_callable=lambda frequency: np.eye(3, dtype=complex))
        np.testing.assert_allclose(good.covariance(1.0, sensors), np.eye(3), atol=0.0)
        with pytest.raises(ValueError, match="shape"):
            ExplicitNoise(
                channel_order=names, covariance_callable=lambda frequency: np.eye(2, dtype=complex)
            ).covariance(1.0, sensors)
        with pytest.raises(ValueError, match="Hermitian"):
            ExplicitNoise(
                channel_order=("A", "B"),
                covariance_callable=lambda frequency: np.array([[1.0, 1.0j], [1.0j, 1.0]], dtype=complex),
            ).covariance(1.0, sensors[:2])
        with pytest.raises(ValueError, match="does not include"):
            ExplicitNoise(
                channel_order=("A", "B"), covariance_callable=lambda frequency: np.eye(2, dtype=complex)
            ).covariance(1.0, (sensors[2],))
        with pytest.raises(ValueError, match="non-empty"):
            ExplicitNoise(channel_order=(), covariance_callable=lambda frequency: np.eye(0, dtype=complex))
        with pytest.raises(ValueError, match="unique"):
            ExplicitNoise(channel_order=("A", "A"), covariance_callable=lambda frequency: np.eye(2, dtype=complex))

    def test_effective_rank_boundaries(self) -> None:
        """A zero matrix has rank zero, the identity has full rank, a rank-one matrix rank one."""
        assert effective_rank(np.zeros((4, 4))) == 0
        assert effective_rank(np.eye(4)) == 4
        assert effective_rank(np.ones((4, 4))) == 1
        assert participation_ratio(np.zeros((4, 4))) == 0.0
        assert participation_ratio(2.0 * np.eye(4)) == pytest.approx(4.0)


class TestNestedDiskArray:
    """Nesting, fixed aperture and fixed per-sensor quality across N."""

    def test_positions_are_nested(self) -> None:
        """The N-sensor array is exactly the first N positions of the master sequence."""
        master = np.array(nested_disk_array(12, aperture=800.0, max_sensors=12).positions())
        for count in (1, 2, 5, 12):
            member = np.array(nested_disk_array(count, aperture=800.0, max_sensors=12).positions())
            np.testing.assert_array_equal(member, master[:count])

    def test_every_sensor_lies_within_the_fixed_aperture(self) -> None:
        """Every position is strictly inside the aperture for every N."""
        for count in (1, 4, 12):
            array = nested_disk_array(count, aperture=800.0, max_sensors=12)
            radii = np.linalg.norm(np.array(array.positions())[:, :2], axis=1)
            assert np.all(radii < 800.0)
            assert array.aperture == 800.0

    def test_per_sensor_quality_is_fixed_across_n(self) -> None:
        """Quantity, response, calibration, self-noise, timing and axis rule do not change with N."""
        small = nested_disk_array(3, aperture=500.0, max_sensors=10, calibration=0.7 - 0.2j, timing_offset=1e-4)
        large = nested_disk_array(10, aperture=500.0, max_sensors=10, calibration=0.7 - 0.2j, timing_offset=1e-4)
        for index in range(3):
            first, second = small.sensors[index], large.sensors[index]
            assert first.quantity is second.quantity
            assert first.calibration == second.calibration
            assert first.timing_offset == second.timing_offset
            assert first.transfer(2.0) == second.transfer(2.0)
            np.testing.assert_allclose(first.unit_orientation, second.unit_orientation, atol=0.0)

    def test_radial_orientation_is_horizontal_and_unit(self) -> None:
        """The radial axis rule gives a horizontal unit vector at every sensor."""
        array = nested_disk_array(6, aperture=400.0, max_sensors=6)
        for sensor in array.sensors:
            axis = sensor.unit_orientation
            assert axis[2] == 0.0
            assert np.linalg.norm(axis) == pytest.approx(1.0)

    def test_rejects_invalid_sizes_and_aperture(self) -> None:
        """N must lie in [1, max_sensors] and the aperture must be positive."""
        with pytest.raises(ValueError, match="n_sensors"):
            nested_disk_array(0, aperture=10.0, max_sensors=4)
        with pytest.raises(ValueError, match="max_sensors"):
            nested_disk_array(5, aperture=10.0, max_sensors=4)
        with pytest.raises(ValueError, match="aperture"):
            nested_disk_array(2, aperture=0.0, max_sensors=4)

    def test_sunflower_sequence_is_aperture_bounded(self) -> None:
        """The master sequence never exceeds the aperture, for any length."""
        for count in (1, 7, 50):
            radii = np.linalg.norm(np.array(sunflower_positions(count, 2.0))[:, :2], axis=1)
            assert np.all(radii < 2.0)
        with pytest.raises(ValueError, match="max_sensors"):
            sunflower_positions(0, 2.0)


class TestMissingChannels:
    """A missing-channel mask drops sensors from channels, covariances and manifests."""

    def _array(self) -> WitnessArray:
        """Return a four-sensor nested array with equal UNANCHORED self-noise."""
        return nested_disk_array(4, aperture=300.0, max_sensors=4, self_noise_psd=_flat_psd)

    def test_missing_sensors_are_dropped_from_channels(self) -> None:
        """The active channel list excludes the missing names."""
        array = self._array().with_missing(("W1", "W3"))
        assert array.channel_names == ("W0", "W2")
        assert array.raw_sensor_count == 4
        assert len(array.active_sensors) == 2

    def test_all_missing_gives_an_empty_array(self) -> None:
        """A fully masked array has no channels and a 0x0 noise covariance."""
        array = self._array().with_missing(("W0", "W1", "W2", "W3"))
        assert array.channel_names == ()
        assert array.active_sensors == ()
        assert array.noise_covariance(2.0).shape == (0, 0)

    def test_rejects_unknown_or_duplicate_missing_names(self) -> None:
        """A missing name must exist and appear once."""
        with pytest.raises(ValueError, match="not in the array"):
            self._array().with_missing(("W9",))
        with pytest.raises(ValueError, match="unique"):
            self._array().with_missing(("W0", "W0"))


class TestArrayManifest:
    """The manifest records geometry, calibration, quantity, timing and rank separately."""

    def _array(self, noise=None) -> WitnessArray:
        """Return a nested array with distinct UNANCHORED calibration and timing."""
        return nested_disk_array(
            4,
            aperture=250.0,
            max_sensors=4,
            self_noise_psd=_flat_psd,
            calibration=1.5 - 0.25j,
            timing_offset=2e-3,
            response=constant_response(2.0 + 1.0j),
            noise=noise,
        )

    def test_manifest_records_geometry_and_calibration(self) -> None:
        """Each sensor record carries position, axis, quantity, calibration, response and self-noise."""
        manifest = self._array().manifest(reference_frequency=2.0)
        record = manifest.sensors[0]
        assert record.name == "W0"
        assert record.position == self._array().sensors[0].position
        assert record.quantity is SensorQuantity.DISPLACEMENT
        assert record.calibration == pytest.approx(1.5 - 0.25j)
        assert record.timing_offset == 2e-3
        assert record.response_at_reference == pytest.approx(2.0 + 1.0j)
        assert record.self_noise_psd_at_reference == 1.0
        assert record.present is True
        assert "constant_response" in record.response_name

    def test_manifest_is_json_serializable(self) -> None:
        """The manifest converts to a JSON-serializable dictionary with complex pairs."""
        payload = self._array().manifest(reference_frequency=2.0).to_dict()
        text = json.dumps(payload)
        assert json.loads(text)["raw_sensor_count"] == 4
        assert payload["sensors"][0]["calibration"] == [1.5, -0.25]

    def test_raw_count_and_effective_rank_are_reported_separately(self) -> None:
        """A strongly correlated noise model has raw count N but effective rank one."""
        correlated = self._array(noise=CorrelatedNoise(correlation_length=1.0e9)).manifest(reference_frequency=2.0)
        independent = self._array().manifest(reference_frequency=2.0)
        assert correlated.raw_sensor_count == 4
        assert correlated.effective_rank == 1
        assert independent.effective_rank == 4
        assert correlated.participation_ratio == pytest.approx(1.0, rel=1e-6)
        assert independent.participation_ratio == pytest.approx(4.0)

    def test_manifest_effective_rank_shrinks_with_the_mask(self) -> None:
        """Missing channels reduce the active channel count and the rank evaluated on them."""
        manifest = self._array().with_missing(("W3",)).manifest(reference_frequency=2.0)
        assert manifest.active_sensor_count == 3
        assert manifest.raw_channel_count == 3
        assert manifest.effective_rank == 3
        assert [record.present for record in manifest.sensors] == [True, True, True, False]


class TestArrayValidation:
    """Structural validation of the array container."""

    def test_rejects_duplicate_sensor_names(self) -> None:
        """Sensor names must be unique."""
        sensors = (
            WitnessSensor(name="W", position=(0.0, 0.0, 0.0)),
            WitnessSensor(name="W", position=(1.0, 0.0, 0.0)),
        )
        with pytest.raises(ValueError, match="unique"):
            WitnessArray(sensors=sensors)

    def test_rejects_non_positive_aperture(self) -> None:
        """A declared aperture must be positive."""
        with pytest.raises(ValueError, match="aperture"):
            WitnessArray(sensors=(WitnessSensor(name="W", position=(0.0, 0.0, 0.0)),), aperture=-1.0)

    def test_permuted_reorders_sensors_and_preserves_metadata(self) -> None:
        """Permuting the sensors preserves the aperture and nesting rule."""
        array = nested_disk_array(4, aperture=200.0, max_sensors=4)
        permuted = array.permuted((3, 1, 0, 2))
        assert permuted.channel_names == ("W3", "W1", "W0", "W2")
        assert permuted.aperture == array.aperture
        assert permuted.nesting_rule == array.nesting_rule
        with pytest.raises(ValueError, match="permutation"):
            array.permuted((0, 0, 1, 2))

    def test_explicit_noise_moves_with_sensor_identity_through_the_array(self) -> None:
        """An explicit non-diagonal covariance permutes and masks with the array's sensors."""
        names = ("A", "B")
        matrix = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=complex)
        sensors = (
            WitnessSensor(name="A", position=(0.0, 0.0, 0.0)),
            WitnessSensor(name="B", position=(1.0, 0.0, 0.0)),
        )
        noise = ExplicitNoise(channel_order=names, covariance_callable=lambda frequency: matrix)
        array = WitnessArray(sensors=sensors, noise=noise)
        np.testing.assert_allclose(array.noise_covariance(1.0), matrix, atol=0.0)
        np.testing.assert_allclose(
            array.permuted((1, 0)).noise_covariance(1.0),
            np.array([[3.0, 0.5], [0.5, 2.0]], dtype=complex),
            atol=0.0,
        )
        np.testing.assert_allclose(array.with_missing(("B",)).noise_covariance(1.0), [[2.0]], atol=0.0)
        np.testing.assert_allclose(array.with_missing(("A",)).noise_covariance(1.0), [[3.0]], atol=0.0)

    def test_explicit_noise_order_must_match_the_declared_sensors(self) -> None:
        """An explicit order that is not a permutation of the array's sensors is rejected."""
        matrix = np.eye(2, dtype=complex)
        sensors = (
            WitnessSensor(name="A", position=(0.0, 0.0, 0.0)),
            WitnessSensor(name="B", position=(1.0, 0.0, 0.0)),
        )
        with pytest.raises(ValueError, match="permutation"):
            WitnessArray(
                sensors=sensors,
                noise=ExplicitNoise(channel_order=("A", "C"), covariance_callable=lambda frequency: matrix),
            )
