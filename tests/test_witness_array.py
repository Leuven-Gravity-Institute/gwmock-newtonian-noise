"""Tests for witness sensors, arrays, noise models, manifests and nested families.

These tests control the sensor model against the field's own steering primitive
and against exact algebra (quantity transfers, timing phases, correlated-noise
ranks), and they pin the nested-array invariants the design requires: the
aperture ceiling and every per-sensor property are fixed across ``N``; in
fixed-aperture mode the realized radius and span are fixed across ``N`` too; and
the manifest reports the raw sensor count and the noise effective rank
separately, and the realized radius next to the aperture ceiling.

All numeric fixtures are UNANCHORED illustrative values chosen only to exercise
the code, not a representative ET sensor deployment.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gwmock_newtonian_noise.witness.array import (
    CorrelatedNoise,
    DiagonalNoise,
    ExplicitNoise,
    NestedArrayMode,
    WitnessArray,
    effective_rank,
    nested_disk_array,
    nested_site_order,
    participation_ratio,
    sunflower_positions,
)
from gwmock_newtonian_noise.witness.sensor import (
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

    def test_velocity_over_displacement_is_minus_i_omega(self) -> None:
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

    @pytest.mark.parametrize(
        ("quantity", "unit"),
        [
            (SensorQuantity.DISPLACEMENT, "m"),
            (SensorQuantity.VELOCITY, "m/s"),
            (SensorQuantity.ACCELERATION, "m/s^2"),
        ],
    )
    def test_default_unit_is_the_si_unit_of_the_quantity(self, quantity: SensorQuantity, unit: str) -> None:
        """Without an explicit output unit, a sensor reports in the SI unit of what it measures."""
        assert WitnessSensor(name="W", position=(0.0, 0.0, 0.0), quantity=quantity).unit == unit

    def test_explicit_output_unit_overrides_the_quantity_unit(self) -> None:
        """A non-dimensionless calibration or response carries its own declared unit."""
        sensor = WitnessSensor(
            name="W", position=(0.0, 0.0, 0.0), quantity=SensorQuantity.VELOCITY, output_unit="counts"
        )
        assert sensor.unit == "counts"
        with pytest.raises(ValueError, match="output_unit"):
            WitnessSensor(name="W", position=(0.0, 0.0, 0.0), output_unit="")

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


_MODES = (NestedArrayMode.FIXED_APERTURE, NestedArrayMode.GROWING_SPAN)


def _pairwise_horizontal_span(positions) -> float:
    """Return the largest horizontal pairwise distance, by an explicit double loop."""
    span = 0.0
    for first in positions:
        for second in positions:
            span = max(span, float(np.hypot(first[0] - second[0], first[1] - second[1])))
    return span


class TestNestedDiskArray:
    """Nesting, fixed aperture ceiling and fixed per-sensor quality across N, in both modes."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_positions_are_nested(self, mode: NestedArrayMode) -> None:
        """The N-sensor array is exactly the first N positions of the N_max member."""
        master = np.array(nested_disk_array(12, aperture=800.0, max_sensors=12, mode=mode).positions())
        for count in (1, 2, 5, 12):
            member = np.array(nested_disk_array(count, aperture=800.0, max_sensors=12, mode=mode).positions())
            np.testing.assert_array_equal(member, master[:count])

    @pytest.mark.parametrize("mode", _MODES)
    def test_positions_are_master_sites_in_the_mode_order(self, mode: NestedArrayMode) -> None:
        """Every member occupies master sites; the full member occupies each site exactly once."""
        sites = np.array(sunflower_positions(12, 800.0))
        order = nested_site_order(12, 800.0, mode)
        assert sorted(order) == list(range(12))
        for count in (1, 4, 12):
            member = np.array(nested_disk_array(count, aperture=800.0, max_sensors=12, mode=mode).positions())
            np.testing.assert_array_equal(member, sites[list(order[:count])])

    @pytest.mark.parametrize("mode", _MODES)
    def test_every_sensor_lies_within_the_fixed_aperture(self, mode: NestedArrayMode) -> None:
        """Every position is strictly inside the aperture ceiling for every N."""
        for count in (1, 4, 12):
            array = nested_disk_array(count, aperture=800.0, max_sensors=12, mode=mode)
            radii = np.linalg.norm(np.array(array.positions())[:, :2], axis=1)
            assert np.all(radii < 800.0)
            assert array.aperture == 800.0

    @pytest.mark.parametrize("mode", _MODES)
    def test_per_sensor_quality_is_fixed_across_n(self, mode: NestedArrayMode) -> None:
        """Quantity, response, calibration, self-noise, timing and axis rule do not change with N."""
        kwargs = {"aperture": 500.0, "max_sensors": 10, "mode": mode, "calibration": 0.7 - 0.2j, "timing_offset": 1e-4}
        small = nested_disk_array(3, **kwargs)
        large = nested_disk_array(10, **kwargs)
        for index in range(3):
            first, second = small.sensors[index], large.sensors[index]
            assert first.name == second.name
            assert first.position == second.position
            assert first.quantity is second.quantity
            assert first.calibration == second.calibration
            assert first.timing_offset == second.timing_offset
            assert first.transfer(2.0) == second.transfer(2.0)
            np.testing.assert_allclose(first.unit_orientation, second.unit_orientation, atol=0.0)

    @pytest.mark.parametrize("mode", _MODES)
    def test_radial_orientation_is_horizontal_and_unit(self, mode: NestedArrayMode) -> None:
        """The radial axis rule gives a horizontal unit vector at every sensor."""
        array = nested_disk_array(6, aperture=400.0, max_sensors=6, mode=mode)
        for sensor in array.sensors:
            axis = sensor.unit_orientation
            assert axis[2] == 0.0
            assert np.linalg.norm(axis) == pytest.approx(1.0)

    def test_rejects_invalid_sizes_and_aperture(self) -> None:
        """N must lie in [1, max_sensors] and the aperture must be positive."""
        mode = NestedArrayMode.FIXED_APERTURE
        with pytest.raises(ValueError, match="n_sensors"):
            nested_disk_array(0, aperture=10.0, max_sensors=4, mode=mode)
        with pytest.raises(ValueError, match="max_sensors"):
            nested_disk_array(5, aperture=10.0, max_sensors=4, mode=mode)
        with pytest.raises(ValueError, match="aperture"):
            nested_disk_array(2, aperture=0.0, max_sensors=4, mode=mode)

    def test_mode_has_no_default_and_must_be_known(self) -> None:
        """An N scan cannot silently get either mode: the mode must be named and valid."""
        with pytest.raises(TypeError, match="mode"):
            nested_disk_array(2, aperture=10.0, max_sensors=4)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="nested"):
            nested_disk_array(2, aperture=10.0, max_sensors=4, mode="nested")
        with pytest.raises(ValueError, match="nested"):
            nested_site_order(4, 10.0, "nested")
        string_mode = nested_disk_array(2, aperture=10.0, max_sensors=4, mode="fixed_aperture")
        enum_mode = nested_disk_array(2, aperture=10.0, max_sensors=4, mode=NestedArrayMode.FIXED_APERTURE)
        assert string_mode.positions() == enum_mode.positions()

    def test_sunflower_sequence_is_aperture_bounded(self) -> None:
        """The master sequence never exceeds the aperture, for any length."""
        for count in (1, 7, 50):
            radii = np.linalg.norm(np.array(sunflower_positions(count, 2.0))[:, :2], axis=1)
            assert np.all(radii < 2.0)
        with pytest.raises(ValueError, match="max_sensors"):
            sunflower_positions(0, 2.0)


class TestFixedApertureMode:
    """In fixed-aperture mode an N scan changes sensor density, never the realized span.

    Stated tolerance: the realized radius is the master sequence's outer radius
    ``aperture * sqrt(1 - 1 / (2 * max_sensors))`` for every N, so its relative
    shortfall from the ceiling is at most ``1 / (2 * max_sensors)``; it is
    compared to that closed form at ``rtol=1e-12`` and is bit-identical across N.
    """

    @pytest.mark.parametrize("max_sensors", [1, 2, 3, 12, 50, 137])
    def test_n_scan_keeps_the_realized_radius_at_the_aperture(self, max_sensors: int) -> None:
        """Every N from 1 to max_sensors has the same realized radius, within the stated tolerance."""
        aperture = 800.0
        expected = aperture * np.sqrt(1.0 - 1.0 / (2.0 * max_sensors))
        radii = [
            nested_disk_array(count, aperture=aperture, max_sensors=max_sensors, mode="fixed_aperture")
            .manifest()
            .realized_radius
            for count in range(1, max_sensors + 1)
        ]
        assert len(set(radii)) == 1
        np.testing.assert_allclose(radii[0], expected, rtol=1e-12, atol=0.0)
        assert 1.0 - radii[0] / aperture <= 1.0 / (2.0 * max_sensors)

    def test_n_scan_values_named_by_the_problem_statement(self) -> None:
        """At aperture 800 m, max_sensors 12: fixed-aperture holds ~783 m where growing span gives ~163/432/783 m."""
        for count, growing in ((1, 163.3), (4, 432.0), (12, 783.2)):
            fixed_manifest = nested_disk_array(count, aperture=800.0, max_sensors=12, mode="fixed_aperture").manifest()
            growing_manifest = nested_disk_array(count, aperture=800.0, max_sensors=12, mode="growing_span").manifest()
            assert fixed_manifest.aperture == growing_manifest.aperture == 800.0
            assert fixed_manifest.realized_radius == pytest.approx(783.2, abs=0.05)
            assert growing_manifest.realized_radius == pytest.approx(growing, abs=0.05)

    @pytest.mark.parametrize("max_sensors", [2, 3, 5, 12, 50, 137, 300])
    def test_n_scan_keeps_the_realized_span_of_the_full_array(self, max_sensors: int) -> None:
        """Every N >= 2 already spans the full master array: the first two sites are its diameter."""
        full = _pairwise_horizontal_span(sunflower_positions(max_sensors, 800.0))
        for count in sorted(n for n in {2, 3, max_sensors // 2, max_sensors} if 2 <= n <= max_sensors):
            array = nested_disk_array(count, aperture=800.0, max_sensors=max_sensors, mode="fixed_aperture")
            np.testing.assert_allclose(array.realized_span(), full, rtol=1e-12, atol=0.0)

    def test_order_is_farthest_point_from_the_outermost_site(self) -> None:
        """Site 0 is outermost; each later site is farthest from the sites already active."""
        center = (30.0, -20.0)
        sites = np.array(sunflower_positions(40, 500.0, center=center))[:, :2]
        order = nested_site_order(40, 500.0, NestedArrayMode.FIXED_APERTURE, center=center)
        radii = np.hypot(sites[:, 0] - center[0], sites[:, 1] - center[1])
        assert radii[order[0]] == np.max(radii)
        for step in range(1, len(order)):
            chosen = sites[list(order[:step])]
            gap = [float(np.min(np.hypot(*(chosen - site).T))) for site in sites]
            remaining = [gap[index] for index in order[step:]]
            assert gap[order[step]] == max(remaining)

    def test_growing_span_mode_is_the_identity_order(self) -> None:
        """Growing-span mode takes the master sequence in its own order, so its radius grows with N."""
        assert nested_site_order(9, 100.0, NestedArrayMode.GROWING_SPAN) == tuple(range(9))
        radii = [
            nested_disk_array(count, aperture=100.0, max_sensors=9, mode="growing_span").realized_radius()
            for count in range(1, 10)
        ]
        assert radii == sorted(radii)
        assert radii[0] < 0.5 * radii[-1]


class TestRealizedGeometry:
    """The manifest's realized radius and span match the stored positions in every mode."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_realized_fields_match_the_recorded_positions(self, mode: NestedArrayMode) -> None:
        """Realized radius and span recomputed from the manifest's own sensor records agree exactly."""
        center = (120.0, -45.0)
        array = nested_disk_array(7, aperture=600.0, max_sensors=11, mode=mode, center=center, depth=250.0)
        manifest = array.with_missing(("W0", "W3")).manifest()
        declared = [record.position for record in manifest.sensors]
        active = [record.position for record in manifest.sensors if record.present]
        assert manifest.mode == mode.value
        assert manifest.center == center
        for positions, radius, span in (
            (declared, manifest.realized_radius, manifest.realized_span),
            (active, manifest.active_realized_radius, manifest.active_realized_span),
        ):
            expected_radius = max(float(np.hypot(p[0] - center[0], p[1] - center[1])) for p in positions)
            assert radius == pytest.approx(expected_radius, rel=1e-14, abs=0.0)
            assert span == pytest.approx(_pairwise_horizontal_span(positions), rel=1e-14, abs=0.0)
        assert manifest.realized_radius <= manifest.aperture

    def test_missing_outermost_sensor_shrinks_only_the_active_radius(self) -> None:
        """Masking the fixed-aperture outer site reduces the active radius, not the declared one."""
        array = nested_disk_array(5, aperture=400.0, max_sensors=5, mode="fixed_aperture").with_missing(("W0",))
        record = array.manifest()
        assert record.active_realized_radius < record.realized_radius
        assert record.realized_radius == pytest.approx(400.0 * np.sqrt(0.9), rel=1e-12, abs=0.0)

    def test_explicit_array_reports_realized_geometry_without_an_aperture(self) -> None:
        """An explicit array has no ceiling but still reports radius from its centre, and span."""
        sensors = (
            WitnessSensor(name="A", position=(3.0, 4.0, 10.0)),
            WitnessSensor(name="B", position=(-3.0, 4.0, 0.0)),
        )
        manifest = WitnessArray(sensors=sensors).manifest()
        assert manifest.aperture is None
        assert manifest.mode == "explicit"
        assert manifest.center == (0.0, 0.0)
        assert manifest.realized_radius == 5.0
        assert manifest.realized_span == 6.0

    def test_empty_and_single_sensor_sets_have_zero_span(self) -> None:
        """No sensors give zero radius and span; one sensor gives zero span."""
        single = WitnessArray(sensors=(WitnessSensor(name="A", position=(3.0, 4.0, 0.0)),))
        assert single.realized_span() == 0.0
        assert single.realized_radius() == 5.0
        empty = single.with_missing(("A",))
        assert empty.realized_radius(active_only=True) == 0.0
        assert empty.realized_span(active_only=True) == 0.0

    def test_rejects_a_malformed_center(self) -> None:
        """The centre must be a finite horizontal pair."""
        sensors = (WitnessSensor(name="A", position=(0.0, 0.0, 0.0)),)
        with pytest.raises(ValueError, match="center"):
            WitnessArray(sensors=sensors, center=(0.0, 0.0, 0.0))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="center"):
            WitnessArray(sensors=sensors, center=(float("nan"), 0.0))


class TestMissingChannels:
    """A missing-channel mask drops sensors from channels, covariances and manifests."""

    def _array(self) -> WitnessArray:
        """Return a four-sensor nested array with equal UNANCHORED self-noise."""
        return nested_disk_array(4, aperture=300.0, max_sensors=4, mode="fixed_aperture", self_noise_psd=_flat_psd)

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
            mode="fixed_aperture",
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
        assert record.unit == "m"

    def test_manifest_records_a_declared_output_unit(self) -> None:
        """A nested array's shared output unit reaches every sensor record and the JSON payload."""
        array = nested_disk_array(2, aperture=250.0, max_sensors=2, mode="fixed_aperture", output_unit="counts")
        manifest = array.manifest(reference_frequency=2.0)
        assert [record.unit for record in manifest.sensors] == ["counts", "counts"]
        assert manifest.to_dict()["sensors"][1]["unit"] == "counts"

    def test_manifest_is_json_serializable(self) -> None:
        """The manifest converts to a JSON-serializable dictionary with complex pairs."""
        payload = self._array().manifest(reference_frequency=2.0).to_dict()
        text = json.dumps(payload)
        assert json.loads(text)["raw_sensor_count"] == 4
        assert payload["sensors"][0]["calibration"] == [1.5, -0.25]
        assert "noise_participation_ratio" in payload
        assert "participation_ratio" not in payload

    def test_raw_count_and_effective_rank_are_reported_separately(self) -> None:
        """A strongly correlated noise model has raw count N but noise effective rank one."""
        correlated = self._array(noise=CorrelatedNoise(correlation_length=1.0e9)).manifest(reference_frequency=2.0)
        independent = self._array().manifest(reference_frequency=2.0)
        assert correlated.raw_sensor_count == 4
        assert correlated.noise_effective_rank == 1
        assert independent.noise_effective_rank == 4
        assert correlated.noise_participation_ratio == pytest.approx(1.0, rel=1e-6)
        assert independent.noise_participation_ratio == pytest.approx(4.0)

    def test_manifest_effective_rank_shrinks_with_the_mask(self) -> None:
        """Missing channels reduce the active channel count and the rank evaluated on them."""
        manifest = self._array().with_missing(("W3",)).manifest(reference_frequency=2.0)
        assert manifest.active_sensor_count == 3
        assert manifest.raw_channel_count == 3
        assert manifest.noise_effective_rank == 3
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
        """Permuting or masking the sensors preserves the aperture, nesting rule, mode and centre."""
        array = nested_disk_array(4, aperture=200.0, max_sensors=4, mode="fixed_aperture", center=(5.0, 6.0))
        permuted = array.permuted((3, 1, 0, 2))
        assert permuted.channel_names == ("W3", "W1", "W0", "W2")
        for copy in (permuted, array.with_missing(("W1",))):
            assert copy.aperture == array.aperture
            assert copy.nesting_rule == array.nesting_rule
            assert copy.mode == array.mode == "fixed_aperture"
            assert copy.center == array.center == (5.0, 6.0)
        assert "fixed_aperture" in array.nesting_rule
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
