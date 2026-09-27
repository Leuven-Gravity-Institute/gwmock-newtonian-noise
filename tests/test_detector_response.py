"""Tests for the orchestrating detector-response, covariance and null construction.

The production :func:`~gwmock_newtonian.detector.response.strain_covariance`
assembles the one-sided three-channel strain cross-spectral density from the
closed-form Newtonian acceleration kernel. These tests control it with
independent routes and analytic identities that do not re-run the production
assembly:

- the direct real-space acceleration integration
  (:func:`~gwmock_newtonian.detector.response.strain_covariance_direct`) must
  reproduce the closed-form covariance -- the "direct test-mass
  acceleration integration agrees with the covariance operator" control;
- the closed shared-vertex triangle telescopes exactly, so ``u^T s = 0`` for
  *any* per-vertex displacement field; a role-dependent suspension breaks that
  symmetry and produces a deliberately null-visible mode, so both a
  null-invisible and a null-visible Newtonian mode are shown to be constructible;
- a common-mode (rigid) acceleration gives exactly zero strain, and a pure
  gradient gives a nonzero strain that is nonetheless exactly null-invisible;
- the covariance is Hermitian positive-semidefinite by construction, and scales
  as ``G**2`` and linearly in the surface PSD (the dimensional controls).

All numeric fixtures here are UNANCHORED illustrative values chosen only to
exercise the code, not a representative ET parameter set.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian.detector.geometry import TriangleGeometry, equilateral_triangle
from gwmock_newtonian.detector.masses import (
    Suspension,
    TestMassLayout,
    free_mass_response,
    independent_mass_layout,
    shared_vertex_layout,
)
from gwmock_newtonian.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian.detector.response import (
    DetectorResponseConfig,
    DetectorStrainCovariance,
    _plane_wave_kernel,
    mode_strain,
    null_invisible_strain,
    null_visible_strain,
    strain_covariance,
    strain_covariance_direct,
)
from gwmock_newtonian.seismic.field import RayleighWaveFieldConfig, steering_vectors
from gwmock_newtonian.seismic.medium import ElasticHalfSpace
from gwmock_newtonian.seismic.spectrum import power_law_psd, uniform_directional_density

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal medium used only to exercise the response pipeline.
_MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED: illustrative arm length and test-mass depth in metres.
_ARM_LENGTH = 10_000.0
_TEST_MASS_DEPTH = 20.0

#: UNANCHORED: a flat illustrative displacement PSD, not a measured site spectrum.
_FLAT_PSD = power_law_psd(reference_frequency=1.0, reference_psd=1.0, exponent=0.0)

#: Relative tolerance for the closed-form versus direct-quadrature covariance.
#: It is a quadrature-resolution design margin: the direct integration truncates a
#: 2D Green's-function tail whose relative size scales as one decay length over
#: the integration radius, so ``radial_max = 40 / k`` leaves a several-percent
#: domain-truncation budget before the oscillatory average reduces it. The value
#: is fixed here from that scaling argument, not read off a measurement.
_DIRECT_TOLERANCE = 0.02

#: Number of horizontal decay lengths ``1 / k`` the direct integration reaches.
_DIRECT_RADIAL_DECAY_LENGTHS = 40.0


def _field(**overrides) -> RayleighWaveFieldConfig:
    """Return an isotropic field fixture on a generic crustal half-space."""
    kwargs = {
        "medium": _MEDIUM,
        "surface_displacement_psd": _FLAT_PSD,
        "directional_density": uniform_directional_density,
        "n_azimuth": 32,
    }
    kwargs.update(overrides)
    return RayleighWaveFieldConfig(**kwargs)


def _config(layout: TestMassLayout, **overrides) -> DetectorResponseConfig:
    """Return a detector-response configuration wrapping ``layout`` and the fixture field."""
    kwargs = {"field": _field(), "layout": layout}
    kwargs.update(overrides)
    return DetectorResponseConfig(**kwargs)


def _geometry() -> TriangleGeometry:
    """Return the canonical equilateral triangle at the fixture arm length."""
    return equilateral_triangle(_ARM_LENGTH)


def _constant_kernel(positions, mode) -> np.ndarray:
    """Return one identical acceleration at every position (a rigid common mode)."""
    del mode
    common = np.array([1.0, -0.5, 0.25], dtype=complex)
    return np.tile(common, (len(positions), 1))


def _gradient_kernel(positions, mode) -> np.ndarray:
    """Return a linear (pure-gradient) acceleration field ``G @ r``."""
    del mode
    gradient = np.array([[1.0e-3, 2.0e-4, 0.0], [2.0e-4, -7.0e-4, 1.0e-4], [0.0, 1.0e-4, 3.0e-4]])
    return np.array([gradient @ np.asarray(position, dtype=float) for position in positions], dtype=complex)


def _curvature_kernel(positions, mode) -> np.ndarray:
    """Return a translation-free, curved acceleration field (deliberately null-visible)."""
    del mode
    return np.array(
        [[position[0] ** 2, position[0] * position[1], position[1] ** 2] for position in positions],
        dtype=complex,
    )


class TestResponseConfig:
    """Configuration and per-mode helpers validate their inputs and use the field's convention."""

    def test_rejects_disabling_both_newtonian_terms(self) -> None:
        """A response with neither the bulk nor the areal term is empty and rejected."""
        with pytest.raises(ValueError, match="at least one"):
            _config(shared_vertex_layout(_geometry()), include_volume=False, include_areal=False)

    def test_mode_wavenumber_uses_the_field_phase_velocity(self) -> None:
        """The mode's wavenumber is ``2 pi f / c`` with ``c`` read from the field."""
        frequency, azimuth = 3.0, 0.4
        config = _config(shared_vertex_layout(_geometry()))
        mode = config.mode(frequency, azimuth)
        phase_velocity = float(config.field.phase_velocity_at(np.array([frequency]))[0])
        assert mode.frequency == frequency
        assert mode.azimuth == azimuth
        assert mode.phase_velocity == pytest.approx(phase_velocity)
        assert mode.wavenumber == pytest.approx(2.0 * np.pi * frequency / phase_velocity)

    def test_density_profile_matches_the_field_steering_primitive(self) -> None:
        """``density_profile`` reproduces the field's bulk density steering at the origin.

        The expected profile is recomputed from the field's own primitive
        :func:`~gwmock_newtonian.seismic.field.steering_vectors` at ``(0, 0, z)``
        (where the plane-wave envelope is unity), not by calling the production
        detector helper a second time, so a sign, scale or offset change in that
        helper can fail this test.
        """
        config = _config(shared_vertex_layout(_geometry()))
        frequency = 2.0
        depths = np.array([0.0, 5.0, 25.0])
        expected = np.real(
            steering_vectors([(0.0, 0.0, float(depth)) for depth in depths], frequency, config.field)["density"][0, :]
        )
        np.testing.assert_allclose(config.density_profile(frequency)(depths), expected, rtol=1e-12)


class TestModeStrain:
    """Per-mode strain is translation-free, shares masses by position and separates null content."""

    def test_rejects_non_positive_frequency(self) -> None:
        """A non-positive frequency is unphysical."""
        config = _config(shared_vertex_layout(_geometry()))
        with pytest.raises(ValueError, match="frequency"):
            mode_strain(config, 0.0, 0.0, _constant_kernel)

    def test_common_mode_acceleration_gives_zero_strain(self) -> None:
        """A rigid common-mode acceleration produces no differential-arm strain."""
        config = _config(shared_vertex_layout(_geometry()))
        strain = mode_strain(config, 2.0, 0.7, _constant_kernel)
        assert strain.shape == (3,)
        np.testing.assert_allclose(strain, 0.0, atol=1e-30)

    def test_shared_vertex_response_is_exactly_null_invisible(self) -> None:
        """``u^T s = 0`` for a curved field on the closed shared-vertex triangle.

        The cyclic arm assignment telescopes exactly, so this holds to floating
        point for any per-vertex field -- the deliberately null-invisible mode.
        """
        geometry = _geometry()
        config = _config(shared_vertex_layout(geometry))
        strain = mode_strain(config, 2.0, 0.0, _curvature_kernel)
        assert np.linalg.norm(strain) > 0.0
        assert abs(geometry.null_vector() @ strain) < 1e-15 * np.linalg.norm(strain)

    def test_pure_gradient_is_null_invisible_but_nonzero(self) -> None:
        """A linear acceleration field gives nonzero strain with exactly zero null projection."""
        geometry = _geometry()
        config = _config(shared_vertex_layout(geometry))
        strain = mode_strain(config, 2.0, 1.1, _gradient_kernel)
        assert np.linalg.norm(strain) > 0.0
        assert abs(geometry.null_vector() @ strain) < 1e-14 * np.linalg.norm(strain)

    def test_role_dependent_suspension_makes_the_mode_null_visible(self) -> None:
        """A role-dependent suspension breaks the telescoping and puts content on the null."""
        geometry = _geometry()

        def end_response(omega: float) -> complex:
            """Return a suspension response that differs from the free-mass default."""
            return free_mass_response(omega) * (1.0 + 0.5j)

        layout = independent_mass_layout(
            geometry,
            test_mass_depth=_TEST_MASS_DEPTH,
            suspensions_by_role={
                "positive_end": Suspension(response=end_response, name="ends"),
                "negative_end": Suspension(response=end_response, name="ends"),
            },
        )
        strain = mode_strain(_config(layout), 2.0, 0.0, _curvature_kernel)
        assert abs(geometry.null_vector() @ strain) > 0.0

    def test_acceleration_is_evaluated_once_per_unique_position(self) -> None:
        """Co-located masses reuse one evaluation, so a shared mass is never double-generated."""
        geometry = _geometry()
        layout = independent_mass_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH)
        # Nine physical masses, but only the three triangle vertices are distinct.
        assert len(layout.masses) == 9
        assert len(layout.unique_positions()) == 3
        calls: list[list[tuple[float, float, float]]] = []

        def kernel(positions, mode) -> np.ndarray:
            """Record the queried positions and return a fixed nonzero acceleration."""
            del mode
            calls.append([tuple(float(value) for value in position) for position in positions])
            return np.tile(np.array([1.0, 2.0, 3.0], dtype=complex), (len(positions), 1))

        mode_strain(_config(layout), 2.0, 0.0, kernel)
        assert len(calls) == 1
        assert len(calls[0]) == len(layout.unique_positions())

    def test_shared_and_independent_layouts_give_identical_covariance(self) -> None:
        """A shared-vertex layout and a co-located independent layout are physically identical."""
        geometry = _geometry()
        frequencies = np.array([0.5, 2.0])
        shared = strain_covariance(
            frequencies, _config(shared_vertex_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        )
        independent = strain_covariance(
            frequencies, _config(independent_mass_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        )
        np.testing.assert_allclose(shared.matrix, independent.matrix, atol=0.0)


class TestRelabelling:
    """Relabelling the detectors permutes the Newtonian response by the same permutation."""

    def test_relabelled_detectors_permute_the_covariance(self) -> None:
        """A cyclic relabelling permutes the covariance rows and columns by the same order.

        The GW-only geometry control checks tensors and ``gw_response``; this closes
        the same acceptance criterion for the Newtonian strain covariance, which is
        assembled through the physical acceleration kernel.
        """
        geometry = _geometry()
        order = (2, 0, 1)
        relabelled = TriangleGeometry(
            vertices=geometry.vertices,
            arm_length=geometry.arm_length,
            detectors=tuple(geometry.detectors[index] for index in order),
        )
        frequencies = np.array([0.5, 2.0])
        covariance = strain_covariance(
            frequencies, _config(shared_vertex_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        )
        permuted = strain_covariance(
            frequencies, _config(shared_vertex_layout(relabelled, test_mass_depth=_TEST_MASS_DEPTH))
        )
        assert permuted.channel_order == tuple(covariance.channel_order[index] for index in order)
        permutation = np.asarray(order)
        np.testing.assert_allclose(
            permuted.matrix,
            covariance.matrix[:, permutation[:, None], permutation[None, :]],
            atol=0.0,
        )

    def test_relabelled_detectors_permute_the_per_mode_strain(self) -> None:
        """The complex per-mode Newtonian strain permutes exactly under the same relabel.

        The covariance is quadratic in the response and can hide a sign or phase
        discrepancy, so the strain vector itself is asserted directly (channel
        ordering included) at several azimuths.
        """
        geometry = _geometry()
        order = (2, 0, 1)
        relabelled = TriangleGeometry(
            vertices=geometry.vertices,
            arm_length=geometry.arm_length,
            detectors=tuple(geometry.detectors[index] for index in order),
        )
        config = _config(shared_vertex_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        relabelled_config = _config(shared_vertex_layout(relabelled, test_mass_depth=_TEST_MASS_DEPTH))
        kernel = _plane_wave_kernel(config, 2.0)
        permutation = np.asarray(order)
        for azimuth in (0.0, 0.3, 1.7):
            strain = mode_strain(config, 2.0, azimuth, kernel)
            permuted = mode_strain(relabelled_config, 2.0, azimuth, kernel)
            np.testing.assert_allclose(permuted, strain[permutation], atol=0.0)


class TestNullConstruction:
    """The deliberately null-visible and null-invisible strain construction helpers."""

    def test_null_visible_strain_is_pure_null_content(self) -> None:
        """``null_visible_strain`` lies along ``u``: nonzero null, zero signal projection."""
        geometry = _geometry()
        strain = null_visible_strain(geometry, amplitude=2.0)
        assert geometry.null_vector() @ strain == pytest.approx(2.0)
        np.testing.assert_allclose(geometry.signal_projector() @ strain, 0.0, atol=1e-15)

    def test_null_invisible_strain_is_orthogonal_to_the_null(self) -> None:
        """``null_invisible_strain`` lies in ``ker(u^T)`` and preserves the coefficient norm."""
        geometry = _geometry()
        strain = null_invisible_strain(geometry, (1.0, 3.0))
        assert geometry.null_vector() @ strain == pytest.approx(0.0, abs=1e-14)
        assert np.linalg.norm(strain) == pytest.approx(np.sqrt(10.0))
        np.testing.assert_allclose(geometry.signal_projector() @ strain, strain, atol=1e-15)

    def test_null_invisible_strain_rejects_wrong_coefficient_count(self) -> None:
        """The signal subspace is two-dimensional, so a length-3 coefficient vector is rejected."""
        with pytest.raises(ValueError, match="coefficients"):
            null_invisible_strain(_geometry(), (1.0, 2.0, 3.0))


class TestCovarianceProperties:
    """Shape, Hermitian-PSD structure and the dimensional scalings of the covariance."""

    def test_covariance_shape_and_channel_labels(self) -> None:
        """The covariance is ``(n_freq, 3, 3)`` and carries the geometry's channel names."""
        geometry = _geometry()
        frequencies = np.array([0.5, 2.0, 5.0])
        covariance = strain_covariance(frequencies, _config(shared_vertex_layout(geometry)))
        assert covariance.matrix.shape == (3, 3, 3)
        assert covariance.channel_order == geometry.names
        assert covariance.frequencies.shape == (3,)

    def test_covariance_is_hermitian_positive_semidefinite(self) -> None:
        """Every frequency slice is Hermitian PSD by construction, with no eigenvalue clipping."""
        geometry = _geometry()
        covariance = strain_covariance(np.array([0.5, 2.0, 5.0]), _config(shared_vertex_layout(geometry)))
        for slice_matrix in covariance.matrix:
            np.testing.assert_allclose(slice_matrix, slice_matrix.conj().T, atol=0.0)
            eigenvalues = np.linalg.eigvalsh(slice_matrix)
            scale = float(np.max(np.abs(eigenvalues)))
            assert np.min(eigenvalues) >= -1e-10 * scale

    def test_zero_psd_field_gives_exactly_zero_covariance(self) -> None:
        """A zero surface PSD produces no correlation at all, not merely a small one."""
        geometry = _geometry()
        config = _config(shared_vertex_layout(geometry), field=_field(surface_displacement_psd=np.zeros_like))
        covariance = strain_covariance(np.array([0.5, 2.0]), config)
        assert np.max(np.abs(covariance.matrix)) == 0.0

    def test_non_positive_frequencies_are_exactly_zero(self) -> None:
        """DC and negative frequencies carry no propagating Rayleigh-wave contribution."""
        geometry = _geometry()
        covariance = strain_covariance(np.array([-1.0, 0.0]), _config(shared_vertex_layout(geometry)))
        assert np.max(np.abs(covariance.matrix)) == 0.0

    def test_covariance_scales_as_gravitational_constant_squared(self) -> None:
        """The strain is linear in ``G``, so the covariance scales as ``G**2`` (dimensional control)."""
        geometry = _geometry()
        frequencies = np.array([2.0])
        base = strain_covariance(frequencies, _config(shared_vertex_layout(geometry)))
        doubled = strain_covariance(
            frequencies,
            _config(shared_vertex_layout(geometry), gravitational_constant=2.0 * GRAVITATIONAL_CONSTANT),
        )
        np.testing.assert_allclose(doubled.matrix, 4.0 * base.matrix, rtol=1e-10)

    def test_covariance_scales_linearly_with_surface_psd(self) -> None:
        """The covariance is linear in the one-sided surface PSD (its normalisation/unit control)."""
        geometry = _geometry()
        frequencies = np.array([2.0])
        base = strain_covariance(frequencies, _config(shared_vertex_layout(geometry)))
        tripled = strain_covariance(
            frequencies,
            _config(
                shared_vertex_layout(geometry),
                field=_field(surface_displacement_psd=power_law_psd(1.0, 3.0, 0.0)),
            ),
        )
        np.testing.assert_allclose(tripled.matrix, 3.0 * base.matrix, rtol=1e-10)

    def test_volume_and_areal_terms_both_contribute(self) -> None:
        """The bulk and free-surface terms give distinct covariances, and both differ from the sum."""
        geometry = _geometry()
        layout = shared_vertex_layout(geometry)
        frequencies = np.array([2.0])
        both = strain_covariance(frequencies, _config(layout))
        volume_only = strain_covariance(frequencies, _config(layout, include_areal=False))
        areal_only = strain_covariance(frequencies, _config(layout, include_volume=False))
        for first, second in ((both, volume_only), (both, areal_only), (volume_only, areal_only)):
            difference = np.linalg.norm(first.matrix - second.matrix)
            scale = np.linalg.norm(second.matrix)
            assert difference > 1e-3 * scale, "the two Newtonian terms gave indistinguishable covariances"


class TestNullProjectionOfCovariance:
    """The null is only ever a labelled projection of the three channels."""

    def test_projections_are_shapes_of_the_three_channel_covariance(self) -> None:
        """``null_projection`` returns one scalar per frequency; the signal projector stays 3x3."""
        geometry = _geometry()
        covariance = strain_covariance(np.array([0.5, 2.0]), _config(shared_vertex_layout(geometry)))
        assert covariance.null_projection().shape == (2,)
        assert covariance.signal_projection().shape == (2, 3, 3)
        assert covariance.visible_fraction().shape == (2,)

    def test_trace_decomposes_into_null_and_signal_projections(self) -> None:
        """``tr(C) = u^T C u + tr(P_g C P_g)`` exactly, for a fabricated and a physical covariance."""
        geometry = _geometry()
        fabricated = _fabricated_covariance(
            geometry, 2.0 * np.outer(geometry.signal_basis()[:, 0], geometry.signal_basis()[:, 0])
        )
        physical = strain_covariance(np.array([2.0]), _config(shared_vertex_layout(geometry)))
        for covariance in (fabricated, physical):
            trace = np.einsum("fii->f", covariance.matrix)
            signal_trace = np.einsum("fii->f", covariance.signal_projection())
            np.testing.assert_allclose(trace.real, covariance.null_projection().real + signal_trace.real, atol=1e-30)
            np.testing.assert_allclose(trace.imag, 0.0, atol=1e-30)

    def test_visible_fraction_is_one_for_a_pure_null_covariance(self) -> None:
        """A covariance built purely from the null direction has all its trace on the null."""
        geometry = _geometry()
        null_only = _fabricated_covariance(geometry, np.outer(geometry.null_vector(), geometry.null_vector()))
        assert null_only.visible_fraction()[0] == pytest.approx(1.0)
        assert abs(complex(null_only.null_projection()[0])) == pytest.approx(1.0)
        np.testing.assert_allclose(null_only.signal_projection()[0], 0.0, atol=1e-15)

    def test_visible_fraction_is_zero_for_a_pure_signal_covariance(self) -> None:
        """A covariance lying wholly in the signal subspace has no null content."""
        geometry = _geometry()
        signal_only = _fabricated_covariance(geometry, geometry.signal_projector())
        assert signal_only.visible_fraction()[0] == pytest.approx(0.0)
        assert abs(complex(signal_only.null_projection()[0])) < 1e-15

    def test_shared_vertex_field_is_null_invisible_and_role_response_is_null_visible(self) -> None:
        """The production covariance is null-invisible for shared vertices but not for split roles."""
        geometry = _geometry()
        frequencies = np.array([0.5, 2.0])
        shared = strain_covariance(
            frequencies, _config(shared_vertex_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        )
        trace = np.einsum("fii->f", shared.matrix).real
        assert np.all(np.abs(shared.null_projection()) < 1e-12 * trace)

        def end_response(omega: float) -> complex:
            """Return a suspension response differing from the free-mass default."""
            return free_mass_response(omega) * (1.0 + 0.5j)

        split = independent_mass_layout(
            geometry,
            test_mass_depth=_TEST_MASS_DEPTH,
            suspensions_by_role={
                "positive_end": Suspension(response=end_response, name="ends"),
                "negative_end": Suspension(response=end_response, name="ends"),
            },
        )
        visible = strain_covariance(frequencies, _config(split))
        # Measured for this fixture: 0.01643 at 0.5 Hz and 0.02076 at 2 Hz, inside
        # the stated ~1.6-2.2% null-visible band. Guard that band, not a floor an
        # order of magnitude below it.
        visible_fraction = visible.visible_fraction()
        assert np.all((visible_fraction > 0.015) & (visible_fraction < 0.022))

    def test_role_offsets_construct_a_null_visible_response(self) -> None:
        """The documented nonzero-offset layout puts measurable content on the null.

        With one mass per vertex the closed triangle is exactly null-invisible
        (null fraction ~1e-17); a documented per-role offset breaks that symmetry
        and lifts the null-visible fraction far above the machine floor, so the
        offset path the docstring advertises is actually reachable.
        """
        geometry = _geometry()
        frequencies = np.array([0.5, 2.0])
        baseline = strain_covariance(
            frequencies, _config(independent_mass_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        )
        offset = strain_covariance(
            frequencies,
            _config(
                independent_mass_layout(
                    geometry, test_mass_depth=_TEST_MASS_DEPTH, offsets={"positive_end": (0.1, 0.0, 0.0)}
                )
            ),
        )
        assert np.all(baseline.visible_fraction() < 1e-15)
        assert np.all(offset.visible_fraction() > 1e-12)


class TestDirectVersusClosedFormCovariance:
    """The direct real-space acceleration integration reproduces the covariance operator."""

    def test_direct_acceleration_integration_agrees_with_closed_form(self) -> None:
        """A direct Green's-function source sum gives the same covariance as the closed-form kernel."""
        geometry = _geometry()
        config = _config(shared_vertex_layout(geometry, test_mass_depth=_TEST_MASS_DEPTH))
        frequencies = np.array([2.0])
        closed = strain_covariance(frequencies, config)
        phase_velocity = float(config.field.phase_velocity_at(frequencies)[0])
        wavenumber = 2.0 * np.pi * frequencies[0] / phase_velocity
        direct = strain_covariance_direct(
            frequencies,
            config,
            radial_max=_DIRECT_RADIAL_DECAY_LENGTHS / wavenumber,
            n_radial=300,
            n_azimuth=64,
            n_depth=160,
        )
        relative = float(np.linalg.norm(closed.matrix - direct.matrix) / np.linalg.norm(closed.matrix))
        assert relative < _DIRECT_TOLERANCE, f"closed-form/direct covariance relative error {relative:.4e}"


class TestSurfaceAmplitudeReadback:
    """The field's surface-vertical-amplitude read-back is wired into both kernels."""

    def test_covariance_threads_the_field_surface_amplitude_readback(self, monkeypatch) -> None:
        """Changing the read-back scales the areal-only covariance by its square.

        The documented read-back is what would catch a change in the field's surface
        normalisation, so it must actually feed both the closed-form and the direct
        kernels; with the bulk term off, doubling it must quadruple the covariance.
        """
        config = _config(shared_vertex_layout(_geometry(), test_mass_depth=_TEST_MASS_DEPTH), include_volume=False)
        frequencies = np.array([2.0])
        phase_velocity = float(config.field.phase_velocity_at(frequencies)[0])
        wavenumber = 2.0 * np.pi * frequencies[0] / phase_velocity
        direct_kwargs = {
            "radial_max": _DIRECT_RADIAL_DECAY_LENGTHS / wavenumber,
            "n_radial": 60,
            "n_azimuth": 16,
            "n_depth": 40,
        }
        baseline_closed = strain_covariance(frequencies, config)
        baseline_direct = strain_covariance_direct(frequencies, config, **direct_kwargs)

        monkeypatch.setattr(
            "gwmock_newtonian.detector.response.surface_vertical_amplitude",
            lambda frequency, field: 2.0,
        )
        doubled_closed = strain_covariance(frequencies, config)
        doubled_direct = strain_covariance_direct(frequencies, config, **direct_kwargs)
        np.testing.assert_allclose(doubled_closed.matrix, 4.0 * baseline_closed.matrix, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(doubled_direct.matrix, 4.0 * baseline_direct.matrix, rtol=1e-10, atol=0.0)


def _fabricated_covariance(geometry: TriangleGeometry, matrix: np.ndarray) -> DetectorStrainCovariance:
    """Return a one-frequency covariance wrapping a caller-supplied ``(3, 3)`` matrix."""
    return DetectorStrainCovariance(
        frequencies=np.array([1.0]),
        matrix=np.asarray(matrix, dtype=complex)[None, :, :],
        channel_order=geometry.names,
        geometry=geometry,
    )
