"""Tests for the analytic covariance API."""

from __future__ import annotations

import numpy as np
import pytest
from gwmock_noise.simulators.joint_protocol import JointCovariance

from gwmock_newtonian_noise.covariance import SAMPLE_VARIANCE, analytic_covariance, equicorrelation_matrix

pytestmark = pytest.mark.unit


class TestEquicorrelationMatrix:
    """The per-sample covariance matrix the scaffold model draws from."""

    def test_diagonal_is_sample_variance(self) -> None:
        """Every channel's own variance is the declared sample variance."""
        matrix = equicorrelation_matrix(["A", "B", "C"], coupling=0.2)
        np.testing.assert_array_equal(np.diag(matrix), np.full(3, SAMPLE_VARIANCE))

    def test_off_diagonal_is_coupling(self) -> None:
        """Every distinct channel pair shares the same coupling coefficient."""
        matrix = equicorrelation_matrix(["A", "B", "C"], coupling=0.2)
        off_diagonal = matrix[~np.eye(3, dtype=bool)]
        np.testing.assert_array_equal(off_diagonal, np.full(6, 0.2))

    @pytest.mark.parametrize("coupling", [0.0, 0.1, 0.5, 0.99])
    def test_positive_definite_over_valid_range(self, coupling: float) -> None:
        """The matrix stays positive definite everywhere the config schema allows."""
        matrix = equicorrelation_matrix(["A", "B", "C", "D"], coupling=coupling)
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert np.all(eigenvalues > 0.0)


class TestAnalyticCovariance:
    """The closed-form one-sided cross-spectral density."""

    def test_returns_joint_covariance(self) -> None:
        """The return type matches the public protocol's covariance type exactly."""
        result = analytic_covariance(["A", "B"], coupling=0.3, sampling_frequency=64.0)
        assert isinstance(result, JointCovariance)

    def test_frequency_grid_matches_rfftfreq_convention(self) -> None:
        """The frequency grid follows the protocol's declared numpy.fft.rfftfreq convention."""
        grid_size = 17
        result = analytic_covariance(["A", "B"], coupling=0.3, sampling_frequency=64.0, grid_size=grid_size)
        expected = np.fft.rfftfreq(2 * (grid_size - 1), d=1.0 / 64.0)
        np.testing.assert_array_equal(result.frequencies, expected)
        assert result.frequencies.shape == (grid_size,)

    def test_matrix_is_hermitian_at_every_frequency(self) -> None:
        """Every frequency slice of the cross-spectral density is Hermitian, as the protocol requires."""
        result = analytic_covariance(["A", "B", "C"], coupling=0.3, sampling_frequency=64.0)
        for slice_matrix in result.matrix:
            np.testing.assert_allclose(slice_matrix, slice_matrix.conj().T, atol=0.0)

    def test_exact_closed_form_value(self) -> None:
        """The reported density equals the exact 2*Sigma/fs value, not a fitted or estimated one."""
        channel_order = ["A", "B"]
        coupling = 0.25
        sampling_frequency = 32.0
        result = analytic_covariance(channel_order, coupling, sampling_frequency)

        sigma = equicorrelation_matrix(channel_order, coupling)
        expected_slice = (2.0 / sampling_frequency) * sigma

        for slice_matrix in result.matrix:
            np.testing.assert_allclose(slice_matrix.real, expected_slice, atol=0.0)
            np.testing.assert_allclose(slice_matrix.imag, np.zeros_like(expected_slice), atol=0.0)

    def test_channel_order_preserved(self) -> None:
        """The returned channel_order matches the input order exactly, not a sorted or reordered one."""
        channel_order = ["Z", "A", "M"]
        result = analytic_covariance(channel_order, coupling=0.1, sampling_frequency=16.0)
        assert result.channel_order == channel_order
