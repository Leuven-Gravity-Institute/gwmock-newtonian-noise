"""Analytic covariance API.

Exposes the joint one-sided cross-spectral density in closed form for the
scaffold's placeholder statistical model (a frequency-independent
equicorrelation structure). No Newtonian-noise or seismic physics is modelled
here; the physical model is the joint backend in :mod:`gwmock_newtonian.joint`.
"""

from __future__ import annotations

import numpy as np
from gwmock_noise.simulators.joint_protocol import JointCovariance

#: Per-sample variance every channel carries in the scaffold model, in
#: ``channel_unit**2``.
SAMPLE_VARIANCE = 1.0
#: Number of one-sided frequency points the scaffold reports by default.
DEFAULT_GRID_SIZE = 129


def equicorrelation_matrix(channel_order: list[str], coupling: float) -> np.ndarray:
    """Return the per-sample covariance matrix for the scaffold's placeholder model.

    Every off-diagonal entry is ``coupling`` and every diagonal entry is
    :data:`SAMPLE_VARIANCE`; positive definite for any channel count whenever
    ``0.0 <= coupling < 1.0`` (eigenvalues are ``1 + (n - 1) * coupling`` with
    multiplicity 1 and ``1 - coupling`` with multiplicity ``n - 1``).

    Args:
        channel_order: Channel names; only ``len(channel_order)`` is used.
        coupling: Equicorrelation coefficient, ``0.0 <= coupling < 1.0``.

    Returns:
        An ``(n, n)`` real, symmetric, positive-definite covariance matrix.
    """
    n = len(channel_order)
    matrix = np.full((n, n), coupling, dtype=float)
    np.fill_diagonal(matrix, SAMPLE_VARIANCE)
    return matrix


def analytic_covariance(
    channel_order: list[str],
    coupling: float,
    sampling_frequency: float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> JointCovariance:
    """Return the exact, closed-form one-sided cross-spectral density.

    Every channel is temporally white in this scaffold model, so the returned
    matrix is frequency-independent: ``2 * Sigma / sampling_frequency`` at
    every frequency, where ``Sigma`` is :func:`equicorrelation_matrix`. This is
    an exact analytic value, not a statistical estimate off a realization --
    the same convention the public ``gwmock_noise`` dummy backend uses, so it
    can be tested against a closed-form value rather than a statistical one.

    Args:
        channel_order: Channel names, in the order the returned matrix indexes
            its last two axes.
        coupling: Equicorrelation coefficient, ``0.0 <= coupling < 1.0``.
        sampling_frequency: Sampling frequency in hertz.
        grid_size: Number of one-sided frequency points to report.

    Returns:
        The joint one-sided cross-spectral density as a
        :class:`~gwmock_noise.JointCovariance`.
    """
    sigma = equicorrelation_matrix(channel_order, coupling)
    frequencies = np.fft.rfftfreq(2 * (grid_size - 1), d=1.0 / sampling_frequency)
    psd_matrix = (2.0 / sampling_frequency) * sigma.astype(complex)
    matrix = np.broadcast_to(psd_matrix, (frequencies.size, *psd_matrix.shape)).copy()
    return JointCovariance(frequencies=frequencies, matrix=matrix, channel_order=list(channel_order))


__all__ = ["DEFAULT_GRID_SIZE", "SAMPLE_VARIANCE", "analytic_covariance", "equicorrelation_matrix"]
