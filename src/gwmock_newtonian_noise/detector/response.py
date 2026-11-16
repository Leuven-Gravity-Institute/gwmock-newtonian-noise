"""Detector/test-mass strain response assembled from the Rayleigh field.

This module is the detector subpackage's public surface. It ties together the explicit
triangle geometry (:mod:`~gwmock_newtonian_noise.detector.geometry`), the physical
test-mass and suspension layout (:mod:`~gwmock_newtonian_noise.detector.masses`) and
the Newtonian-noise acceleration of the test masses
(:mod:`~gwmock_newtonian_noise.detector.newtonian`) into the one-sided strain
cross-spectral density of the three ET channels.

The covariance is assembled exactly as the field module assembles its own: each
Rayleigh mode (frequency, azimuth) carries the shared random phase of the
spectral-representation synthesis, and the strain covariance is the nonnegative
azimuth-weighted sum

    C_strain(f) = sum_phi w_phi S_u(f) s(f, phi) s(f, phi)^dagger,

where ``s(f, phi)`` is the complex strain vector *per unit surface vertical
reference amplitude* of that mode, including the suspension response and the
explicit arm contraction. This is the covariance operator: it is linear in the
field's one-sided displacement PSD and never materializes a fourth channel -- the
null is only ever a labelled projection ``u^T`` of the three strains.

The production per-mode response uses the closed-form plane-wave acceleration
kernel; :func:`strain_covariance_direct` recomputes the same covariance from the
direct real-space source integration instead, which the test suite uses as the
independent cross-check of the closed form.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian_noise.detector.geometry import N_DETECTORS, TriangleGeometry
from gwmock_newtonian_noise.detector.masses import TestMassLayout
from gwmock_newtonian_noise.detector.newtonian import (
    GRAVITATIONAL_CONSTANT,
    SURFACE_AMPLITUDE_REFERENCE_METRES,
    RayleighMode,
    acceleration_quadrature,
    density_depth_amplitude,
    plane_wave_acceleration,
    surface_vertical_amplitude,
)
from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig

#: Signature of a per-mode acceleration kernel: ``(positions, mode) -> (n, 3)``.
AccelerationKernel = Callable[[Sequence[Sequence[float]], RayleighMode], NDArray[np.complex128]]


@dataclass(frozen=True)
class DetectorResponseConfig:
    """Caller-supplied configuration for the detector/test-mass strain response.

    Attributes:
        field: The seismic-field configuration (medium, PSD, directionality).
        layout: Physical test masses, their sharing structure and suspension.
        include_volume: Include the bulk mass-conservation Newtonian term.
        include_areal: Include the free-surface areal Newtonian term.
        n_depth: Depth quadrature nodes per side of the test-mass depth kink.
        reference_amplitude: Surface vertical reference amplitude in metres.
        gravitational_constant: Gravitational constant in SI units.
    """

    field: RayleighWaveFieldConfig
    layout: TestMassLayout
    include_volume: bool = True
    include_areal: bool = True
    n_depth: int = 400
    reference_amplitude: float = SURFACE_AMPLITUDE_REFERENCE_METRES
    gravitational_constant: float = GRAVITATIONAL_CONSTANT

    def __post_init__(self) -> None:
        """Reject a configuration in which both Newtonian terms are disabled."""
        if not self.include_volume and not self.include_areal:
            raise ValueError("at least one of include_volume/include_areal must be true.")

    def mode(self, frequency: float, azimuth: float) -> RayleighMode:
        """Return the :class:`RayleighMode` at one frequency and azimuth."""
        phase_velocity = float(self.field.phase_velocity_at(np.array([frequency]))[0])
        wavenumber = 2.0 * np.pi * frequency / phase_velocity
        return RayleighMode(
            frequency=frequency,
            wavenumber=wavenumber,
            azimuth=azimuth,
            phase_velocity=phase_velocity,
        )

    def density_profile(self, frequency: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the bulk density amplitude callable at one frequency."""
        return lambda depths: density_depth_amplitude(depths, frequency, self.field)


def mode_strain(
    config: DetectorResponseConfig,
    frequency: float,
    azimuth: float,
    kernel: AccelerationKernel,
) -> NDArray[np.complex128]:
    """Return the three complex strains of one mode per unit surface amplitude.

    Args:
        config: The detector-response configuration.
        frequency: Frequency in hertz, strictly positive.
        azimuth: Propagation azimuth in radians.
        kernel: Per-mode acceleration kernel (closed form or quadrature).

    Returns:
        A length-3 complex strain vector.
    """
    if frequency <= 0.0:
        raise ValueError(f"frequency must be positive, got {frequency}.")
    mode = config.mode(frequency, azimuth)
    positions = config.layout.unique_positions()
    accelerations = kernel(positions, mode)
    acceleration_by_position = {position: accelerations[index] for index, position in enumerate(positions)}
    acceleration_by_mass = config.layout.broadcast_by_position(acceleration_by_position)
    return config.layout.strain_from_accelerations(acceleration_by_mass, 2.0 * np.pi * frequency)


def _plane_wave_kernel(config: DetectorResponseConfig, frequency: float) -> AccelerationKernel:
    """Return the closed-form plane-wave acceleration kernel for one frequency."""
    profile = config.density_profile(frequency)
    medium_density = config.field.medium.density
    vertical_amplitude = surface_vertical_amplitude(frequency, config.field)

    def kernel(positions: Sequence[Sequence[float]], mode: RayleighMode) -> NDArray[np.complex128]:
        return plane_wave_acceleration(
            positions,
            mode,
            profile,
            medium_density,
            reference_amplitude=config.reference_amplitude,
            surface_vertical_displacement=vertical_amplitude,
            include_volume=config.include_volume,
            include_areal=config.include_areal,
            n_depth=config.n_depth,
            gravitational_constant=config.gravitational_constant,
        )

    return kernel


def _quadrature_kernel(  # noqa: PLR0913 -- one parameter per physically distinct input.
    config: DetectorResponseConfig,
    frequency: float,
    *,
    radial_max: float,
    n_radial: int,
    n_azimuth: int,
    n_depth: int,
    depth_max: float | None,
) -> AccelerationKernel:
    """Return the direct real-space quadrature acceleration kernel for one frequency."""
    profile = config.density_profile(frequency)
    medium_density = config.field.medium.density
    vertical_amplitude = surface_vertical_amplitude(frequency, config.field)

    def kernel(positions: Sequence[Sequence[float]], mode: RayleighMode) -> NDArray[np.complex128]:
        return acceleration_quadrature(
            positions,
            mode,
            profile,
            medium_density,
            radial_max=radial_max,
            n_radial=n_radial,
            n_azimuth=n_azimuth,
            depth_max=depth_max,
            n_depth=n_depth,
            reference_amplitude=config.reference_amplitude,
            surface_vertical_displacement=vertical_amplitude,
            include_volume=config.include_volume,
            include_areal=config.include_areal,
            gravitational_constant=config.gravitational_constant,
        )

    return kernel


def _surface_psd(frequency: float, field: RayleighWaveFieldConfig) -> float:
    """Return the field's one-sided surface vertical displacement PSD at one frequency."""
    value = float(np.asarray(field.surface_displacement_psd(np.array([frequency])), dtype=float)[0])
    if value < 0.0:
        raise ValueError(f"surface_displacement_psd returned a negative value ({value}) at f={frequency}.")
    return value


def _assemble_covariance(
    frequencies: NDArray[np.floating],
    config: DetectorResponseConfig,
    kernel_for: Callable[[float], AccelerationKernel],
) -> NDArray[np.complex128]:
    """Assemble the one-sided strain covariance from whichever kernel is supplied."""
    frequencies = np.asarray(frequencies, dtype=float)
    matrix = np.zeros((frequencies.size, N_DETECTORS, N_DETECTORS), dtype=complex)
    azimuths = config.field.azimuth_grid()
    weights = config.field.azimuth_weights()
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0:
            continue
        surface_psd = _surface_psd(float(frequency), config.field)
        if surface_psd == 0.0:
            continue
        kernel = kernel_for(float(frequency))
        for azimuth, weight in zip(azimuths, weights, strict=True):
            strain = mode_strain(config, float(frequency), float(azimuth), kernel)
            matrix[index] += weight * surface_psd * np.outer(strain, strain.conj())
    return matrix


@dataclass(frozen=True)
class DetectorStrainCovariance:
    """One-sided strain cross-spectral density of the three ET channels.

    Attributes:
        frequencies: One-sided frequency grid in hertz, shape ``(n_freq,)``.
        matrix: Hermitian ``(n_freq, 3, 3)`` one-sided cross-spectral density.
        channel_order: The three channel names, matching ``matrix``'s axes.
        geometry: The triangle geometry, retained so the null is a labelled
            projection of these three channels and never a fourth channel.
    """

    frequencies: NDArray[np.float64]
    matrix: NDArray[np.complex128]
    channel_order: tuple[str, ...]
    geometry: TriangleGeometry

    def null_projection(self) -> NDArray[np.complex128]:
        """Return the null-channel PSD ``u^T C u`` per frequency, shape ``(n_freq,)``."""
        u = self.geometry.null_vector()
        return np.einsum("i,fij,j->f", u, self.matrix, u)

    def signal_projection(self) -> NDArray[np.complex128]:
        """Return ``P_g C P_g`` per frequency, the covariance in the GW-sensitive subspace."""
        projector = self.geometry.signal_projector()
        return np.einsum("ia,fab,jb->fij", projector, self.matrix, projector)

    def visible_fraction(self) -> NDArray[np.float64]:
        """Return the fraction of the total trace that lies along the null direction.

        This is the labelled projector content ``tr(P_n C) / tr(C)``, a scalar per
        frequency; it is a diagnostic of these three channels, not a fourth
        measurement.
        """
        total = np.einsum("fii->f", self.matrix).real
        null = self.null_projection().real
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(total > 0.0, null / total, 0.0)
        return np.asarray(fraction, dtype=float)


def strain_covariance(frequencies: NDArray[np.floating], config: DetectorResponseConfig) -> DetectorStrainCovariance:
    """Return the production one-sided strain covariance using the closed-form kernel.

    Args:
        frequencies: One-sided frequency grid in hertz.
        config: The detector-response configuration.

    Returns:
        The :class:`DetectorStrainCovariance`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    matrix = _assemble_covariance(frequencies, config, lambda f: _plane_wave_kernel(config, f))
    return DetectorStrainCovariance(
        frequencies=frequencies,
        matrix=matrix,
        channel_order=config.layout.geometry.names,
        geometry=config.layout.geometry,
    )


def strain_covariance_direct(  # noqa: PLR0913 -- one parameter per physically distinct input.
    frequencies: NDArray[np.floating],
    config: DetectorResponseConfig,
    *,
    radial_max: float,
    n_radial: int = 200,
    n_azimuth: int = 64,
    n_depth: int = 160,
    depth_max: float | None = None,
) -> DetectorStrainCovariance:
    """Return the strain covariance using the direct real-space acceleration integration.

    This is the independent cross-check of :func:`strain_covariance`: it replaces
    the closed-form per-mode kernel with an explicit Green's-function sum over a
    source quadrature, and is expected to agree with the production route once
    the quadrature resolves the mode.

    Args:
        frequencies: One-sided frequency grid in hertz.
        config: The detector-response configuration.
        radial_max: Horizontal source-integration radius in metres.
        n_radial: Number of radial quadrature nodes.
        n_azimuth: Number of azimuthal quadrature nodes.
        n_depth: Number of depth quadrature nodes.
        depth_max: Optional depth integration bound in metres.

    Returns:
        The :class:`DetectorStrainCovariance`.
    """
    frequencies = np.asarray(frequencies, dtype=float)

    def kernel_for(frequency: float) -> AccelerationKernel:
        return _quadrature_kernel(
            config,
            frequency,
            radial_max=radial_max,
            n_radial=n_radial,
            n_azimuth=n_azimuth,
            n_depth=n_depth,
            depth_max=depth_max,
        )

    matrix = _assemble_covariance(frequencies, config, kernel_for)
    return DetectorStrainCovariance(
        frequencies=frequencies,
        matrix=matrix,
        channel_order=config.layout.geometry.names,
        geometry=config.layout.geometry,
    )


def null_visible_strain(geometry: TriangleGeometry, amplitude: float = 1.0) -> NDArray[np.float64]:
    """Return a deliberately null-visible strain vector ``amplitude * u``."""
    return amplitude * geometry.null_vector()


def null_invisible_strain(geometry: TriangleGeometry, coefficients: Sequence[float]) -> NDArray[np.float64]:
    """Return a deliberately null-invisible strain vector from signal-subspace coefficients.

    Args:
        geometry: The triangle geometry.
        coefficients: Two coefficients in the orthonormal basis of ``ker(u^T)``.

    Returns:
        A length-3 strain vector orthogonal to the null vector.
    """
    basis = geometry.signal_basis()
    coefficients_array = np.asarray(coefficients, dtype=float)
    if coefficients_array.shape != (2,):
        raise ValueError(f"coefficients must have shape (2,), got {coefficients_array.shape}.")
    return basis @ coefficients_array


__all__ = [
    "AccelerationKernel",
    "DetectorResponseConfig",
    "DetectorStrainCovariance",
    "mode_strain",
    "null_invisible_strain",
    "null_visible_strain",
    "strain_covariance",
    "strain_covariance_direct",
]
