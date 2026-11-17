"""Minimal stationary, horizontally homogeneous Rayleigh-wave seismic field.

Builds the field as a continuous superposition of independent fundamental-mode
Rayleigh plane waves, each labeled by frequency ``f`` and horizontal
propagation azimuth ``phi``, following the standard directional-wave-spectrum
construction of a stationary random surface-wave field: see
Longuet-Higgins, M. S. (1957), "The statistical analysis of a random, moving
surface," Philosophical Transactions of the Royal Society A, 249(966),
321-387, for the general construction, and Harms, J. (2019), "Terrestrial
Gravity Fluctuations," Living Reviews in Relativity, 22, 6, Secs. 4-5, for
its specialization to Rayleigh-wave seismic fields used in Newtonian-noise
modeling (directional/depth-resolved cross-spectral density and the
mass-conservation density-perturbation source term this module implements).

Each mode's spatial dependence is fixed by
:mod:`gwmock_newtonian_noise.seismic.depth`'s closed-form depth eigenfunctions; the
per-position, per-component, per-mode complex response returned by
:func:`steering_vectors` is the one physical model both
:func:`analytic_covariance` (an azimuth-quadrature integral of nonnegative-
weighted outer products, Hermitian PSD *by construction*, with no eigenvalue
clipping anywhere in this module) and :func:`realize` (a seeded time-domain
synthesis) are built from.

Realizations use the spectral-representation method of Shinozuka, M. and
Deodatis, G. (1991), "Simulation of Stochastic Processes by Spectral
Representation," Applied Mechanics Reviews, 44(4), 191-204: each strictly
positive discrete frequency bin, crossed with each azimuth quadrature cell,
carries one independent random-phase complex mode amplitude shared by every
requested (position, component) channel -- the deterministic steering
response alone sets the correlation between channels.

Density (:class:`~gwmock_newtonian_noise.seismic.medium.ElasticHalfSpace`),
dispersion (the optional ``phase_velocity`` override), directional anisotropy
(the ``directional_density`` callable) and depth (the ``z`` coordinate of any
requested position) are all caller-supplied configuration -- this module
fixes no "realistic ET" parameter set.

Scope note: this module models only the bulk mass-conservation density
perturbation ``delta_rho = -rho0 * div(u)``. The additional free-surface areal
density perturbation sourced by the moving ground surface (the dominant
Newtonian-noise term for a gravimeter near the surface; see Harms (2019)
Sec. 5.1) is *not* modeled here; it is applied downstream in
:mod:`gwmock_newtonian_noise.detector.newtonian`, which consumes this field and
exposes the term through its ``include_areal`` switch. This module is unchanged
by that downstream implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from gwmock_newtonian_noise.rng import create_generator
from gwmock_newtonian_noise.seismic.depth import (
    RayleighDepthRates,
    depth_rates,
    dilatation_shape,
    horizontal_shape,
    vertical_shape,
)
from gwmock_newtonian_noise.seismic.medium import ElasticHalfSpace

#: A single field-sample location: ``(x, y, z)`` in meters, ``z`` positive downward from the free surface.
Position = tuple[float, float, float]

#: The displacement components :func:`steering_vectors` and :func:`analytic_covariance` support.
DISPLACEMENT_COMPONENTS: tuple[str, ...] = ("x", "y", "z")

#: Fewest samples a realization can produce: at least a start and end sample.
_MIN_REALIZATION_SAMPLES = 2

#: Fewest azimuth quadrature points that can resolve a direction at all.
_MIN_AZIMUTH_POINTS = 4


@dataclass(frozen=True)
class RayleighWaveFieldConfig:
    """Configuration for the stationary, homogeneous Rayleigh-wave field.

    Attributes:
        medium: The homogeneous elastic half-space.
        surface_displacement_psd: One-sided power spectral density of the
            *surface* (``z = 0``) vertical displacement, in meters^2/hertz,
            as a function of a strictly-positive frequency array in hertz.
            Must return non-negative values.
        directional_density: Directional density over propagation azimuth,
            in radians^-1, as a function of an azimuth array in radians. Need
            not integrate to exactly 1 over ``[0, 2*pi)``: it is renormalized
            over :meth:`azimuth_grid` internally.
        phase_velocity: Optional dispersion override, a callable mapping a
            frequency array (hertz) to a phase-velocity array (meters/second,
            each strictly below ``medium.s_wave_speed``). ``None`` (the
            default) uses the non-dispersive homogeneous half-space value,
            ``medium.rayleigh_phase_velocity``, at every frequency.
        n_azimuth: Number of azimuth quadrature points used to discretize the
            ``[0, 2*pi)`` directional integral, must be at least 4.
    """

    medium: ElasticHalfSpace
    surface_displacement_psd: Callable[[np.ndarray], np.ndarray]
    directional_density: Callable[[np.ndarray], np.ndarray]
    phase_velocity: Callable[[np.ndarray], np.ndarray] | None = None
    n_azimuth: int = 64

    def __post_init__(self) -> None:
        """Reject a directional quadrature grid too coarse to resolve azimuth at all."""
        if self.n_azimuth < _MIN_AZIMUTH_POINTS:
            raise ValueError(f"n_azimuth must be at least {_MIN_AZIMUTH_POINTS}, got {self.n_azimuth}.")

    def azimuth_grid(self) -> np.ndarray:
        """Return the ``n_azimuth``-point uniform midpoint quadrature grid over ``[0, 2*pi)``."""
        return (np.arange(self.n_azimuth) + 0.5) * (2.0 * np.pi / self.n_azimuth)

    def azimuth_weights(self) -> np.ndarray:
        """Return normalized directional quadrature weights over :meth:`azimuth_grid`, summing to 1."""
        raw = np.asarray(self.directional_density(self.azimuth_grid()), dtype=float)
        if np.any(raw < 0.0):
            raise ValueError("directional_density must be non-negative everywhere.")
        total = np.sum(raw)
        if total <= 0.0:
            raise ValueError("directional_density must be positive somewhere on the azimuth grid.")
        return raw / total

    def phase_velocity_at(self, frequencies: np.ndarray) -> np.ndarray:
        """Return the phase velocity at each frequency, from the dispersion override or the default."""
        frequencies = np.asarray(frequencies, dtype=float)
        if self.phase_velocity is not None:
            c = np.asarray(self.phase_velocity(frequencies), dtype=float)
            if c.shape != frequencies.shape:
                raise ValueError("phase_velocity callable must return an array matching frequencies.shape.")
            return c
        return np.full_like(frequencies, self.medium.rayleigh_phase_velocity)


def _mode_rates(config: RayleighWaveFieldConfig, frequency: float) -> RayleighDepthRates:
    """Return the depth-decay rates of the mode at one strictly-positive frequency."""
    if frequency <= 0.0:
        raise ValueError(f"frequency must be positive, got {frequency}.")
    c = float(config.phase_velocity_at(np.array([frequency]))[0])
    return depth_rates(config.medium, 2.0 * np.pi * frequency, c)


def steering_vectors(
    positions: Sequence[Position], frequency: float, config: RayleighWaveFieldConfig
) -> dict[str, np.ndarray]:
    """Return each mode's complex per-azimuth, per-position, per-component response.

    Every returned steering value is the (position, component) response to
    one unit of the *surface reference amplitude* ``U`` -- the complex
    coefficient such that a single mode with amplitude ``U`` produces surface
    (``z = 0``) vertical displacement exactly ``U * exp(i*(k*n.r_h - w*t))``.
    Scaling and summing these against random or analytic mode weights (as
    :func:`analytic_covariance` and :func:`realize` do) builds the physical
    field; see the module docstring for the derivation of the underlying
    depth eigenfunctions.

    Args:
        positions: Field-sample locations.
        frequency: Strictly-positive frequency, in hertz.
        config: The field configuration.

    Returns:
        A dict with keys ``"x"``, ``"y"``, ``"z"``, ``"density"``, each an
        ``(config.n_azimuth, len(positions))`` complex array. The displacement
        components carry units of meters per unit ``U`` (dimensionless);
        ``"density"`` carries units of kilograms/meter^3 per unit ``U``
        (meters), i.e. kilograms/meter^4.
    """
    rates = _mode_rates(config, frequency)
    f_v0 = float(vertical_shape(0.0, rates))
    if f_v0 == 0.0:
        raise ValueError(f"Degenerate mode at frequency={frequency}: surface vertical shape f_v(0) is zero.")

    phi = config.azimuth_grid()
    cos_phi = np.cos(phi)[:, None]
    sin_phi = np.sin(phi)[:, None]

    positions_arr = np.asarray(positions, dtype=float)
    x, y, z = positions_arr[:, 0], positions_arr[:, 1], positions_arr[:, 2]

    horizontal_phase = rates.k * (x[None, :] * cos_phi + y[None, :] * sin_phi)
    envelope = np.exp(1j * horizontal_phase)  # (n_azimuth, n_pos)

    f_h = horizontal_shape(z, rates)[None, :]
    f_v = vertical_shape(z, rates)[None, :]
    dilatation = dilatation_shape(z, rates)[None, :]

    along = -1j * f_h / f_v0
    steer_z = f_v / f_v0
    steer_density = (config.medium.density * dilatation) / (rates.k * f_v0)

    return {
        "x": along * cos_phi * envelope,
        "y": along * sin_phi * envelope,
        "z": steer_z * envelope,
        "density": steer_density * envelope,
    }


def _cross_spectral_matrix(
    positions: Sequence[Position], components: Sequence[str], frequency: float, config: RayleighWaveFieldConfig
) -> np.ndarray:
    """Return the Hermitian PSD one-sided cross-spectral matrix at one frequency.

    Channels are ordered position-major: all requested ``components`` of
    ``positions[0]``, then all of ``positions[1]``, etc. Built as a
    nonnegative-weighted azimuth-quadrature sum of outer products
    ``weight * steer @ steer.conj().T``, which is Hermitian PSD by
    construction for any nonnegative weights -- no eigenvalue clipping is
    applied anywhere in this function.
    """
    s_u = float(np.asarray(config.surface_displacement_psd(np.array([frequency])))[0])
    if s_u < 0.0:
        raise ValueError(f"surface_displacement_psd returned a negative value ({s_u}) at frequency={frequency}.")

    steering = steering_vectors(positions, frequency, config)
    weight = config.azimuth_weights()

    n_pos = len(positions)
    stacked = np.empty((n_pos * len(components), config.n_azimuth), dtype=complex)
    for pi in range(n_pos):
        for ci, component in enumerate(components):
            stacked[pi * len(components) + ci] = steering[component][:, pi]

    weighted = stacked * weight[None, :]
    matrix = (weighted @ stacked.conj().T) * s_u
    return matrix


def analytic_covariance(
    positions: Sequence[Position],
    frequencies: np.ndarray,
    config: RayleighWaveFieldConfig,
    components: Sequence[str] = DISPLACEMENT_COMPONENTS,
) -> np.ndarray:
    """Return the analytic one-sided cross-spectral density, shape ``(n_freq, n_channel, n_channel)``.

    ``n_channel = len(positions) * len(components)``, ordered position-major
    (see :func:`_cross_spectral_matrix`). Non-positive frequencies (no
    propagating Rayleigh-wave contribution at DC or negative frequency) get
    an all-zero slice.

    Args:
        positions: Field-sample locations.
        frequencies: Frequency array, in hertz.
        config: The field configuration.
        components: Which per-position components to include, a subset of
            ``("x", "y", "z", "density")``.

    Returns:
        The complex, Hermitian-per-slice cross-spectral density array.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    n_channel = len(positions) * len(components)
    matrix = np.zeros((frequencies.size, n_channel, n_channel), dtype=complex)
    for i, f in enumerate(frequencies):
        if f > 0.0:
            matrix[i] = _cross_spectral_matrix(positions, components, float(f), config)
    return matrix


def analytic_density_covariance(
    positions: Sequence[Position], frequencies: np.ndarray, config: RayleighWaveFieldConfig
) -> np.ndarray:
    """Return the analytic one-sided cross-spectral density of the bulk density perturbation.

    Equivalent to ``analytic_covariance(positions, frequencies, config, components=("density",))``.

    Args:
        positions: Field-sample locations.
        frequencies: Frequency array, in hertz.
        config: The field configuration.

    Returns:
        A ``(len(frequencies), len(positions), len(positions))`` complex array, in
        (kilograms/meter^3)^2/hertz.
    """
    return analytic_covariance(positions, frequencies, config, components=("density",))


def realize(  # noqa: PLR0913, PLR0917 -- one parameter per physically distinct input; splitting would obscure the API.
    positions: Sequence[Position],
    duration: float,
    sampling_frequency: float,
    config: RayleighWaveFieldConfig,
    seed: int | None,
    components: Sequence[str] = DISPLACEMENT_COMPONENTS,
) -> dict[tuple[Position, str], np.ndarray]:
    """Synthesize a seeded, real-valued time-domain realization of the field.

    Uses the spectral-representation method (Shinozuka & Deodatis 1991, see
    the module docstring): at each strictly-positive ``numpy.fft.rfftfreq``
    bin and each azimuth quadrature cell, one independent random-phase
    complex mode amplitude is drawn and combined with the deterministic
    :func:`steering_vectors` response shared by every requested channel.

    Args:
        positions: Field-sample locations.
        duration: Realization duration, in seconds, must be positive.
        sampling_frequency: Sampling frequency, in hertz, must be positive.
        config: The field configuration.
        seed: Integer seed for a reproducible realization, or ``None`` for a
            non-reproducible, OS-entropy-seeded one (see
            :mod:`gwmock_newtonian_noise.rng`).
        components: Which per-position components to synthesize, a subset of
            ``("x", "y", "z", "density")``.

    Returns:
        A dict keyed by ``(position, component)`` mapping to a real
        ``(n_samples,)`` time series, where ``n_samples = round(duration * sampling_frequency)``.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}.")
    if sampling_frequency <= 0.0:
        raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}.")
    n_samples = round(duration * sampling_frequency)
    if n_samples < _MIN_REALIZATION_SAMPLES:
        raise ValueError("duration and sampling_frequency must produce at least two samples.")

    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    d_freq = sampling_frequency / n_samples
    rng = create_generator(seed)
    t = np.arange(n_samples) / sampling_frequency
    weight = config.azimuth_weights()

    signals: dict[tuple[Position, str], np.ndarray] = {
        (position, component): np.zeros(n_samples, dtype=float) for position in positions for component in components
    }

    for f in frequencies:
        if f <= 0.0:
            continue
        s_u = float(np.asarray(config.surface_displacement_psd(np.array([f])))[0])
        if s_u < 0.0:
            raise ValueError(f"surface_displacement_psd returned a negative value ({s_u}) at frequency={f}.")
        if s_u == 0.0:
            continue

        steering = steering_vectors(positions, float(f), config)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=config.n_azimuth)
        mode_amplitude = np.sqrt(2.0 * s_u * weight * d_freq) * np.exp(1j * theta)
        phase = np.exp(-1j * 2.0 * np.pi * f * t)

        for pi, position in enumerate(positions):
            for component in components:
                combined_coefficient = np.sum(mode_amplitude * steering[component][:, pi])
                signals[(position, component)] += np.real(combined_coefficient * phase)

    return signals


__all__ = [
    "DISPLACEMENT_COMPONENTS",
    "Position",
    "RayleighWaveFieldConfig",
    "analytic_covariance",
    "analytic_density_covariance",
    "realize",
    "steering_vectors",
]
