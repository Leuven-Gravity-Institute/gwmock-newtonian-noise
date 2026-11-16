"""Newtonian-noise acceleration of the ET test masses from the Rayleigh field.

Gravity perturbations from a seismic field are sourced by mass-density
fluctuations. Two terms contribute:

- the **bulk** term, the mass-conservation density perturbation
  ``delta_rho = -rho0 * div(u)`` already modeled by
  :mod:`gwmock_newtonian_noise.seismic.field`; and
- the **free-surface areal** term, the mass per unit area ``delta_sigma``
  created by the moving ground surface. For a surface vertical displacement
  ``xi_z`` (positive **up**) it is ``delta_sigma = rho0 * xi_z``. This term
  dominates the Newtonian noise of a near-surface test mass and is the term
  :mod:`gwmock_newtonian_noise.seismic.field` deliberately deferred to this
  module.

The governing equations are

    delta_Phi(r0) = -G * integral delta_rho(r) / |r - r0| d^3r,
    delta_a(r0)   = -grad_{r0} delta_Phi(r0),

with the areal term handled identically on the free surface. The primary
sources are Harms, J. (2019), "Terrestrial Gravity Fluctuations," Living
Reviews in Relativity, 22, 6, Sec. 5 (density perturbations and the
surface/areal layer), and Coughlin, M., Mukund, N., Harms, J., Driggers, J.,
Adhikari, R. and Mitra, S. (2016), "Towards a first design of a Newtonian-noise
cancellation system for Advanced LIGO," Classical and Quantum Gravity, 33,
244001, arXiv:1606.01716, Eq. (1), which gives the plane-wave Rayleigh
surface-displacement acceleration

    delta_a_x(r0, omega) = -2 pi i G rho0 xi_z(r0, omega) exp(-k h) k_x / k

for a test mass at height ``h`` above the surface. The bulk term partially
cancels the areal term; in the plane-wave Rayleigh limit the sum is the areal
term times an O(1) Poisson-ratio-dependent factor (Harms 2019; Coughlin et al.
2016). Both terms are therefore implemented and kept separately selectable.

Two independent routes to the acceleration are provided:

- :func:`plane_wave_acceleration` -- closed-form plane-wave kernel, exact for a
  horizontally infinite, depth-resolved source; and
- :func:`acceleration_quadrature` -- a direct real-space sum over a source
  quadrature, an independent cross-check of the closed form.

The conversion from acceleration to strain is deliberately not done here: this
module returns per-position accelerations, and
:mod:`gwmock_newtonian_noise.detector.response` applies the explicit suspension and
strain conversion. Physical inputs -- density, wave speeds, grid extents -- are
all caller-supplied; no "realistic ET" parameter set is fixed here.

All accelerations are per unit *surface vertical reference amplitude*
(:data:`SURFACE_AMPLITUDE_REFERENCE_METRES`, set explicitly) and in
metres/second^2; a covariance carries an extra factor of the surface amplitude
PSD.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig, steering_vectors

#: Newtonian gravitational constant, in m^3 kg^-1 s^-2. Anchored to CODATA 2018
#: (Tiesinga, E. et al. (2021), Reviews of Modern Physics, 93, 025010).
GRAVITATIONAL_CONSTANT = 6.67430e-11

#: Surface vertical reference amplitude the per-mode responses are normalized to,
#: in metres. The field's steering vectors are per unit surface displacement, so
#: this is the explicit "unit" that turns them into physical amplitudes.
SURFACE_AMPLITUDE_REFERENCE_METRES = 1.0

#: Quadrature extent in horizontal decay lengths ``1 / k``; bounds the radial
#: domain of :func:`acceleration_quadrature`.
_QUADRATURE_EXTENT_DECAY_LENGTHS = 12.0


@dataclass(frozen=True)
class RayleighMode:
    """One fundamental-mode Rayleigh plane wave.

    Attributes:
        frequency: Frequency in hertz, strictly positive.
        wavenumber: Horizontal wavenumber ``k = 2 pi f / c`` in radians/metre.
        azimuth: Propagation azimuth ``phi`` in radians; ``k_hat = (cos phi, sin phi)``.
        phase_velocity: Phase velocity ``c`` in metres/second.
    """

    frequency: float
    wavenumber: float
    azimuth: float
    phase_velocity: float

    def __post_init__(self) -> None:
        """Reject non-physical mode values."""
        if self.frequency <= 0.0:
            raise ValueError(f"frequency must be positive, got {self.frequency}.")
        if self.wavenumber <= 0.0:
            raise ValueError(f"wavenumber must be positive, got {self.wavenumber}.")
        if self.phase_velocity <= 0.0:
            raise ValueError(f"phase_velocity must be positive, got {self.phase_velocity}.")

    @property
    def direction(self) -> NDArray[np.float64]:
        """Return the horizontal propagation unit vector ``k_hat``, length 3 with zero z."""
        return np.array([np.cos(self.azimuth), np.sin(self.azimuth), 0.0])

    @property
    def decay_length(self) -> float:
        """Return the horizontal scale ``1 / k`` on which the kernels oscillate and decay."""
        return 1.0 / self.wavenumber


def density_depth_amplitude(
    depths: NDArray[np.floating], frequency: float, config: RayleighWaveFieldConfig
) -> NDArray[np.float64]:
    """Return the bulk density perturbation ``delta_rho`` per unit surface amplitude.

    Evaluates the field's ``"density"`` steering component (bulk mass
    conservation ``-rho0 * div(u)``) at horizontal origin, where the plane-wave
    envelope is unity, so the returned profile is real and depends on depth only.

    Args:
        depths: Depth coordinates in metres (``z`` positive downward).
        frequency: Frequency in hertz, strictly positive.
        config: The seismic-field configuration.

    Returns:
        The real density perturbation profile in kilograms/metre^3 per metre of
        surface amplitude (i.e. kilograms/metre^4).
    """
    positions = [(0.0, 0.0, float(z)) for z in np.asarray(depths, dtype=float)]
    steering = steering_vectors(positions, frequency, config)
    return np.real(steering["density"][0, :])


def surface_vertical_amplitude(frequency: float, config: RayleighWaveFieldConfig) -> float:
    """Return the surface vertical displacement of the field per unit surface amplitude.

    The field normalizes every steering vector to a unit surface vertical
    displacement, so this is identically 1; it is read from the field rather
    than assumed, so a change of field convention cannot silently pass.
    """
    steering = steering_vectors([(0.0, 0.0, 0.0)], frequency, config)
    return float(np.real(steering["z"][0, 0]))


def _depth_integrals(
    profile: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    mode: RayleighMode,
    mass_depth: float,
    depth_max: float,
    n_depth: int,
) -> tuple[complex, complex]:
    """Return the smooth and signed depth overlap integrals of the mode.

    Computes ``I1 = integral profile(z) exp(-k |z - z0|) dz`` and
    ``I2 = integral profile(z) sign(z - z0) exp(-k |z - z0|) dz`` on
    ``[0, depth_max]`` with Gauss-Legendre rules on each side of the kink at
    ``z0`` (a single rule across the kink would be inaccurate).

    Args:
        profile: Callable returning the source amplitude at depths.
        mode: The Rayleigh mode.
        mass_depth: Test-mass depth ``z0`` in metres.
        depth_max: Upper integration bound in metres.
        n_depth: Number of quadrature nodes per side of the kink.

    Returns:
        ``(I1, I2)``.
    """
    if not (0.0 <= mass_depth <= depth_max):
        raise ValueError(f"mass_depth must lie in [0, depth_max], got {mass_depth} not in [0, {depth_max}].")
    k = mode.wavenumber
    nodes, weights = np.polynomial.legendre.leggauss(n_depth)
    inner = float(max(mass_depth, 0.0))
    z_lower = 0.5 * inner * (nodes + 1.0)
    w_lower = 0.5 * inner * weights
    z_upper = inner + 0.5 * (depth_max - inner) * (nodes + 1.0)
    w_upper = 0.5 * (depth_max - inner) * weights
    lower_source = np.asarray(profile(z_lower), dtype=complex)
    upper_source = np.asarray(profile(z_upper), dtype=complex)
    lower_kernel = np.exp(-k * (mass_depth - z_lower))
    upper_kernel = np.exp(-k * (z_upper - mass_depth))
    smooth = complex(np.sum(w_lower * lower_source * lower_kernel) + np.sum(w_upper * upper_source * upper_kernel))
    signed = complex(-np.sum(w_lower * lower_source * lower_kernel) + np.sum(w_upper * upper_source * upper_kernel))
    return smooth, signed


def plane_wave_acceleration(  # noqa: PLR0913 -- one parameter per physically distinct input.
    positions: Sequence[Sequence[float]],
    mode: RayleighMode,
    density_profile: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    medium_density: float,
    *,
    reference_amplitude: float = SURFACE_AMPLITUDE_REFERENCE_METRES,
    surface_vertical_displacement: float = 1.0,
    include_volume: bool = True,
    include_areal: bool = True,
    depth_max: float | None = None,
    n_depth: int = 400,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> NDArray[np.complex128]:
    """Return the closed-form plane-wave Newtonian acceleration at each position.

    For a horizontally infinite mode with density perturbation
    ``delta_rho(r) = A(z) exp(i k_hat . r_h)`` (``A`` supplied as
    ``density_profile``, per unit surface amplitude) and areal density
    ``delta_sigma = -rho0 * xi_z`` with ``xi_z`` the package's down-positive
    surface vertical displacement, reducing the horizontal integral with
    ``integral exp(i q . s) / |s, h| d^2s = (2 pi / q) exp(-q |h|)`` gives

        a_h =  i 2 pi G k_hat phase (I1 - rho0 xi_z exp(-k z0)),
        a_z =    2 pi G       phase (I2 + rho0 xi_z exp(-k z0)),

    with ``phase = exp(i k_hat . r0_h)`` and ``I1``/``I2`` the smooth and signed
    depth overlap integrals of :func:`_depth_integrals`.

    Args:
        positions: Test-mass positions ``(x, y, z)`` in metres.
        mode: The Rayleigh mode.
        density_profile: Bulk density amplitude ``A(z)`` per unit surface
            amplitude, in kilograms/metre^4.
        medium_density: Background density ``rho0`` in kilograms/metre^3.
        reference_amplitude: Surface reference amplitude in metres, matching the
            normalization of ``density_profile``.
        surface_vertical_displacement: Surface vertical displacement per unit
            surface amplitude (identically 1 for the field's normalization).
        include_volume: Include the bulk mass-conservation term.
        include_areal: Include the free-surface areal term.
        depth_max: Depth integration bound in metres; defaults to ``z0 + 12 / k``.
        n_depth: Gauss-Legendre nodes per side of the kink.
        gravitational_constant: Gravitational constant in SI units.

    Returns:
        A complex ``(n_positions, 3)`` array of accelerations in metres/second^2.
    """
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:  # noqa: PLR2004
        raise ValueError(f"positions must have shape (n, 3), got {positions_array.shape}.")
    k = mode.wavenumber
    direction = mode.direction
    phase = np.exp(1j * k * (positions_array[:, 0] * direction[0] + positions_array[:, 1] * direction[1]))
    areal_amplitude = medium_density * reference_amplitude * surface_vertical_displacement
    return _assemble_plane_wave(
        positions_array,
        mode,
        density_profile,
        reference_amplitude,
        areal_amplitude,
        phase,
        k,
        direction,
        include_volume,
        include_areal,
        depth_max,
        n_depth,
        gravitational_constant,
    )


def _assemble_plane_wave(  # noqa: PLR0913, PLR0917 -- one parameter per physically distinct input.
    positions_array: NDArray[np.float64],
    mode: RayleighMode,
    density_profile: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    reference_amplitude: float,
    areal_amplitude: float,
    phase: NDArray[np.complex128],
    k: float,
    direction: NDArray[np.float64],
    include_volume: bool,
    include_areal: bool,
    depth_max: float | None,
    n_depth: int,
    gravitational_constant: float,
) -> NDArray[np.complex128]:
    """Assemble the plane-wave acceleration from the depth overlap integrals.

    The bulk profile is supplied per unit surface amplitude, so ``I1``/``I2`` are
    scaled by ``reference_amplitude`` here, exactly as the areal amplitude and the
    direct quadrature's source already are.
    """
    decay = 1.0 / k
    acceleration = np.zeros((positions_array.shape[0], 3), dtype=complex)
    for index, z0 in enumerate(positions_array[:, 2]):
        upper = depth_max if depth_max is not None else float(z0) + _QUADRATURE_EXTENT_DECAY_LENGTHS * decay
        i1, i2 = _depth_integrals(density_profile, mode, float(z0), upper, n_depth)
        horizontal = 0.0j
        vertical = 0.0j
        if include_volume:
            horizontal += reference_amplitude * i1
            vertical += reference_amplitude * i2
        if include_areal:
            areal = areal_amplitude * np.exp(-k * max(float(z0), 0.0))
            horizontal -= areal
            vertical += areal
        prefactor = 2.0 * np.pi * gravitational_constant * phase[index]
        acceleration[index, :2] = 1j * prefactor * direction[:2] * horizontal
        acceleration[index, 2] = prefactor * vertical
    return acceleration


def _green_plane_sum(
    s_x: NDArray[np.float64],
    s_y: NDArray[np.float64],
    area_weight: NDArray[np.float64],
    phase_grating: NDArray[np.complex128],
    height: float,
) -> NDArray[np.complex128]:
    """Return ``sum exp(i k.s) (-s_x, -s_y, height) / |s, height|^3`` over the grid."""
    radius_squared = s_x**2 + s_y**2 + height**2
    factor = area_weight * phase_grating / radius_squared**1.5
    return np.array(
        [-np.sum(factor * s_x), -np.sum(factor * s_y), np.sum(factor * height)],
        dtype=complex,
    )


def acceleration_quadrature(  # noqa: PLR0913 -- one parameter per physically distinct input.
    positions: Sequence[Sequence[float]],
    mode: RayleighMode,
    density_profile: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    medium_density: float,
    *,
    radial_max: float,
    n_radial: int = 200,
    n_azimuth: int = 64,
    depth_max: float | None = None,
    n_depth: int = 160,
    reference_amplitude: float = SURFACE_AMPLITUDE_REFERENCE_METRES,
    surface_vertical_displacement: float = 1.0,
    include_volume: bool = True,
    include_areal: bool = True,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> NDArray[np.complex128]:
    """Return the Newtonian acceleration by direct real-space source integration.

    This is the independent cross-check of :func:`plane_wave_acceleration`: it
    does not use the Fourier-reduced kernel, but sums the Green's function
    ``-G (r0 - r) / |r0 - r|^3`` times the source density over an explicit
    polar/depth quadrature (bulk in the half-space, areal on the surface).

    Args:
        positions: Test-mass positions ``(x, y, z)`` in metres.
        mode: The Rayleigh mode.
        density_profile: Bulk density amplitude per unit surface amplitude, in
            kilograms/metre^4.
        medium_density: Background density ``rho0`` in kilograms/metre^3.
        radial_max: Horizontal integration radius in metres, strictly positive.
        n_radial: Number of radial quadrature nodes.
        n_azimuth: Number of azimuthal quadrature nodes.
        depth_max: Depth integration bound in metres; defaults to a multiple of
            the mode decay length above the deepest test mass.
        n_depth: Number of depth quadrature nodes.
        reference_amplitude: Surface reference amplitude in metres.
        surface_vertical_displacement: Surface vertical displacement per unit
            surface amplitude (identically 1 for the field's normalization).
        include_volume: Include the bulk term.
        include_areal: Include the areal term.
        gravitational_constant: Gravitational constant in SI units.

    Returns:
        A complex ``(n_positions, 3)`` array of accelerations in metres/second^2.
    """
    if radial_max <= 0.0:
        raise ValueError(f"radial_max must be positive, got {radial_max}.")
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:  # noqa: PLR2004
        raise ValueError(f"positions must have shape (n, 3), got {positions_array.shape}.")
    k = mode.wavenumber
    direction = mode.direction
    decay = 1.0 / k

    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    radii = 0.5 * radial_max * (nodes + 1.0)
    radial_weights = 0.5 * radial_max * weights
    azimuths = (np.arange(n_azimuth) + 0.5) * (2.0 * np.pi / n_azimuth)
    azimuth_weight = 2.0 * np.pi / n_azimuth
    s_x = np.cos(azimuths)[None, :] * radii[:, None]
    s_y = np.sin(azimuths)[None, :] * radii[:, None]
    area_weight = radial_weights[:, None] * radii[:, None] * azimuth_weight
    phase_grating = np.exp(1j * k * (s_x * direction[0] + s_y * direction[1]))

    acceleration = np.zeros((positions_array.shape[0], 3), dtype=complex)
    for index, position in enumerate(positions_array):
        z0 = float(position[2])
        upper = depth_max if depth_max is not None else z0 + _QUADRATURE_EXTENT_DECAY_LENGTHS * decay
        total = np.zeros(3, dtype=complex)
        if include_volume:
            depth_nodes, depth_weights = np.polynomial.legendre.leggauss(n_depth)
            depths = 0.5 * upper * (depth_nodes + 1.0)
            depth_step = 0.5 * upper * depth_weights
            source = np.asarray(density_profile(depths), dtype=float) * reference_amplitude
            for z_source, weight, amplitude in zip(depths, depth_step, source, strict=True):
                total += weight * amplitude * _green_plane_sum(s_x, s_y, area_weight, phase_grating, z0 - z_source)
        if include_areal:
            areal_source = -medium_density * reference_amplitude * surface_vertical_displacement
            total += areal_source * _green_plane_sum(s_x, s_y, area_weight, phase_grating, z0)
        phase = np.exp(1j * k * (position[0] * direction[0] + position[1] * direction[1]))
        acceleration[index] = -gravitational_constant * total * phase
    return acceleration


__all__ = [
    "GRAVITATIONAL_CONSTANT",
    "SURFACE_AMPLITUDE_REFERENCE_METRES",
    "RayleighMode",
    "acceleration_quadrature",
    "density_depth_amplitude",
    "plane_wave_acceleration",
    "surface_vertical_amplitude",
]
