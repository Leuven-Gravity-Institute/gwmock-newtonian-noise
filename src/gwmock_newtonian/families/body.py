"""Body-wave (P, SV, SH) Newtonian-noise family with an explicit P/S composition.

The field is an incoherent superposition of plane body waves in a homogeneous
isotropic elastic medium, labelled by wave type (P, SV, SH) and propagation
direction on the sphere. Each type has its own caller-supplied one-sided PSD of
the *incident* displacement amplitude, so the P/S composition is always explicit
and frequency-dependent; none of the three PSDs has a default. Directions use the
equal-area :class:`~gwmock_newtonian.families.base.SphereGrid` and a
caller-supplied directional density on the sphere, so anisotropy is a property
of the configuration, not a special case.

Polarizations (``n`` the propagation direction, ``t = (-sin phi, cos phi, 0)``
the horizontal transverse unit vector of azimuth ``phi``): P along ``n``; SH
along ``t``; SV along ``t x n``, in the vertical plane of propagation. This
definition is regular at every non-vertical direction and fixes the SV sign.

Two geometries are offered, both mandatory to choose explicitly:

- ``free_surface=False`` -- the **infinite-medium limit**: each mode is a single
  plane wave and directions cover the whole sphere. The Newtonian acceleration
  is the exact full-space relation ``a = 4 pi G rho0 u_P``
  (:func:`~gwmock_newtonian.families.gravity.full_space_acceleration`): shear
  waves source no gravity at all in an infinite homogeneous medium.
- ``free_surface=True`` -- the **half-space** ``z >= 0``: only upgoing incident
  directions are gridded, and each mode is the incident wave plus the reflected
  P, SV and SH waves that make the surface ``z = 0`` traction-free. The
  reflection amplitudes are solved from the three traction conditions
  ``sigma_xz = sigma_yz = sigma_zz = 0`` with Hooke's law, so P-SV conversion
  and post-critical (evanescent, complex vertical wavenumber) reflected P are
  included automatically and SH decouples. The free-surface reflection problem is
  Knott, C. G. (1899), "Reflexion and refraction of elastic waves, with
  seismological applications," Philosophical Magazine, 48, 64-97; the test suite
  checks the solution against the textbook closed-form P-incidence coefficient
  (Aki, K. and Richards, P. G. (2002), *Quantitative Seismology*, 2nd ed.,
  Chap. 5), against the convention-free vertical energy-flux balance, and against the traction condition evaluated by
  numerical differentiation. The Newtonian acceleration is the closed-form
  half-space bulk plus free-surface areal term
  (:func:`~gwmock_newtonian.families.gravity.half_space_acceleration`).

What is not modelled: layering, attenuation, topography, cavity *scattering* of
the seismic field, and the surface waves a half-space diffuse field would also
contain. The equipartitioned diffuse field of :func:`equipartitioned_body_waves`
is therefore offered only in the infinite-medium limit, where its energy ratio
is derived.

UNANCHORED: the medium, all three PSDs and the directional density are site
parameters with no default; any values in tests are illustrative only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.families.base import (
    PSDCallable,
    SphereDensity,
    SphereGrid,
    as_positions,
    evaluate_psd,
    horizontal_basis,
    require_positive_frequency,
    uniform_sphere_density,
)
from gwmock_newtonian.families.gravity import (
    PlaneWaveComponent,
    component_displacement,
    full_space_acceleration,
    half_space_acceleration,
)
from gwmock_newtonian.seismic.medium import ElasticHalfSpace

#: The body-wave types, in mode order (type-major).
BODY_WAVE_TYPES: tuple[str, ...] = ("P", "SV", "SH")


def _vertical_wavenumber(wavenumber: float, horizontal: float) -> complex:
    """Return the downgoing vertical wavenumber ``sqrt(K^2 - k_h^2)`` on the decaying branch.

    ``numpy.sqrt`` of a negative real (as complex) returns ``+i sqrt(.)``, so a
    post-critical component ``exp(i q z)`` decays with depth.
    """
    return complex(np.sqrt(complex(wavenumber**2 - horizontal**2)))


def _polarization(
    wave_type: str, wavevector: NDArray[np.complex128], wavenumber: float, transverse: NDArray
) -> NDArray:
    """Return the polarization of a P, SV or SH component with wavevector ``K`` and ``K . K = wavenumber^2``."""
    if wave_type == "P":
        return wavevector / wavenumber
    if wave_type == "SH":
        return transverse.astype(complex)
    return np.cross(transverse, wavevector) / wavenumber


def _traction(component: PlaneWaveComponent, medium: ElasticHalfSpace) -> NDArray[np.complex128]:
    """Return the traction ``sigma_i3`` (``i`` = x, y, z) on a horizontal plane at the origin, per unit amplitude.

    Hooke's law ``sigma_ij = lambda delta_ij div(u) + mu (d_i u_j + d_j u_i)`` with
    ``d_j -> i K_j`` for one plane-wave component.
    """
    k, e = component.wavevector, component.polarization
    traction = 1j * medium.lame_mu * (k * e[2] + k[2] * e)
    traction[2] += 1j * medium.lame_lambda * complex(np.sum(k * e))
    return traction * component.amplitude


def free_surface_components(
    medium: ElasticHalfSpace, wave_type: str, direction: NDArray[np.float64], frequency: float
) -> tuple[PlaneWaveComponent, ...]:
    """Return the incident wave and its three free-surface reflections for one upgoing body wave.

    Args:
        medium: The homogeneous half-space.
        wave_type: ``"P"``, ``"SV"`` or ``"SH"``.
        direction: Unit propagation direction of the incident wave, ``n_z < 0``
            (upgoing) and not exactly vertical.
        frequency: Frequency in hertz, strictly positive.

    Returns:
        ``(incident, reflected P, reflected SV, reflected SH)``; the incident
        amplitude is 1 and the reflected amplitudes make the surface ``z = 0``
        traction-free.
    """
    if wave_type not in BODY_WAVE_TYPES:
        raise ValueError(f"wave_type must be one of {BODY_WAVE_TYPES}, got {wave_type!r}.")
    direction = np.asarray(direction, dtype=float)
    if direction[2] >= 0.0:
        raise ValueError(f"an incident wave must propagate upward (n_z < 0), got direction {direction}.")
    omega = 2.0 * np.pi * require_positive_frequency(frequency)
    speed = medium.p_wave_speed if wave_type == "P" else medium.s_wave_speed
    k_incident = omega / speed
    horizontal = k_incident * float(np.hypot(direction[0], direction[1]))
    if horizontal == 0.0:
        raise ValueError("an exactly vertical incident wave has no defined SV/SH basis; use a midpoint grid.")
    k_hat, transverse = horizontal_basis(float(np.arctan2(direction[1], direction[0])))

    incident_k = (k_incident * direction).astype(complex)
    incident = PlaneWaveComponent(incident_k, _polarization(wave_type, incident_k, k_incident, transverse))

    k_p, k_s = omega / medium.p_wave_speed, omega / medium.s_wave_speed
    reflected_p_k = np.array([horizontal * k_hat[0], horizontal * k_hat[1], _vertical_wavenumber(k_p, horizontal)])
    reflected_s_k = np.array([horizontal * k_hat[0], horizontal * k_hat[1], _vertical_wavenumber(k_s, horizontal)])
    basis = (
        PlaneWaveComponent(reflected_p_k, _polarization("P", reflected_p_k, k_p, transverse)),
        PlaneWaveComponent(reflected_s_k, _polarization("SV", reflected_s_k, k_s, transverse)),
        PlaneWaveComponent(reflected_s_k, _polarization("SH", reflected_s_k, k_s, transverse)),
    )
    system = np.stack([_traction(component, medium) for component in basis], axis=1)
    amplitudes = np.linalg.solve(system, -_traction(incident, medium))
    reflected = tuple(
        PlaneWaveComponent(component.wavevector, component.polarization, complex(amplitude))
        for component, amplitude in zip(basis, amplitudes, strict=True)
    )
    return (incident, *reflected)


@dataclass(frozen=True)
class BodyWaveFamily:
    """Incoherent P/SV/SH plane body waves with explicit composition and directionality.

    Attributes:
        medium: The homogeneous isotropic medium (UNANCHORED).
        p_psd: One-sided PSD of the incident P displacement amplitude, m^2/Hz (UNANCHORED).
        sv_psd: One-sided PSD of the incident SV displacement amplitude, m^2/Hz (UNANCHORED).
        sh_psd: One-sided PSD of the incident SH displacement amplitude, m^2/Hz (UNANCHORED).
        directional_density: Density of propagation directions on the sphere
            (UNANCHORED); renormalized over the grid, shared by all three types.
        free_surface: ``True`` for the half-space with free-surface reflections
            (incident directions upgoing only), ``False`` for the infinite-medium
            limit (directions over the whole sphere). No default: the geometry
            is a modelling choice the caller must state.
        n_polar: Number of ``cos(theta)`` cells of the direction grid.
        n_azimuth: Number of azimuth cells of the direction grid.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]] = ("medium", "p_psd", "sv_psd", "sh_psd", "directional_density")

    medium: ElasticHalfSpace
    p_psd: PSDCallable
    sv_psd: PSDCallable
    sh_psd: PSDCallable
    directional_density: SphereDensity
    free_surface: bool
    n_polar: int = 8
    n_azimuth: int = 16

    @property
    def grid(self) -> SphereGrid:
        """Return the direction grid (the upgoing hemisphere when there is a free surface)."""
        return SphereGrid(self.n_polar, self.n_azimuth, upgoing_only=self.free_surface)

    @property
    def n_modes(self) -> int:
        """Number of latent modes: three wave types times the direction cells."""
        return len(BODY_WAVE_TYPES) * self.grid.size

    def type_power(self, frequency: float) -> dict[str, float]:
        """Return the one-sided incident-amplitude PSD of each wave type at one frequency."""
        frequency = require_positive_frequency(frequency)
        return {
            "P": evaluate_psd(self.p_psd, frequency, "p_psd"),
            "SV": evaluate_psd(self.sv_psd, frequency, "sv_psd"),
            "SH": evaluate_psd(self.sh_psd, frequency, "sh_psd"),
        }

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the ``(n_modes,)`` mode powers: each type's PSD times the direction weights."""
        weights = self.grid.weights(self.directional_density)
        powers = self.type_power(frequency)
        return np.concatenate([powers[wave_type] * weights for wave_type in BODY_WAVE_TYPES])

    def mode_components(self, frequency: float) -> list[tuple[PlaneWaveComponent, ...]]:
        """Return the plane-wave components of every mode, type-major then direction."""
        frequency = require_positive_frequency(frequency)
        omega = 2.0 * np.pi * frequency
        modes: list[tuple[PlaneWaveComponent, ...]] = []
        for wave_type in BODY_WAVE_TYPES:
            speed = self.medium.p_wave_speed if wave_type == "P" else self.medium.s_wave_speed
            for direction in self.grid.directions():
                if self.free_surface:
                    modes.append(free_surface_components(self.medium, wave_type, direction, frequency))
                    continue
                wavevector = (omega / speed * direction).astype(complex)
                _, transverse = horizontal_basis(float(np.arctan2(direction[1], direction[0])))
                polarization = _polarization(wave_type, wavevector, omega / speed, transverse)
                modes.append((PlaneWaveComponent(wavevector, polarization),))
        return modes

    def _check_positions(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        positions = as_positions(positions)
        if self.free_surface and np.any(positions[:, 2] < 0.0):
            raise ValueError("with a free surface, positions must lie in the medium (z >= 0).")
        return positions

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_modes, n_positions, 3)`` displacement per unit incident amplitude."""
        positions = self._check_positions(positions)
        return np.stack([component_displacement(mode, positions) for mode in self.mode_components(frequency)])

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_modes, n_positions, 3)`` Newtonian acceleration per unit incident amplitude."""
        positions = self._check_positions(positions)
        kernel = half_space_acceleration if self.free_surface else full_space_acceleration
        return np.stack([kernel(mode, positions, self.medium.density) for mode in self.mode_components(frequency)])

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the (uniform) background density at each position."""
        return np.full(as_positions(positions).shape[0], self.medium.density)


def equipartition_shear_to_p_ratio(medium: ElasticHalfSpace) -> float:
    """Return the equipartitioned S-to-P energy ratio ``2 (alpha / beta)^3`` of a diffuse elastic field.

    Primary source: Weaver, R. L. (1982), "On diffuse waves in solid media,"
    Journal of the Acoustical Society of America, 71(6), 1608-1609 -- in a fully
    diffuse field every mode carries equal energy, and the density of shear
    modes (two polarizations, speed ``beta``) exceeds that of P modes (one
    polarization, speed ``alpha``) by ``2 (alpha / beta)^3``.
    """
    return 2.0 * (medium.p_wave_speed / medium.s_wave_speed) ** 3


def equipartitioned_body_waves(
    medium: ElasticHalfSpace, total_psd: PSDCallable, *, n_polar: int = 8, n_azimuth: int = 16
) -> BodyWaveFamily:
    """Return the isotropic, equipartitioned diffuse body-wave field of an infinite medium.

    A scattering proxy: the late-coda limit of multiple scattering is an
    isotropic field with energy equipartitioned between P and S modes
    (:func:`equipartition_shear_to_p_ratio`), split equally between SV and SH.
    With equal frequency for every mode the displacement power splits in the same
    ratio as the energy, so ``P : SV : SH = 1 : R/2 : R/2`` with ``R = 2 (alpha/beta)^3``.
    Offered in the infinite-medium limit only (see the module docstring).

    Args:
        medium: The homogeneous medium (UNANCHORED).
        total_psd: One-sided PSD of the summed P+SV+SH incident amplitudes (UNANCHORED).
        n_polar: Number of ``cos(theta)`` cells.
        n_azimuth: Number of azimuth cells.

    Returns:
        The :class:`BodyWaveFamily`.
    """
    ratio = equipartition_shear_to_p_ratio(medium)
    p_fraction = 1.0 / (1.0 + ratio)
    s_fraction = 0.5 * ratio / (1.0 + ratio)

    def p_psd(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        return p_fraction * np.asarray(total_psd(frequencies), dtype=float)

    def s_psd(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        return s_fraction * np.asarray(total_psd(frequencies), dtype=float)

    return BodyWaveFamily(
        medium=medium,
        p_psd=p_psd,
        sv_psd=s_psd,
        sh_psd=s_psd,
        directional_density=uniform_sphere_density,
        free_surface=False,
        n_polar=n_polar,
        n_azimuth=n_azimuth,
    )


__all__ = [
    "BODY_WAVE_TYPES",
    "BodyWaveFamily",
    "equipartition_shear_to_p_ratio",
    "equipartitioned_body_waves",
    "free_surface_components",
]
