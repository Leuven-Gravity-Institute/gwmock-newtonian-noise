"""Fundamental-mode Love waves in a single layer over a half-space.

Love waves are horizontally polarized shear (SH) waves guided by a low-velocity
surface layer; they do not exist on a homogeneous half-space, so this family
carries its own layered medium. The model is a layer ``0 <= z < H`` (shear speed
``beta1``, density ``rho1``) over a half-space ``z >= H`` (``beta2 > beta1``,
``rho2``), with a traction-free surface at ``z = 0``. A mode travelling along
azimuth ``phi`` has displacement purely along ``t = (-sin phi, cos phi, 0)``,

    u = U * w(z) * t * exp(i k (x cos phi + y sin phi)),
    w(z) = cos(nu1 z)                              for 0 <= z < H,
    w(z) = cos(nu1 H) exp(-k s2 (z - H))           for z >= H,

with ``nu1 = k sqrt(c^2/beta1^2 - 1)``, ``s2 = sqrt(1 - c^2/beta2^2)``, so the
reference amplitude ``U`` is the surface transverse displacement. Continuity of
the shear traction ``mu dw/dz`` at ``z = H`` gives Love's dispersion relation

    mu1 sqrt(c^2/beta1^2 - 1) tan(k H sqrt(c^2/beta1^2 - 1)) = mu2 sqrt(1 - c^2/beta2^2),

primary source Love, A. E. H. (1911), *Some Problems of Geodynamics*, Cambridge
University Press. The fundamental mode has no cut-off and its phase
velocity lies in ``(beta1, beta2)``; :func:`love_phase_velocity` solves for it.

Newtonian coupling. The displacement is divergence-free (it is transverse to the
horizontal wavevector and has no vertical component), so it sources no bulk
density perturbation, and it has no vertical displacement at the free surface or
at the interface, so it sources no areal density either. Its field-intrinsic
Newtonian acceleration is therefore *exactly zero*, and :meth:`LoveWaveFamily.acceleration`
returns zeros. This is a derived result for this idealized geometry, not an
assumption that Love waves are harmless: they are seen by every horizontal
witness, and they couple to a test mass through the cavity-wall term applied by
:mod:`gwmock_newtonian.families.response`. Topography, lateral heterogeneity and
mode conversion, which can restore direct coupling (Hughes, S. A. and Thorne,
K. S. (1998), "Seismic gravity-gradient noise in interferometric
gravitational-wave detectors," Physical Review D, 58, 122002), are not modelled.

Layered Rayleigh modes are not implemented; only the minimal homogeneous
half-space Rayleigh family exists in this package.

UNANCHORED: the layer and half-space properties, the PSD and the directional
density are site parameters with no default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from gwmock_newtonian.families.base import (
    PSDCallable,
    as_positions,
    evaluate_psd,
    require_positive_frequency,
)

#: Fewest azimuth quadrature points that can resolve a direction at all.
_MIN_AZIMUTH_POINTS = 4

#: Relative offset from the bracket ends of the dispersion root search, where the
#: dispersion function is singular or degenerate.
_BRACKET_EPSILON = 1e-12


@dataclass(frozen=True)
class LayerOverHalfSpace:
    """A single elastic layer over a half-space, for SH (Love) motion.

    Attributes:
        layer_thickness: Layer thickness ``H`` in metres, positive.
        layer_s_wave_speed: Layer shear speed ``beta1`` in metres/second, positive.
        layer_density: Layer density ``rho1`` in kilograms/metre^3, positive.
        half_space_s_wave_speed: Half-space shear speed ``beta2 > beta1``.
        half_space_density: Half-space density ``rho2`` in kilograms/metre^3, positive.
    """

    layer_thickness: float
    layer_s_wave_speed: float
    layer_density: float
    half_space_s_wave_speed: float
    half_space_density: float

    def __post_init__(self) -> None:
        """Reject a medium that cannot guide a Love wave."""
        for name in ("layer_thickness", "layer_s_wave_speed", "layer_density", "half_space_density"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}.")
        if not (self.half_space_s_wave_speed > self.layer_s_wave_speed):
            raise ValueError(
                "a Love wave needs a slower layer: half_space_s_wave_speed must exceed layer_s_wave_speed "
                f"(got {self.half_space_s_wave_speed} <= {self.layer_s_wave_speed})."
            )

    @property
    def layer_mu(self) -> float:
        """Layer shear modulus ``rho1 beta1^2`` in pascals."""
        return self.layer_density * self.layer_s_wave_speed**2

    @property
    def half_space_mu(self) -> float:
        """Half-space shear modulus ``rho2 beta2^2`` in pascals."""
        return self.half_space_density * self.half_space_s_wave_speed**2

    def density_at(self, depths: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the background density at each depth (the interface belongs to the half-space)."""
        depths = np.asarray(depths, dtype=float)
        return np.where(depths < self.layer_thickness, self.layer_density, self.half_space_density)


def love_dispersion_residual(medium: LayerOverHalfSpace, frequency: float, phase_velocity: float) -> float:
    """Return ``mu1 s1 sin(k H s1) - mu2 s2 cos(k H s1)``, zero on a Love mode (pole-free form)."""
    k = 2.0 * np.pi * frequency / phase_velocity
    s1 = np.sqrt(phase_velocity**2 / medium.layer_s_wave_speed**2 - 1.0)
    s2 = np.sqrt(1.0 - phase_velocity**2 / medium.half_space_s_wave_speed**2)
    argument = k * medium.layer_thickness * s1
    return float(medium.layer_mu * s1 * np.sin(argument) - medium.half_space_mu * s2 * np.cos(argument))


def love_phase_velocity(medium: LayerOverHalfSpace, frequency: float) -> float:
    """Return the fundamental-mode Love phase velocity at one frequency.

    The fundamental root has ``k H s1`` in ``(0, pi/2)``: the residual is negative
    just above ``beta1`` and positive either where ``k H s1`` reaches ``pi/2`` or at
    ``beta2``, whichever comes first, so a sign-changing bracket always exists.
    """
    frequency = require_positive_frequency(frequency)
    beta1, beta2 = medium.layer_s_wave_speed, medium.half_space_s_wave_speed
    lower = beta1 * (1.0 + _BRACKET_EPSILON)
    upper = beta2 * (1.0 - _BRACKET_EPSILON)

    def quarter_wave_excess(c: float) -> float:
        k = 2.0 * np.pi * frequency / c
        return k * medium.layer_thickness * np.sqrt(c**2 / beta1**2 - 1.0) - 0.5 * np.pi

    if quarter_wave_excess(upper) > 0.0:
        upper = float(brentq(quarter_wave_excess, lower, upper, xtol=1e-14, rtol=1e-14)) * (1.0 - _BRACKET_EPSILON)
    return float(brentq(lambda c: love_dispersion_residual(medium, frequency, c), lower, upper, xtol=1e-12, rtol=1e-14))


@dataclass(frozen=True)
class LoveWaveFamily:
    """Incoherent fundamental-mode Love waves over azimuth.

    Attributes:
        medium: The layer over half-space (UNANCHORED).
        surface_displacement_psd: One-sided PSD of the surface transverse
            displacement, m^2/Hz (UNANCHORED).
        directional_density: Density over propagation azimuth (UNANCHORED),
            renormalized over the midpoint azimuth grid.
        n_azimuth: Number of azimuth cells, at least 4.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]] = ("medium", "surface_displacement_psd", "directional_density")

    medium: LayerOverHalfSpace
    surface_displacement_psd: PSDCallable
    directional_density: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    n_azimuth: int = 32

    def __post_init__(self) -> None:
        """Reject an azimuth grid too coarse to resolve a direction."""
        if self.n_azimuth < _MIN_AZIMUTH_POINTS:
            raise ValueError(f"n_azimuth must be at least {_MIN_AZIMUTH_POINTS}, got {self.n_azimuth}.")

    @property
    def n_modes(self) -> int:
        """Number of latent modes: one per azimuth cell."""
        return self.n_azimuth

    def azimuth_grid(self) -> NDArray[np.float64]:
        """Return the midpoint azimuth grid over ``[0, 2 pi)``."""
        return (np.arange(self.n_azimuth) + 0.5) * (2.0 * np.pi / self.n_azimuth)

    def azimuth_weights(self) -> NDArray[np.float64]:
        """Return the normalized azimuth weights, summing to 1."""
        raw = np.asarray(self.directional_density(self.azimuth_grid()), dtype=float)
        if raw.shape != (self.n_azimuth,) or np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("directional_density must return finite non-negative values, one per azimuth.")
        if np.sum(raw) <= 0.0:
            raise ValueError("directional_density must be positive somewhere on the azimuth grid.")
        return raw / np.sum(raw)

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the ``(n_modes,)`` mode powers: the surface PSD times the azimuth weights."""
        frequency = require_positive_frequency(frequency)
        return (
            evaluate_psd(self.surface_displacement_psd, frequency, "surface_displacement_psd") * self.azimuth_weights()
        )

    def depth_shape(self, depths: NDArray[np.float64], frequency: float) -> NDArray[np.float64]:
        """Return the real depth eigenfunction ``w(z)``, normalized to ``w(0) = 1``."""
        depths = np.asarray(depths, dtype=float)
        if np.any(depths < 0.0):
            raise ValueError("Love-wave depths must be non-negative (z is depth, positive downward).")
        c = love_phase_velocity(self.medium, frequency)
        k = 2.0 * np.pi * frequency / c
        nu1 = k * np.sqrt(c**2 / self.medium.layer_s_wave_speed**2 - 1.0)
        decay = k * np.sqrt(1.0 - c**2 / self.medium.half_space_s_wave_speed**2)
        thickness = self.medium.layer_thickness
        below = np.cos(nu1 * thickness) * np.exp(-decay * np.maximum(depths - thickness, 0.0))
        return np.where(depths < thickness, np.cos(nu1 * depths), below)

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_modes, n_positions, 3)`` displacement per unit surface amplitude."""
        positions = as_positions(positions)
        frequency = require_positive_frequency(frequency)
        k = 2.0 * np.pi * frequency / love_phase_velocity(self.medium, frequency)
        shape = self.depth_shape(positions[:, 2], frequency)
        phi = self.azimuth_grid()
        phase = np.exp(
            1j * k * (np.cos(phi)[:, None] * positions[None, :, 0] + np.sin(phi)[:, None] * positions[None, :, 1])
        )
        transverse = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        return (phase * shape[None, :])[:, :, None] * transverse[:, None, :]

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return zeros: a Love wave sources no bulk or areal density (see the module docstring)."""
        positions = as_positions(positions)
        require_positive_frequency(frequency)
        if np.any(positions[:, 2] < 0.0):
            raise ValueError("Love-wave positions must lie in the medium (z >= 0).")
        return np.zeros((self.n_modes, positions.shape[0], 3), dtype=complex)

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the layer or half-space density at each position."""
        return self.medium.density_at(as_positions(positions)[:, 2])


__all__ = ["LayerOverHalfSpace", "LoveWaveFamily", "love_dispersion_residual", "love_phase_velocity"]
