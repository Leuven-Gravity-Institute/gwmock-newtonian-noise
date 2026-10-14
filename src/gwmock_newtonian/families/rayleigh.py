"""The minimal Rayleigh-wave field exposed through the field-family contract.

This adapter adds no physics. Its modes are the azimuth quadrature cells of a
:class:`~gwmock_newtonian.seismic.field.RayleighWaveFieldConfig`; its
displacement is the field's own
:func:`~gwmock_newtonian.seismic.field.steering_vectors` and its acceleration is
the production closed-form kernel
:func:`~gwmock_newtonian.detector.newtonian.plane_wave_acceleration`, with the
same bulk/areal switches and depth quadrature the detector response uses. The
test suite checks that the generic family consumer reproduces the existing
joint strain--witness covariance entry for entry through this adapter, which is
what lets a Rayleigh field be mixed with the other families.

It also provides a laterally-inhomogeneous-medium *proxy*,
:func:`phase_velocity_spread`: an incoherent mixture of copies of one Rayleigh
field at a spread of phase velocities, mimicking the broadened apparent
wavenumber spectrum that lateral heterogeneity produces under an array. It is a
proxy, not a solution of a heterogeneous-medium problem: each copy is a
homogeneous-half-space mode evaluated at an overridden phase velocity (the
existing field's dispersion override).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.detector.newtonian import (
    RayleighMode,
    density_depth_amplitude,
    plane_wave_acceleration,
    surface_vertical_amplitude,
)
from gwmock_newtonian.families.base import as_positions, require_positive_frequency
from gwmock_newtonian.seismic.field import RayleighWaveFieldConfig, steering_vectors


@dataclass(frozen=True)
class RayleighFamily:
    """The homogeneous half-space Rayleigh field as a family of azimuth modes.

    Attributes:
        field: The Rayleigh field configuration (UNANCHORED; see its own docstring).
        include_volume: Include the bulk mass-conservation Newtonian term.
        include_areal: Include the free-surface areal Newtonian term.
        n_depth: Depth quadrature nodes per side of the evaluation depth.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]] = ("field",)

    field: RayleighWaveFieldConfig
    include_volume: bool = True
    include_areal: bool = True
    n_depth: int = 400

    def __post_init__(self) -> None:
        """Reject a configuration with both Newtonian terms disabled."""
        if not (self.include_volume or self.include_areal):
            raise ValueError("at least one of include_volume/include_areal must be true.")

    @property
    def n_modes(self) -> int:
        """Number of latent modes: one per azimuth quadrature cell."""
        return self.field.n_azimuth

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the surface PSD times the normalized azimuth weights."""
        frequency = require_positive_frequency(frequency)
        surface_psd = float(np.asarray(self.field.surface_displacement_psd(np.array([frequency])), dtype=float)[0])
        if surface_psd < 0.0:
            raise ValueError(f"surface_displacement_psd returned a negative value ({surface_psd}) at f={frequency}.")
        return surface_psd * self.field.azimuth_weights()

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_azimuth, n_positions, 3)`` displacement steering of the field."""
        positions = as_positions(positions)
        steering = steering_vectors([tuple(p) for p in positions], require_positive_frequency(frequency), self.field)
        return np.stack([steering["x"], steering["y"], steering["z"]], axis=-1)

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_azimuth, n_positions, 3)`` closed-form plane-wave Newtonian acceleration."""
        positions = as_positions(positions)
        frequency = require_positive_frequency(frequency)
        phase_velocity = float(self.field.phase_velocity_at(np.array([frequency]))[0])
        wavenumber = 2.0 * np.pi * frequency / phase_velocity
        vertical = surface_vertical_amplitude(frequency, self.field)

        def profile(depths: NDArray[np.float64]) -> NDArray[np.float64]:
            return density_depth_amplitude(depths, frequency, self.field)

        return np.stack(
            [
                plane_wave_acceleration(
                    positions,
                    RayleighMode(frequency, wavenumber, float(azimuth), phase_velocity),
                    profile,
                    self.field.medium.density,
                    surface_vertical_displacement=vertical,
                    include_volume=self.include_volume,
                    include_areal=self.include_areal,
                    n_depth=self.n_depth,
                )
                for azimuth in self.field.azimuth_grid()
            ]
        )

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the (uniform) half-space density at each position."""
        return np.full(as_positions(positions).shape[0], self.field.medium.density)


def phase_velocity_spread(
    family: RayleighFamily, velocity_factors: Sequence[float], weights: Sequence[float]
) -> tuple[RayleighFamily, ...]:
    """Return copies of a Rayleigh family at scaled phase velocities and split power.

    Copy ``j`` has phase velocity ``velocity_factors[j]`` times the original
    (default or overridden) one and surface PSD ``weights[j] / sum(weights)``
    times the original, so the summed power is unchanged. Combine the copies
    with :class:`~gwmock_newtonian.families.mixture.FamilyMixture`. A scaled
    velocity must stay below the shear speed; the field raises otherwise.

    Args:
        family: The Rayleigh family to spread.
        velocity_factors: Positive phase-velocity multipliers (UNANCHORED).
        weights: Non-negative power weights, one per factor, not all zero (UNANCHORED).

    Returns:
        The tuple of Rayleigh families.
    """
    factors = np.asarray(velocity_factors, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if factors.ndim != 1 or factors.size == 0 or factors.shape != weight_array.shape:
        raise ValueError("velocity_factors and weights must be non-empty 1-D sequences of equal length.")
    if np.any(factors <= 0.0) or np.any(weight_array < 0.0) or np.sum(weight_array) <= 0.0:
        raise ValueError("velocity factors must be positive and weights non-negative with a positive sum.")
    fractions = weight_array / np.sum(weight_array)
    field = family.field
    copies = []
    for factor, fraction in zip(factors, fractions, strict=True):

        def velocity(frequencies: NDArray[np.float64], factor: float = float(factor)) -> NDArray[np.float64]:
            return factor * field.phase_velocity_at(frequencies)

        def psd(frequencies: NDArray[np.float64], fraction: float = float(fraction)) -> NDArray[np.float64]:
            return fraction * np.asarray(field.surface_displacement_psd(frequencies), dtype=float)

        copies.append(replace(family, field=replace(field, phase_velocity=velocity, surface_displacement_psd=psd)))
    return tuple(copies)


__all__ = ["RayleighFamily", "phase_velocity_spread"]
