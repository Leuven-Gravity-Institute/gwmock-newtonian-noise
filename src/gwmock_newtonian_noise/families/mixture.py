"""Incoherent mixtures of independent field families.

Families in a mixture are statistically independent: their modes are simply
concatenated, so the mixture's covariance is the sum of the component
covariances and a realization draws an independent phase for every mode of every
component. Mixing is how a P/S/Rayleigh/Love composition, a set of local
sources over a background, or a velocity-spread proxy is assembled.

All components must describe the *same* background medium where the cavity term
needs it: :meth:`FamilyMixture.density_at` rejects positions at which the
components disagree on the density, rather than silently picking one (a mixture
of, say, a Love layer and a half-space Rayleigh field with a different density is
not one physical medium).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian_noise.families.base import FieldFamily, as_positions

#: Relative tolerance within which component densities must agree.
_DENSITY_RTOL = 1e-12


@dataclass(frozen=True)
class FamilyMixture:
    """An incoherent superposition of independent families; modes are concatenated in order.

    Attributes:
        components: The component families, at least one.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]] = ("components",)

    components: tuple[FieldFamily, ...]

    def __post_init__(self) -> None:
        """Reject an empty mixture."""
        if len(self.components) == 0:
            raise ValueError("a mixture needs at least one component family.")

    @property
    def n_modes(self) -> int:
        """Total number of latent modes across components."""
        return sum(component.n_modes for component in self.components)

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the concatenated mode powers."""
        return np.concatenate([component.mode_power(frequency) for component in self.components])

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the concatenated per-mode displacements."""
        return np.concatenate([component.displacement(positions, frequency) for component in self.components])

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the concatenated per-mode Newtonian accelerations."""
        return np.concatenate([component.acceleration(positions, frequency) for component in self.components])

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the common background density, rejecting components that disagree."""
        positions = as_positions(positions)
        densities = [np.asarray(component.density_at(positions), dtype=float) for component in self.components]
        for other in densities[1:]:
            if not np.allclose(other, densities[0], rtol=_DENSITY_RTOL, atol=0.0):
                raise ValueError("mixture components disagree on the background density at a requested position.")
        return densities[0]


__all__ = ["FamilyMixture"]
