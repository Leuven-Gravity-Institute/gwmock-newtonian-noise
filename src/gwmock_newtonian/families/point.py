"""Localized sources: independent isotropic (explosion-type) P-wave point sources.

Each source sits at a declared three-dimensional position and radiates a
spherical compressional wave into a homogeneous infinite medium. Its
displacement is the gradient of the outgoing monopole potential,

    u(r) = U * grad( e^{i k R} / R ) / g(R_ref),     R = |r - r_s|,  k = omega / alpha,
    grad( e^{i k R} / R ) = R_hat e^{i k R} (i k / R - 1 / R^2),

normalized by ``g(R_ref) = e^{i k R_ref} (i k / R_ref - 1 / R_ref^2)`` so the mode
reference amplitude ``U`` is the radial displacement at the declared reference
distance ``R_ref``. This is the far-field-plus-near-field P radiation of an
isotropic moment source (Aki, K. and Richards, P. G. (2002), *Quantitative
Seismology*, 2nd ed., Sec. 4.2), and the source spectrum is a caller-supplied
PSD of ``U``. The near field (``1/R^2``) is retained, so positions close to a
source see a field that is neither planar nor locally homogeneous -- the point of
this family as an out-of-family test.

Newtonian coupling. The displacement is irrotational, ``u = grad(psi)``, and
decays at infinity, so ``Phi = -4 pi G rho0 psi`` solves Poisson's equation with
``delta_rho = -rho0 div(u)`` exactly -- including the source point itself, where
``div(u)`` carries the delta function of the injected volume -- and the
acceleration is the exact local relation ``a = 4 pi G rho0 u`` (see
:mod:`gwmock_newtonian.families.gravity`). The test suite checks it against
Gauss's law on spheres around the source, with the enclosed mass integrated
independently.

What is not modelled: a free surface (no image sources or surface waves; this is
the infinite-medium limit), shear radiation (force or double-couple sources),
finite source size and attenuation. Evaluation points closer to a source than
``min_distance`` are rejected.

UNANCHORED: source positions, spectra and the medium are site parameters with
no default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian.families.base import PSDCallable, as_positions, evaluate_psd, require_positive_frequency
from gwmock_newtonian.seismic.medium import ElasticHalfSpace


@dataclass(frozen=True)
class PointSource:
    """One isotropic P-wave point source.

    Attributes:
        position: Source position ``(x, y, z)`` in metres (UNANCHORED).
        displacement_psd: One-sided PSD of the radial displacement at
            ``reference_distance``, m^2/Hz (UNANCHORED).
        reference_distance: Distance at which the amplitude is defined, metres, positive.
    """

    position: tuple[float, float, float]
    displacement_psd: PSDCallable
    reference_distance: float

    def __post_init__(self) -> None:
        """Reject a malformed source."""
        if len(self.position) != 3 or not np.all(np.isfinite(self.position)):  # noqa: PLR2004 -- 3-D position.
            raise ValueError(f"position must be a finite 3-vector, got {self.position}.")
        if not (np.isfinite(self.reference_distance) and self.reference_distance > 0.0):
            raise ValueError(f"reference_distance must be positive, got {self.reference_distance}.")


def _radial_profile(distance: NDArray[np.float64], wavenumber: float) -> NDArray[np.complex128]:
    """Return ``e^{i k R} (i k / R - 1 / R^2)``, the radial derivative of the monopole potential."""
    return np.exp(1j * wavenumber * distance) * (1j * wavenumber / distance - 1.0 / distance**2)


@dataclass(frozen=True)
class PointSourceFamily:
    """Independent isotropic P-wave point sources in an infinite medium; one mode per source.

    Attributes:
        medium: The homogeneous medium (UNANCHORED); only ``p_wave_speed`` and
            ``density`` enter.
        sources: The point sources (UNANCHORED), at least one.
        min_distance: Smallest admissible source-to-position distance in metres;
            the near field diverges as ``1/R^2``.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]] = ("medium", "sources")

    medium: ElasticHalfSpace
    sources: tuple[PointSource, ...]
    min_distance: float = 1.0

    def __post_init__(self) -> None:
        """Reject an empty source list or a non-positive cut-off."""
        if len(self.sources) == 0:
            raise ValueError("a point-source family needs at least one source.")
        if not (self.min_distance > 0.0):
            raise ValueError(f"min_distance must be positive, got {self.min_distance}.")

    @property
    def n_modes(self) -> int:
        """Number of latent modes: one per source."""
        return len(self.sources)

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the ``(n_sources,)`` source PSDs at one frequency."""
        frequency = require_positive_frequency(frequency)
        return np.array(
            [evaluate_psd(source.displacement_psd, frequency, "displacement_psd") for source in self.sources]
        )

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_sources, n_positions, 3)`` displacement per unit reference amplitude."""
        positions = as_positions(positions)
        wavenumber = 2.0 * np.pi * require_positive_frequency(frequency) / self.medium.p_wave_speed
        result = np.empty((self.n_modes, positions.shape[0], 3), dtype=complex)
        for index, source in enumerate(self.sources):
            offset = positions - np.asarray(source.position, dtype=float)[None, :]
            distance = np.linalg.norm(offset, axis=1)
            if np.any(distance < self.min_distance):
                raise ValueError(f"a position lies within min_distance={self.min_distance} m of a point source.")
            normalization = _radial_profile(np.array([source.reference_distance]), wavenumber)[0]
            radial = _radial_profile(distance, wavenumber) / normalization
            result[index] = radial[:, None] * offset / distance[:, None]
        return result

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the exact infinite-medium Newtonian acceleration ``4 pi G rho0 u``."""
        return 4.0 * np.pi * GRAVITATIONAL_CONSTANT * self.medium.density * self.displacement(positions, frequency)

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the (uniform) background density at each position."""
        return np.full(as_positions(positions).shape[0], self.medium.density)


def point_sources(
    medium: ElasticHalfSpace,
    positions: Sequence[tuple[float, float, float]],
    displacement_psd: PSDCallable,
    reference_distance: float,
) -> PointSourceFamily:
    """Return a family of independent sources at ``positions`` sharing one spectrum.

    Args:
        medium: The homogeneous medium (UNANCHORED).
        positions: Source positions (UNANCHORED).
        displacement_psd: Shared one-sided PSD at ``reference_distance`` (UNANCHORED).
        reference_distance: Reference distance in metres.

    Returns:
        The :class:`PointSourceFamily`.
    """
    return PointSourceFamily(
        medium=medium,
        sources=tuple(PointSource(tuple(map(float, p)), displacement_psd, reference_distance) for p in positions),
    )


__all__ = ["PointSource", "PointSourceFamily", "point_sources"]
