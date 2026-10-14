"""The field-family contract shared by every Newtonian-noise source family.

Every family in this subpackage describes a stationary Gaussian seismic field as
a finite set of *independent latent modes* at each frequency. Mode ``m`` at
frequency ``f`` carries

- a non-negative one-sided power ``P[m]`` (metres^2/hertz): the PSD of the
  mode's complex reference amplitude, already multiplied by its quadrature
  weight, so the field's covariance is the plain sum over modes; and
- two deterministic linear responses per unit reference amplitude: the ground
  displacement ``u[m]`` (metres per metre) at any requested position, and the
  Newtonian (gravity-perturbation) acceleration ``a[m]`` (metres/second^2 per
  metre) at any requested position.

That is exactly the structure the minimal Rayleigh field already has (its modes
are the azimuth quadrature cells), so one generic consumer
(:mod:`gwmock_newtonian.families.response`) turns any family into strain and
witness responses, a Hermitian positive-semidefinite joint covariance and seeded
realizations, and independent families combine by concatenating their modes
(:mod:`gwmock_newtonian.families.mixture`).

Frame: ``(x, y, z)`` in metres with ``z`` positive *downward*; a family with a
free surface puts it at ``z = 0``. Time convention ``exp(-i * omega * t)``, as
in the rest of the package.

Site parameters: every family lists the configuration fields that describe a
site (medium, layering, spectra, source positions, ...) in its
``UNANCHORED_PARAMETERS`` class attribute. None of them has a default value:
this package fixes no Einstein Telescope site model, and the test suite checks
that no such field acquires a default.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: Signature of a one-sided PSD callable: strictly positive frequencies (hertz) to
#: non-negative PSD values of the same shape.
PSDCallable = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: Signature of a directional density on the unit sphere: ``(m, 3)`` unit
#: propagation vectors (package frame, ``z`` down) to ``(m,)`` non-negative values.
SphereDensity = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: Fewest quadrature points per angular axis that can resolve a direction at all.
_MIN_ANGULAR_POINTS = 2


@runtime_checkable
class FieldFamily(Protocol):
    """A stationary Gaussian seismic field represented by independent latent modes.

    Implementations must return a mode count that does not depend on frequency
    (a mode with no power at some frequency reports ``0`` power there), so the
    responses of a family on a frequency grid stack into one array.
    """

    UNANCHORED_PARAMETERS: ClassVar[tuple[str, ...]]

    @property
    def n_modes(self) -> int:
        """Number of latent modes at every frequency."""
        ...

    def mode_power(self, frequency: float) -> NDArray[np.float64]:
        """Return the ``(n_modes,)`` non-negative one-sided mode powers, metres^2/hertz."""
        ...

    def displacement(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_modes, n_positions, 3)`` displacement per unit mode amplitude."""
        ...

    def acceleration(self, positions: NDArray[np.float64], frequency: float) -> NDArray[np.complex128]:
        """Return the ``(n_modes, n_positions, 3)`` Newtonian acceleration per unit mode amplitude.

        This is the gravity perturbation of the field itself, *without* any
        cavity around the evaluation point; the cavity-wall term is applied by
        the generic consumer (:class:`~gwmock_newtonian.families.response.CavityModel`).
        """
        ...

    def density_at(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the background density (kilograms/metre^3) of the medium at each position."""
        ...


def as_positions(positions: Sequence[Sequence[float]] | NDArray[np.floating]) -> NDArray[np.float64]:
    """Return ``positions`` as a validated ``(n, 3)`` float array."""
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:  # noqa: PLR2004 -- three spatial coordinates.
        raise ValueError(f"positions must have shape (n, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must be finite.")
    return array


def require_positive_frequency(frequency: float) -> float:
    """Return ``frequency`` as a float, rejecting a non-positive or non-finite value."""
    value = float(frequency)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"frequency must be positive and finite, got {frequency}.")
    return value


def evaluate_psd(psd: PSDCallable, frequency: float, label: str) -> float:
    """Return one non-negative PSD value of ``psd`` at ``frequency``."""
    value = float(np.asarray(psd(np.array([frequency], dtype=float)), dtype=float).reshape(-1)[0])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value} at f={frequency}.")
    return value


@dataclass(frozen=True)
class SphereGrid:
    """Equal-area midpoint quadrature over directions of propagation.

    Cells are midpoints of a uniform grid in ``cos(theta)`` and in azimuth
    ``phi``, so every cell subtends the same solid angle and the quadrature
    weights are just the normalized directional density. ``theta`` is measured
    from the ``+z`` (downward) axis: ``n = (sin t cos p, sin t sin p, cos t)``.

    Attributes:
        n_polar: Number of ``cos(theta)`` cells, at least 2.
        n_azimuth: Number of azimuth cells, at least 2.
        upgoing_only: If true, only directions with ``n_z < 0`` (propagating up
            toward a free surface at ``z = 0``) are gridded -- the incident
            hemisphere of a half-space. Otherwise the whole sphere is gridded.
    """

    n_polar: int
    n_azimuth: int
    upgoing_only: bool = False

    def __post_init__(self) -> None:
        """Reject a grid too coarse to resolve a direction."""
        if self.n_polar < _MIN_ANGULAR_POINTS or self.n_azimuth < _MIN_ANGULAR_POINTS:
            raise ValueError(
                f"n_polar and n_azimuth must be at least {_MIN_ANGULAR_POINTS}, got {self.n_polar}, {self.n_azimuth}."
            )

    @property
    def size(self) -> int:
        """Number of direction cells."""
        return self.n_polar * self.n_azimuth

    def directions(self) -> NDArray[np.float64]:
        """Return the ``(size, 3)`` unit propagation vectors, polar-major."""
        lower = -1.0
        upper = 0.0 if self.upgoing_only else 1.0
        cos_theta = lower + (np.arange(self.n_polar) + 0.5) * (upper - lower) / self.n_polar
        phi = (np.arange(self.n_azimuth) + 0.5) * (2.0 * np.pi / self.n_azimuth)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        ct, pp = np.meshgrid(cos_theta, phi, indexing="ij")
        st = np.broadcast_to(sin_theta[:, None], ct.shape)
        return np.stack([st * np.cos(pp), st * np.sin(pp), ct], axis=-1).reshape(-1, 3)

    def weights(self, density: SphereDensity) -> NDArray[np.float64]:
        """Return the normalized quadrature weights of ``density`` over the grid, summing to 1."""
        raw = np.asarray(density(self.directions()), dtype=float)
        if raw.shape != (self.size,):
            raise ValueError(f"directional density must return shape ({self.size},), got {raw.shape}.")
        if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("directional density must be finite and non-negative everywhere.")
        total = float(np.sum(raw))
        if total <= 0.0:
            raise ValueError("directional density must be positive somewhere on the grid.")
        return raw / total


def uniform_sphere_density(directions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the isotropic directional density ``1 / (4 pi)`` at every direction."""
    directions = np.asarray(directions, dtype=float)
    return np.full(directions.shape[0], 1.0 / (4.0 * np.pi))


def von_mises_fisher_density(mean_direction: Sequence[float], kappa: float) -> SphereDensity:
    """Return an (unnormalized) von Mises--Fisher density ``exp(kappa * (n . mu))`` on the sphere.

    The distribution is Fisher's: Fisher, R. A. (1953), "Dispersion on a
    sphere," Proceedings of the Royal Society A, 217(1130), 295-305. ``kappa = 0``
    is isotropic; large ``kappa`` concentrates the propagation directions around
    ``mean_direction``. The normalization is irrelevant here because
    :meth:`SphereGrid.weights` renormalizes; the maximum is scaled to 1 so a
    large ``kappa`` cannot overflow.

    Args:
        mean_direction: Preferred propagation direction (normalized internally).
        kappa: Concentration, non-negative.

    Returns:
        A density callable on ``(m, 3)`` unit vectors.
    """
    if not (np.isfinite(kappa) and kappa >= 0.0):
        raise ValueError(f"kappa must be finite and non-negative, got {kappa}.")
    mu = np.asarray(mean_direction, dtype=float)
    norm = float(np.linalg.norm(mu))
    if mu.shape != (3,) or norm == 0.0:
        raise ValueError(f"mean_direction must be a non-zero 3-vector, got {mean_direction}.")
    mu = mu / norm

    def _density(directions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(kappa * (np.asarray(directions, dtype=float) @ mu - 1.0))

    return _density


def horizontal_basis(azimuth: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the horizontal unit vectors ``(k_hat, t_hat)`` along and transverse to ``azimuth``.

    ``k_hat = (cos phi, sin phi, 0)`` and ``t_hat = (-sin phi, cos phi, 0)``; ``t_hat``
    is the SH / Love polarization of a wave travelling along ``k_hat``.
    """
    return (
        np.array([np.cos(azimuth), np.sin(azimuth), 0.0]),
        np.array([-np.sin(azimuth), np.cos(azimuth), 0.0]),
    )


__all__ = [
    "FieldFamily",
    "PSDCallable",
    "SphereDensity",
    "SphereGrid",
    "as_positions",
    "evaluate_psd",
    "horizontal_basis",
    "require_positive_frequency",
    "uniform_sphere_density",
    "von_mises_fisher_density",
]
