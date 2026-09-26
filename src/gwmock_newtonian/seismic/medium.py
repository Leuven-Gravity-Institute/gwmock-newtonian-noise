"""Homogeneous isotropic elastic half-space medium and the Rayleigh dispersion relation.

The governing physics is anchored to primary sources. The free-surface
Rayleigh-wave problem and its dispersion relation are from Rayleigh, Lord
(J. W. Strutt) (1885), "On Waves Propagated along the Plane Surface of an
Elastic Solid," Proceedings of the London Mathematical Society, s1-17(1),
4-11. The dilatational/shear potential representation of the same
free-surface boundary-value problem, and the reduction of the secular
relation to a cubic in the squared sub-shear speed ratio ``x = (c_R/beta)**2``,
are from Lamb, H. (1904), "On the Propagation of Tremors over the Surface of
an Elastic Solid," Philosophical Transactions of the Royal Society A, 203,
1-42 -- the primary treatment of the relation solved here.

What is *not* primary is the modern normalized notation and the particular
grouping of the equation. The potential convention and the normalized
``r = sqrt(1 - x*kappa**2)``, ``s = sqrt(1 - x)`` spelling follow the
textbook presentation of Aki, K. and Richards, P. G. (2002), *Quantitative
Seismology*, 2nd ed., University Science Books, Sec. 5.7 and Sec. 7.2; the
algebraic route to the cubic (squaring the secular equation, which admits
the spurious ``x = 0`` root, then removing that factor) follows Achenbach,
J. D. (1973), *Wave Propagation in Elastic Solids*, North-Holland, Sec. 6.2.
Those two citations are conventions, not primary derivations: the relation
they express is the one given by Rayleigh (1885) and Lamb (1904).

Density and the P-/S-wave speeds are free configuration parameters: this
module fixes no "representative ET site" values. Any concrete numeric value used in tests or
examples elsewhere in this package is tagged UNANCHORED at its point of use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Absolute tolerance used to decide whether a cubic root's imaginary part is
#: numerical noise (and the root therefore real) relative to its real part.
_REAL_ROOT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ElasticHalfSpace:
    """A homogeneous, isotropic elastic half-space: ``z >= 0`` depth positive downward, free surface at ``z = 0``.

    Attributes:
        p_wave_speed: Compressional (P) wave speed alpha, in meters/second.
            Must exceed ``s_wave_speed`` (elastic stability of an isotropic
            solid requires a Poisson ratio below 0.5).
        s_wave_speed: Shear (S) wave speed beta, in meters/second, must be
            positive.
        density: Background mass density rho0, in kilograms/meter^3, must be
            positive.
    """

    p_wave_speed: float
    s_wave_speed: float
    density: float

    def __post_init__(self) -> None:
        """Reject a non-physical half-space."""
        if not (self.s_wave_speed > 0.0):
            raise ValueError(f"s_wave_speed must be positive, got {self.s_wave_speed}.")
        if not (self.p_wave_speed > self.s_wave_speed):
            raise ValueError(
                "p_wave_speed must exceed s_wave_speed for a stable isotropic elastic solid "
                f"(got p_wave_speed={self.p_wave_speed}, s_wave_speed={self.s_wave_speed})."
            )
        if not (self.density > 0.0):
            raise ValueError(f"density must be positive, got {self.density}.")

    @property
    def velocity_ratio(self) -> float:
        """Return kappa = s_wave_speed / p_wave_speed, in (0, 1)."""
        return self.s_wave_speed / self.p_wave_speed

    @property
    def lame_mu(self) -> float:
        """Return the shear modulus mu = rho0 * beta**2, in pascals."""
        return self.density * self.s_wave_speed**2

    @property
    def lame_lambda(self) -> float:
        """Return the first Lame parameter, lambda = rho0 * (alpha**2 - 2 * beta**2), in pascals."""
        return self.density * (self.p_wave_speed**2 - 2.0 * self.s_wave_speed**2)

    @property
    def rayleigh_phase_velocity(self) -> float:
        """Return the non-dispersive fundamental-mode Rayleigh phase velocity c_R, in meters/second.

        A homogeneous half-space is non-dispersive: this single velocity
        applies at every frequency. Callers that want frequency-dependent
        dispersion supply their own ``phase_velocity`` override to
        :class:`~gwmock_newtonian.seismic.field.RayleighWaveFieldConfig`
        instead of using this property.
        """
        x = rayleigh_velocity_ratio(self.velocity_ratio)
        return float(np.sqrt(x)) * self.s_wave_speed


def rayleigh_secular_cubic_coefficients(kappa: float) -> np.ndarray:
    """Return the coefficients of the Rayleigh secular cubic in ``x = (c_R / beta) ** 2``.

    Squaring the free-surface secular equation ``(2 - x) ** 2 = 4 * r * s``,
    with ``r = sqrt(1 - x * kappa ** 2)`` and ``s = sqrt(1 - x)``, gives a
    quartic in ``x`` that factors as ``x`` times this cubic (the ``x = 0``
    factor is the spurious static root, ``c = 0``):

        x**3 - 8*x**2 + (24 - 16*kappa**2)*x - 16*(1 - kappa**2) = 0

    Args:
        kappa: Shear-to-compressional speed ratio, ``beta / alpha``, in
            ``(0, 1)``.

    Returns:
        The four cubic coefficients, highest degree first, suitable for
        :func:`numpy.roots`.
    """
    if not (0.0 < kappa < 1.0):
        raise ValueError(f"kappa = s_wave_speed / p_wave_speed must lie in (0, 1), got {kappa}.")
    kappa2 = kappa**2
    return np.array([1.0, -8.0, 24.0 - 16.0 * kappa2, -16.0 * (1.0 - kappa2)])


def rayleigh_velocity_ratio(kappa: float) -> float:
    """Solve the Rayleigh secular equation for ``x = (c_R / beta) ** 2``.

    Finds the unique root of :func:`rayleigh_secular_cubic_coefficients` in
    ``(0, 1)`` and confirms it also satisfies the *unsquared* secular
    equation ``(2 - x) ** 2 = 4 * r * s`` (the cubic comes from squaring,
    which admits a spurious sign branch that the unsquared check rejects).

    Args:
        kappa: Shear-to-compressional speed ratio, ``beta / alpha``, in
            ``(0, 1)``.

    Returns:
        ``x = (c_R / beta) ** 2``, in ``(0, 1)``.
    """
    kappa2 = kappa**2
    roots = np.roots(rayleigh_secular_cubic_coefficients(kappa))
    real_mask = np.abs(roots.imag) < _REAL_ROOT_TOLERANCE * np.maximum(np.abs(roots.real), 1.0)
    real_roots = roots[real_mask].real
    physical = real_roots[(real_roots > 0.0) & (real_roots < 1.0)]
    if physical.size != 1:
        raise ValueError(
            f"Expected exactly one physical Rayleigh root in (0, 1) for kappa={kappa}, "
            f"found {physical.size}: {physical}."
        )
    x = float(physical[0])
    r = np.sqrt(1.0 - x * kappa2)
    s = np.sqrt(1.0 - x)
    if not np.isclose((2.0 - x) ** 2, 4.0 * r * s, rtol=1e-8):
        raise ValueError(f"Root x={x} of the squared secular equation does not satisfy the unsquared equation.")
    return x


__all__ = ["ElasticHalfSpace", "rayleigh_secular_cubic_coefficients", "rayleigh_velocity_ratio"]
