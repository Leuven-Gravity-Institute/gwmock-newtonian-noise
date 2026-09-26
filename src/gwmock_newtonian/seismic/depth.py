"""Depth eigenfunctions of the fundamental Rayleigh mode in a homogeneous half-space.

Derived from the free-surface elastodynamic boundary-value problem: a
dilatational (P) potential ``phi = exp(-k*r*z) * exp(i*(k*x - w*t))`` and a
shear (SV) potential ``psi = B * exp(-k*s*z) * exp(i*(k*x - w*t))``, each
individually satisfying the elastic wave equation, combined through
``u = grad(phi) + curl(psi * y_hat)`` and required to leave the free surface
``z = 0`` traction-free (``sigma_zz = sigma_xz = 0``). Solving those two
boundary conditions for ``B`` and for the phase velocity ``c = w / k``
reproduces the Rayleigh (1885) secular equation implemented in
:mod:`gwmock_newtonian.seismic.medium`; substituting the resulting ``B`` back
into ``u`` gives the closed-form depth shapes below.

The potential representation of the free-surface problem and the secular
relation are primary-sourced: Rayleigh, Lord (J. W. Strutt) (1885), "On Waves
Propagated along the Plane Surface of an Elastic Solid," Proceedings of the
London Mathematical Society, s1-17(1), 4-11, for the original surface-wave
problem, and Lamb, H. (1904), "On the Propagation of Tremors over the Surface
of an Elastic Solid," Philosophical Transactions of the Royal Society A, 203,
1-42, for the dilatational/shear potential treatment of the half-space
boundary-value problem and the resulting secular relation. The *notation*
-- in particular writing the potentials as ``phi``/``psi`` with this sign and
decay convention -- is a textbook convention, not a primary-source result;
it follows Aki, K. and Richards, P. G. (2002), *Quantitative Seismology*,
2nd ed., University Science Books, Sec. 7.2, with the same disclaimer recorded
in :mod:`gwmock_newtonian.seismic.medium`.

These closed forms are independently checked in the test suite against
Newton's second law (the elastodynamic Navier-Cauchy equation) and against
the traction-free boundary condition, evaluated directly from Hooke's law and
numerical differentiation of the displacement field -- not by re-deriving
this module's own algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gwmock_newtonian.seismic.medium import ElasticHalfSpace


@dataclass(frozen=True)
class RayleighDepthRates:
    """Vertical-decay parameters of one Rayleigh mode at a fixed angular frequency.

    Attributes:
        k: Horizontal wavenumber, ``omega / c``, in radians/meter.
        c: Phase velocity, in meters/second.
        r: Dimensionless P-potential decay parameter, ``sqrt(1 - (c/alpha)**2)``.
        s: Dimensionless S-potential decay parameter, ``sqrt(1 - (c/beta)**2)``.
    """

    k: float
    c: float
    r: float
    s: float

    @property
    def q_alpha(self) -> float:
        """Vertical wavenumber of the P potential, ``k * r``, in radians/meter."""
        return self.k * self.r

    @property
    def q_beta(self) -> float:
        """Vertical wavenumber of the S potential, ``k * s``, in radians/meter."""
        return self.k * self.s


def depth_rates(medium: ElasticHalfSpace, angular_frequency: float, phase_velocity: float) -> RayleighDepthRates:
    """Return the vertical-decay parameters of a surface-guided mode at one angular frequency.

    Args:
        medium: The homogeneous elastic half-space.
        angular_frequency: Angular frequency omega = 2*pi*f, in radians/second, must be positive.
        phase_velocity: Horizontal phase velocity c = omega / k, in meters/second. Must lie in
            ``(0, medium.s_wave_speed)`` for the mode to be evanescent (surface-guided) in both
            potentials; the non-dispersive default is ``medium.rayleigh_phase_velocity``, but a
            caller-supplied dispersive ``phase_velocity(f)`` is equally valid as long as it stays
            below the shear speed.

    Returns:
        The mode's :class:`RayleighDepthRates`.
    """
    if angular_frequency <= 0.0:
        raise ValueError(f"angular_frequency must be positive, got {angular_frequency}.")
    if not (0.0 < phase_velocity < medium.s_wave_speed):
        raise ValueError(
            "phase_velocity must lie in (0, s_wave_speed) for an evanescent surface-guided mode "
            f"(got phase_velocity={phase_velocity}, s_wave_speed={medium.s_wave_speed})."
        )
    k = angular_frequency / phase_velocity
    r = float(np.sqrt(1.0 - (phase_velocity / medium.p_wave_speed) ** 2))
    s = float(np.sqrt(1.0 - (phase_velocity / medium.s_wave_speed) ** 2))
    return RayleighDepthRates(k=k, c=phase_velocity, r=r, s=s)


def horizontal_shape(z: np.ndarray | float, rates: RayleighDepthRates) -> np.ndarray:
    """Dimensionless horizontal (propagation-direction) displacement depth shape.

    The displacement along the propagation direction is
    ``u_along(z) = 1j * rates.k * horizontal_shape(z, rates) * Phi`` for a
    potential-amplitude mode coefficient ``Phi`` (see
    :mod:`gwmock_newtonian.seismic.field` for the physically-normalized
    surface-displacement parameterization used by the public field API).

    Args:
        z: Depth(s), in meters, positive downward from the free surface.
        rates: The mode's :class:`RayleighDepthRates`.

    Returns:
        The dimensionless shape value(s), same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    r, s, k = rates.r, rates.s, rates.k
    return np.exp(-k * r * z) - 0.5 * (1.0 + s**2) * np.exp(-k * s * z)


def vertical_shape(z: np.ndarray | float, rates: RayleighDepthRates) -> np.ndarray:
    """Dimensionless vertical displacement depth shape.

    ``u_z(z) = -rates.k * vertical_shape(z, rates) * Phi`` for potential
    amplitude ``Phi`` (see :func:`horizontal_shape`).

    Args:
        z: Depth(s), in meters, positive downward from the free surface.
        rates: The mode's :class:`RayleighDepthRates`.

    Returns:
        The dimensionless shape value(s), same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    r, s, k = rates.r, rates.s, rates.k
    return r * np.exp(-k * r * z) - (1.0 + s**2) / (2.0 * s) * np.exp(-k * s * z)


def dilatation_shape(z: np.ndarray | float, rates: RayleighDepthRates) -> np.ndarray:
    """Dimensionless dilatation (volumetric strain, ``div(u)``) depth shape.

    ``div(u)(z) = dilatation_shape(z, rates) * Phi``. The shear potential is
    solenoidal (``div(curl(...)) = 0``), so only the P-potential's depth
    decay ``exp(-k*r*z)`` survives here -- an exact identity
    (``div(u) = laplacian(phi) = -(omega/alpha)**2 * phi``) independent of
    the free-surface boundary conditions used to fix ``r`` and ``s``, and a
    useful cross-check on this module's algebra.

    Args:
        z: Depth(s), in meters, positive downward from the free surface.
        rates: The mode's :class:`RayleighDepthRates`.

    Returns:
        The dimensionless shape value(s), same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    r, k = rates.r, rates.k
    return -(k**2) * (1.0 - r**2) * np.exp(-k * r * z)


__all__ = ["RayleighDepthRates", "depth_rates", "dilatation_shape", "horizontal_shape", "vertical_shape"]
