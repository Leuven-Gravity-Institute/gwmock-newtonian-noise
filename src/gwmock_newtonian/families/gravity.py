"""Displacement and Newtonian acceleration of elastic plane-wave components.

A body-wave or guided-wave mode is a finite sum of plane-wave *components*
``u(r) = c * e * exp(i K . r)`` with a complex amplitude ``c``, a (possibly
complex) polarization ``e`` and a wavevector ``K`` whose horizontal part is real
and whose vertical part may be complex (an evanescent component). This module
evaluates the displacement of such a sum and the gravity perturbation it sources
in two independent geometries.

Governing equations (the package's, see :mod:`gwmock_newtonian.detector.newtonian`):
``delta_rho = -rho0 div(u)`` in the bulk, ``del^2 Phi = 4 pi G delta_rho`` and
``a = -grad Phi``. For one component ``div(u) = i (K . e) c exp(i K . r)``, so a
transverse (shear) component sources no bulk density at all.

**Full space** (no free surface; the infinite-medium limit). Solving Poisson's
equation for one component gives the exact local relation

    a(r) = 4 pi G rho0 c (K . e) K / (K . K) exp(i K . r),

which for a longitudinal (P) component, ``e = K / sqrt(K . K)``, is
``a = 4 pi G rho0 u_P`` and for a shear component is zero. The same relation
holds for any irrotational displacement ``u = grad(psi)`` decaying at infinity
(``Phi = -4 pi G rho0 psi``), which the point-source family uses. The
coefficient ``4 pi G rho0`` is the infinite-medium P-wave coupling of Harms, J.
(2019), "Terrestrial Gravity Fluctuations," Living Reviews in Relativity, 22, 6,
DOI 10.1007/s41114-019-0022-2, and of Badaracco, F. and Harms, J. (2019), "Optimization of seismometer arrays
for the cancellation of Newtonian noise from seismic body waves," Classical and
Quantum Gravity, DOI 10.1088/1361-6382/ab28c1, arXiv:1903.07936. Its sign here follows from the
package's own ``delta_rho`` and ``a = -grad Phi`` conventions and is checked in
the test suite against a direct real-space Green's-function sum, not against a
quoted sign.

**Half space** ``z >= 0`` with a free surface at ``z = 0``. Each component's
density is a horizontal grating ``A(z) exp(i k_h . r_h)`` with
``A(z) = -i rho0 (K . e) c exp(i q z)``, ``q = K_z``. The horizontal integral of
the Newtonian kernel reduces exactly as in
:func:`~gwmock_newtonian.detector.newtonian.plane_wave_acceleration`; here the
depth overlap integrals of the exponential profile are done in closed form,

    I1 = A0 [ (e^{iqz0} - e^{-k z0}) / (k + i q) + e^{iqz0} / (k - i q) ],
    I2 = A0 [ -(e^{iqz0} - e^{-k z0}) / (k + i q) + e^{iqz0} / (k - i q) ],

with ``k = |k_h| > 0``, and the free-surface areal term
``delta_sigma = -rho0 xi_z`` uses the component's surface vertical displacement
``xi_z = c e_z``. The closed form is checked in the test suite against the
package's independent numerical depth quadrature
(:func:`~gwmock_newtonian.detector.newtonian.plane_wave_acceleration`), and its
deep limit against the full-space relation.

Neither geometry includes a cavity around the evaluation point; that term is
applied by :mod:`gwmock_newtonian.families.response`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.detector.newtonian import GRAVITATIONAL_CONSTANT

#: Smallest admissible horizontal wavenumber, relative to ``|K|``, for the half-space
#: kernel: an exactly vertical component has no horizontal direction and the
#: Fourier-reduced kernel is singular there. Midpoint direction grids never hit it.
_MIN_RELATIVE_HORIZONTAL_WAVENUMBER = 1e-9


@dataclass(frozen=True)
class PlaneWaveComponent:
    """One elastic plane-wave component ``u(r) = amplitude * polarization * exp(i K . r)``.

    Attributes:
        wavevector: Complex ``(3,)`` wavevector ``K`` in radians/metre; its
            horizontal part must be real.
        polarization: Complex ``(3,)`` polarization ``e`` (dimensionless).
        amplitude: Complex amplitude ``c`` per unit mode reference amplitude.
    """

    wavevector: NDArray[np.complex128]
    polarization: NDArray[np.complex128]
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        """Reject a malformed component."""
        if np.shape(self.wavevector) != (3,) or np.shape(self.polarization) != (3,):
            raise ValueError("wavevector and polarization must both have shape (3,).")
        if np.any(np.abs(np.imag(self.wavevector[:2])) > 0.0):
            raise ValueError("the horizontal wavevector of a plane-wave component must be real.")

    @property
    def divergence_factor(self) -> complex:
        """Return ``K . e``: ``div(u) = i (K . e) c exp(i K . r)``."""
        return complex(np.sum(self.wavevector * self.polarization))


def component_displacement(
    components: Sequence[PlaneWaveComponent], positions: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Return the ``(n_positions, 3)`` displacement of a sum of plane-wave components."""
    total = np.zeros((positions.shape[0], 3), dtype=complex)
    for component in components:
        phase = np.exp(1j * (positions @ component.wavevector))
        total += component.amplitude * phase[:, None] * component.polarization[None, :]
    return total


def full_space_acceleration(
    components: Sequence[PlaneWaveComponent],
    positions: NDArray[np.float64],
    density: float,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> NDArray[np.complex128]:
    """Return the ``(n_positions, 3)`` infinite-medium Newtonian acceleration of the components.

    Uses ``a = 4 pi G rho0 c (K . e) K / (K . K) exp(i K . r)`` per component (see
    the module docstring); shear components contribute exactly zero.
    """
    total = np.zeros((positions.shape[0], 3), dtype=complex)
    for component in components:
        k_dot_k = complex(np.sum(component.wavevector * component.wavevector))
        if k_dot_k == 0.0:
            raise ValueError("a plane-wave component must have a non-null wavevector (K . K != 0).")
        coupling = (
            4.0 * np.pi * gravitational_constant * density * component.amplitude * component.divergence_factor / k_dot_k
        )
        phase = np.exp(1j * (positions @ component.wavevector))
        total += coupling * phase[:, None] * component.wavevector[None, :]
    return total


def half_space_acceleration(
    components: Sequence[PlaneWaveComponent],
    positions: NDArray[np.float64],
    density: float,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> NDArray[np.complex128]:
    """Return the ``(n_positions, 3)`` half-space Newtonian acceleration, bulk plus free-surface areal.

    Every position must lie in the medium (``z >= 0``) and every component must
    have a non-zero horizontal wavenumber; see the module docstring for the
    closed form.
    """
    if np.any(positions[:, 2] < 0.0):
        raise ValueError("half-space positions must have z >= 0 (z is depth, positive downward).")
    z0 = positions[:, 2]
    total = np.zeros((positions.shape[0], 3), dtype=complex)
    for component in components:
        k_vec = np.real(component.wavevector[:2])
        k = float(np.hypot(*k_vec))
        if k <= _MIN_RELATIVE_HORIZONTAL_WAVENUMBER * float(np.linalg.norm(component.wavevector)):
            raise ValueError("the half-space kernel needs a non-vertical component (non-zero horizontal wavenumber).")
        k_hat = k_vec / k
        q = complex(component.wavevector[2])
        source = -1j * density * component.divergence_factor * component.amplitude
        surface_down = component.amplitude * complex(component.polarization[2])
        decay = np.exp(-k * z0)
        oscillation = np.exp(1j * q * z0)
        upper = (oscillation - decay) / (k + 1j * q)
        lower = oscillation / (k - 1j * q)
        smooth = source * (upper + lower)
        signed = source * (lower - upper)
        areal = density * surface_down * decay
        prefactor = 2.0 * np.pi * gravitational_constant * np.exp(1j * (positions[:, :2] @ k_vec))
        total[:, 0] += 1j * prefactor * k_hat[0] * (smooth - areal)
        total[:, 1] += 1j * prefactor * k_hat[1] * (smooth - areal)
        total[:, 2] += prefactor * (signed + areal)
    return total


__all__ = [
    "PlaneWaveComponent",
    "component_displacement",
    "full_space_acceleration",
    "half_space_acceleration",
]
