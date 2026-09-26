"""Configurable directional and power spectra for the Rayleigh-wave field.

Two families are provided, both standard, citable closed forms rather than a
fitted "realistic ET" spectrum:

- Directional density over azimuth: :func:`uniform_directional_density`
  (isotropic) and :func:`von_mises_directional_density` (a single-lobe
  circular distribution; its concentration parameter ``kappa`` tunes
  anisotropy continuously from isotropic, ``kappa = 0``, to a near-plane-wave
  beam, large ``kappa``). The von Mises density and its Fourier
  (Jacobi-Anger) coefficients ``I_m(kappa) / I_0(kappa)`` are standard
  results in circular statistics; see Mardia, K. V. and Jupp, P. E. (2000),
  *Directional Statistics*, Wiley, Sec. 3.5.4.
- Power spectrum over frequency: :func:`power_law_psd`, a generic power-law
  one-sided displacement PSD. No numeric default is supplied: any concrete
  ``(reference_frequency, reference_psd, exponent)`` used elsewhere in this
  package (tests, examples) is UNANCHORED -- illustrative only, not fit to a
  measured site spectrum. A real fit would use a measured reference such as
  Peterson, J. R. (1993), "Observations and Modeling of Seismic Background
  Noise," U.S. Geological Survey Open-File Report 93-322, which is not
  performed here.

Neither family is privileged by the field API in
:mod:`gwmock_newtonian.seismic.field`: it accepts any ``directional_density``
and ``surface_displacement_psd`` callable with the documented signature, so a
caller can supply an arbitrary configured shape instead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import i0


def uniform_directional_density(phi: np.ndarray) -> np.ndarray:
    """Return the isotropic directional density, uniform over azimuth.

    Args:
        phi: Azimuth angle(s), in radians.

    Returns:
        ``1 / (2 * pi)`` at every input angle, same shape as ``phi``.
    """
    phi = np.asarray(phi, dtype=float)
    return np.full(phi.shape, 1.0 / (2.0 * np.pi))


def von_mises_directional_density(mu: float, kappa: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a von Mises directional density callable, ``exp(kappa*cos(phi-mu)) / (2*pi*I0(kappa))``.

    ``kappa = 0`` reduces exactly to :func:`uniform_directional_density`;
    increasing ``kappa`` concentrates propagation increasingly tightly around
    azimuth ``mu``.

    Args:
        mu: Preferred propagation azimuth, in radians.
        kappa: Concentration parameter, must be non-negative.

    Returns:
        A callable mapping an azimuth array to the (already-normalized)
        density array.
    """
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa}.")
    normalization = 2.0 * np.pi * i0(kappa)

    def _density(phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        return np.exp(kappa * np.cos(phi - mu)) / normalization

    return _density


def power_law_psd(
    reference_frequency: float, reference_psd: float, exponent: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a power-law one-sided displacement PSD callable.

    ``S(f) = reference_psd * (f / reference_frequency) ** (-exponent)``.

    UNANCHORED: this functional shape, and any numeric
    ``(reference_frequency, reference_psd, exponent)`` triple passed to it
    elsewhere in this package, is illustrative only -- not fit to any
    measured seismic spectrum. See the module docstring.

    Args:
        reference_frequency: Reference frequency, in hertz, must be positive.
        reference_psd: One-sided PSD value at ``reference_frequency``, in
            meters^2/hertz, must be non-negative.
        exponent: Spectral index (PSD falls as ``f ** (-exponent)``).

    Returns:
        A callable mapping a strictly-positive frequency array to the PSD
        array, in meters^2/hertz.
    """
    if reference_frequency <= 0.0:
        raise ValueError(f"reference_frequency must be positive, got {reference_frequency}.")
    if reference_psd < 0.0:
        raise ValueError(f"reference_psd must be non-negative, got {reference_psd}.")

    def _psd(frequencies: np.ndarray) -> np.ndarray:
        frequencies = np.asarray(frequencies, dtype=float)
        if np.any(frequencies <= 0.0):
            raise ValueError("power_law_psd is only defined for strictly positive frequencies.")
        return reference_psd * (frequencies / reference_frequency) ** (-exponent)

    return _psd


__all__ = ["power_law_psd", "uniform_directional_density", "von_mises_directional_density"]
