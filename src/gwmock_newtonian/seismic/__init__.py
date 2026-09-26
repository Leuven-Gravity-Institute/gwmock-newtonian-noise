"""Minimal stationary homogeneous Rayleigh-wave seismic field.

This subpackage implements the field itself -- medium, dispersion, depth
eigenfunctions, directional/power spectra, complex cross-spectral density and
seeded realizations -- for analytic Newtonian-noise screening. It does not
implement detector, test-mass or witness response (those live in
:mod:`gwmock_newtonian.detector` and :mod:`gwmock_newtonian.witness`) and it fixes no "realistic
ET" parameter set: density, dispersion, anisotropy, depth and spectra are all
caller-supplied configuration.
"""

from __future__ import annotations

from gwmock_newtonian.seismic.depth import (
    RayleighDepthRates,
    depth_rates,
    dilatation_shape,
    horizontal_shape,
    vertical_shape,
)
from gwmock_newtonian.seismic.field import (
    RayleighWaveFieldConfig,
    analytic_covariance,
    analytic_density_covariance,
    realize,
    steering_vectors,
)
from gwmock_newtonian.seismic.medium import (
    ElasticHalfSpace,
    rayleigh_secular_cubic_coefficients,
    rayleigh_velocity_ratio,
)
from gwmock_newtonian.seismic.spectrum import power_law_psd, uniform_directional_density, von_mises_directional_density

__all__ = [
    "ElasticHalfSpace",
    "RayleighDepthRates",
    "RayleighWaveFieldConfig",
    "analytic_covariance",
    "analytic_density_covariance",
    "depth_rates",
    "dilatation_shape",
    "horizontal_shape",
    "power_law_psd",
    "rayleigh_secular_cubic_coefficients",
    "rayleigh_velocity_ratio",
    "realize",
    "steering_vectors",
    "uniform_directional_density",
    "vertical_shape",
    "von_mises_directional_density",
]
