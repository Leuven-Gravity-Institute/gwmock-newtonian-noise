"""ET triangle detector, test-mass and Newtonian-noise strain response.

Public subpackage implementing the detector physics: explicit arm vectors,
detector tensors, the ideal gravitational-wave response and its null vector,
physical test masses with shared-realization bookkeeping, the bulk plus
free-surface Newtonian acceleration, and the one-sided three-channel strain
covariance.
"""

from __future__ import annotations

from gwmock_newtonian.detector.geometry import (
    CANONICAL_DETECTOR_VERTICES,
    DetectorDefinition,
    TriangleGeometry,
    equilateral_triangle,
    rotation_matrix,
)
from gwmock_newtonian.detector.masses import (
    DetectorOptics,
    Suspension,
    TestMass,
    TestMassLayout,
    free_mass_response,
    independent_mass_layout,
    shared_vertex_layout,
)
from gwmock_newtonian.detector.newtonian import (
    GRAVITATIONAL_CONSTANT,
    SURFACE_AMPLITUDE_REFERENCE_METRES,
    RayleighMode,
    acceleration_quadrature,
    density_depth_amplitude,
    plane_wave_acceleration,
    surface_vertical_amplitude,
)
from gwmock_newtonian.detector.response import (
    DetectorResponseConfig,
    DetectorStrainCovariance,
    mode_strain,
    null_invisible_strain,
    null_visible_strain,
    strain_covariance,
    strain_covariance_direct,
)

__all__ = [
    "CANONICAL_DETECTOR_VERTICES",
    "GRAVITATIONAL_CONSTANT",
    "SURFACE_AMPLITUDE_REFERENCE_METRES",
    "DetectorDefinition",
    "DetectorOptics",
    "DetectorResponseConfig",
    "DetectorStrainCovariance",
    "RayleighMode",
    "Suspension",
    "TestMass",
    "TestMassLayout",
    "TriangleGeometry",
    "acceleration_quadrature",
    "density_depth_amplitude",
    "equilateral_triangle",
    "free_mass_response",
    "independent_mass_layout",
    "mode_strain",
    "null_invisible_strain",
    "null_visible_strain",
    "plane_wave_acceleration",
    "rotation_matrix",
    "shared_vertex_layout",
    "strain_covariance",
    "strain_covariance_direct",
    "surface_vertical_amplitude",
]
