"""Newtonian-noise field families beyond the minimal Rayleigh field.

Each family describes a stationary Gaussian seismic field as independent latent
modes with a displacement and a Newtonian-acceleration response
(:mod:`~gwmock_newtonian.families.base`), so one generic consumer
(:mod:`~gwmock_newtonian.families.response`) builds strain, witness and joint
covariances and realizations for all of them, and independent families combine
by mixing (:mod:`~gwmock_newtonian.families.mixture`). The families are staged:

- :mod:`~gwmock_newtonian.families.rayleigh` -- the existing minimal Rayleigh
  field through the family contract, and a phase-velocity-spread proxy for
  lateral inhomogeneity;
- :mod:`~gwmock_newtonian.families.body` -- P/SV/SH body waves with explicit
  composition and three-dimensional directionality, in the infinite-medium limit
  or a half-space with free-surface reflections, and the equipartitioned diffuse
  field as a scattering proxy;
- :mod:`~gwmock_newtonian.families.love` -- fundamental Love waves of a layer
  over a half-space;
- :mod:`~gwmock_newtonian.families.point` -- localized isotropic point sources.

No family fixes a site model: every site parameter is configuration without a
default and is listed in the family's ``UNANCHORED_PARAMETERS``.
"""

from __future__ import annotations

from gwmock_newtonian.families.base import (
    FieldFamily,
    SphereGrid,
    uniform_sphere_density,
    von_mises_fisher_density,
)
from gwmock_newtonian.families.body import (
    BODY_WAVE_TYPES,
    BodyWaveFamily,
    equipartition_shear_to_p_ratio,
    equipartitioned_body_waves,
    free_surface_components,
)
from gwmock_newtonian.families.love import LayerOverHalfSpace, LoveWaveFamily, love_phase_velocity
from gwmock_newtonian.families.mixture import FamilyMixture
from gwmock_newtonian.families.point import PointSource, PointSourceFamily, point_sources
from gwmock_newtonian.families.rayleigh import RayleighFamily, phase_velocity_spread
from gwmock_newtonian.families.response import (
    CavityModel,
    FamilyJointModel,
    family_joint_covariance,
    family_joint_mode_responses,
    family_strain_responses,
    family_witness_responses,
    newtonian_accelerations,
    synthesize_family_joint,
)

__all__ = [
    "BODY_WAVE_TYPES",
    "BodyWaveFamily",
    "CavityModel",
    "FamilyJointModel",
    "FamilyMixture",
    "FieldFamily",
    "LayerOverHalfSpace",
    "LoveWaveFamily",
    "PointSource",
    "PointSourceFamily",
    "RayleighFamily",
    "SphereGrid",
    "equipartition_shear_to_p_ratio",
    "equipartitioned_body_waves",
    "family_joint_covariance",
    "family_joint_mode_responses",
    "family_strain_responses",
    "family_witness_responses",
    "free_surface_components",
    "love_phase_velocity",
    "newtonian_accelerations",
    "phase_velocity_spread",
    "point_sources",
    "synthesize_family_joint",
    "uniform_sphere_density",
    "von_mises_fisher_density",
]
