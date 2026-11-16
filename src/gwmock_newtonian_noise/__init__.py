"""Top-level package for gwmock_newtonian_noise."""

from __future__ import annotations

from gwmock_newtonian_noise.adapter import StrainOnlyAdapter
from gwmock_newtonian_noise.config import NewtonianScaffoldConfig
from gwmock_newtonian_noise.covariance import analytic_covariance, equicorrelation_matrix
from gwmock_newtonian_noise.joint import NewtonianJointSimulator
from gwmock_newtonian_noise.provenance import (
    JOINT_PROVENANCE_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    JointRealizationProvenance,
    RealizationProvenance,
)
from gwmock_newtonian_noise.rng import RNG_BIT_GENERATOR, create_generator
from gwmock_newtonian_noise.simulator import BACKEND_NAME, NewtonianScaffoldSimulator
from gwmock_newtonian_noise.version import __version__

__all__ = [
    "BACKEND_NAME",
    "JOINT_PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "RNG_BIT_GENERATOR",
    "JointRealizationProvenance",
    "NewtonianJointSimulator",
    "NewtonianScaffoldConfig",
    "NewtonianScaffoldSimulator",
    "RealizationProvenance",
    "StrainOnlyAdapter",
    "__version__",
    "analytic_covariance",
    "create_generator",
    "equicorrelation_matrix",
]
