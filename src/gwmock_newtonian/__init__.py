"""Top-level package for gwmock_newtonian."""

from __future__ import annotations

from gwmock_newtonian.adapter import StrainOnlyAdapter
from gwmock_newtonian.config import NewtonianScaffoldConfig
from gwmock_newtonian.covariance import analytic_covariance, equicorrelation_matrix
from gwmock_newtonian.joint import NewtonianJointSimulator
from gwmock_newtonian.provenance import (
    JOINT_PROVENANCE_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    JointRealizationProvenance,
    RealizationProvenance,
)
from gwmock_newtonian.rng import RNG_BIT_GENERATOR, create_generator
from gwmock_newtonian.simulator import BACKEND_NAME, NewtonianScaffoldSimulator
from gwmock_newtonian.version import __version__

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
