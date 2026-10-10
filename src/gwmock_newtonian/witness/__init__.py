"""Seismic witness-sensor responses and controlled arrays.

Public subpackage implementing the witness physics: sensors with an
explicit measured quantity, position, orientation, complex frequency-dependent
response, calibration, self-noise, timing offset and array-level correlated
noise; missing channels; nested benchmark arrays in a named fixed-aperture or
growing-span mode, with manifests that report raw count and noise effective
rank separately and the realized radius next to the aperture ceiling; and the
joint strain--witness covariance and realization built by reusing the
detector response operator.
"""

from __future__ import annotations

from gwmock_newtonian.witness.array import (
    ArrayManifest,
    CorrelatedNoise,
    DiagonalNoise,
    ExplicitNoise,
    NestedArrayMode,
    NoiseModel,
    SensorManifest,
    WitnessArray,
    effective_rank,
    nested_disk_array,
    nested_site_order,
    participation_ratio,
    sunflower_positions,
)
from gwmock_newtonian.witness.response import (
    JointStrainWitnessCovariance,
    WitnessCovariance,
    joint_covariance,
    joint_mode_responses,
    joint_strain_covariance_block,
    realize_joint,
    synthesize_joint,
    witness_covariance,
    witness_mode_responses,
)
from gwmock_newtonian.witness.sensor import (
    ComplexResponse,
    PSDCallable,
    SensorQuantity,
    WitnessSensor,
    constant_response,
    quantity_transfer,
    unit_response,
)

__all__ = [
    "ArrayManifest",
    "ComplexResponse",
    "CorrelatedNoise",
    "DiagonalNoise",
    "ExplicitNoise",
    "JointStrainWitnessCovariance",
    "NestedArrayMode",
    "NoiseModel",
    "PSDCallable",
    "SensorManifest",
    "SensorQuantity",
    "WitnessArray",
    "WitnessCovariance",
    "WitnessSensor",
    "constant_response",
    "effective_rank",
    "joint_covariance",
    "joint_mode_responses",
    "joint_strain_covariance_block",
    "nested_disk_array",
    "nested_site_order",
    "participation_ratio",
    "quantity_transfer",
    "realize_joint",
    "sunflower_positions",
    "synthesize_joint",
    "unit_response",
    "witness_covariance",
    "witness_mode_responses",
]
