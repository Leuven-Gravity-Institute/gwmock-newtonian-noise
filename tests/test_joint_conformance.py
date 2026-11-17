"""The shared joint-backend conformance suite, run against the public dummy and both of this package's backends.

Every backend is obtained the same way -- by name, through ``gwmock_noise``'s
public entry-point discovery -- and must pass exactly the checks of the
``JointStrainWitnessSimulator`` contract that ``gwmock-noise`` runs against its
own dummy backend (``JointBackendConformance``). The public dummy is
``gwmock_noise``'s own ``dummy_correlated`` backend; this suite does not
special-case it.
"""

from __future__ import annotations

import pytest
from gwmock_noise import load_joint_backend
from gwmock_noise.simulators.joint_protocol import JointStrainWitnessSimulator
from gwmock_noise.testing.joint_conformance import JointBackendCase, JointBackendConformance

from gwmock_newtonian_noise.config import NewtonianScaffoldConfig
from gwmock_newtonian_noise.rng import RNG_BIT_GENERATOR

pytestmark = pytest.mark.unit


def _dummy(seed: int | None) -> JointStrainWitnessSimulator:
    return load_joint_backend("dummy_correlated")(
        detectors=["H1", "L1"], witnesses=["SEIS1", "SEIS2"], sampling_frequency=64.0, seed=seed
    )


def _scaffold(seed: int | None) -> JointStrainWitnessSimulator:
    config = NewtonianScaffoldConfig(
        detectors=("ET1", "ET2"), witnesses=("SEIS1",), sampling_frequency=64.0, duration=2.0, seed=seed
    )
    return load_joint_backend("newtonian_scaffold")(config)


@pytest.fixture(params=["dummy_correlated", "newtonian_scaffold", "newtonian_rayleigh"], ids=lambda name: name)
def joint_backend(request, joint_detector_config, joint_array) -> JointBackendCase:
    """Each registered backend, loaded through the public discovery surface."""
    if request.param == "dummy_correlated":
        return JointBackendCase(request.param, _dummy, 64.0, "joint_dummy_correlated")
    if request.param == "newtonian_scaffold":
        return JointBackendCase(request.param, _scaffold, 64.0, request.param)

    def physical(seed: int | None) -> JointStrainWitnessSimulator:
        return load_joint_backend("newtonian_rayleigh")(
            joint_detector_config, joint_array, sampling_frequency=16.0, segment_duration=2.0, seed=seed
        )

    return JointBackendCase(request.param, physical, 16.0, request.param)


class TestConformance(JointBackendConformance):
    """The shared contract every joint backend must meet."""


class TestPrivateProvenance:
    """This package's backends record more provenance than the shared contract requires."""

    @pytest.mark.parametrize("joint_backend", ["newtonian_scaffold", "newtonian_rayleigh"], indirect=True)
    def test_provenance_names_the_bit_generator_and_channel_order(self, joint_backend: JointBackendCase) -> None:
        """Provenance names this package's bit generator and every realized channel once."""
        simulator = joint_backend.build(3)
        provenance = joint_backend.generate(simulator, seed=3).provenance
        assert provenance["rng_bit_generator"] == RNG_BIT_GENERATOR == "PCG64"
        assert sorted(provenance["channel_order"]) == sorted([*simulator.detectors, *simulator.witnesses])


class TestPhysicalBackendNaming:
    """The physical backend's channels are its geometry's detectors and its array's sensors."""

    def test_channels_come_from_geometry_and_array(self, joint_detector_config, joint_array) -> None:
        """Detector names are the triangle's; witness names are the active sensors'."""
        simulator = load_joint_backend("newtonian_rayleigh")(
            joint_detector_config, joint_array, sampling_frequency=16.0, segment_duration=2.0
        )
        assert tuple(simulator.detectors) == joint_detector_config.layout.geometry.names
        assert tuple(simulator.witnesses) == joint_array.channel_names

    def test_a_witness_named_like_a_detector_is_refused(self, joint_detector_config, joint_array) -> None:
        """A sensor carrying a detector name cannot be constructed into the joint backend."""
        from dataclasses import replace

        from gwmock_newtonian_noise.witness.array import WitnessArray

        disguised = WitnessArray(sensors=(replace(joint_array.sensors[0], name="ET1"), *joint_array.sensors[1:]))
        with pytest.raises(ValueError, match="disjoint"):
            load_joint_backend("newtonian_rayleigh")(
                joint_detector_config, disguised, sampling_frequency=16.0, segment_duration=2.0
            )
