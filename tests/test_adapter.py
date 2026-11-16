"""Tests for StrainOnlyAdapter: the gwmock-noise compatibility adapter."""

from __future__ import annotations

import numpy as np
import pytest
from gwmock_noise.simulators.protocol import NoiseSimulator

from gwmock_newtonian_noise.adapter import StrainOnlyAdapter
from gwmock_newtonian_noise.config import NewtonianScaffoldConfig
from gwmock_newtonian_noise.simulator import NewtonianScaffoldSimulator

pytestmark = pytest.mark.unit


class TestAdapterConstruction:
    """The adapter only accepts a protocol-conformant joint simulator."""

    def test_wraps_conformant_simulator(self, base_config: NewtonianScaffoldConfig) -> None:
        """A NewtonianScaffoldSimulator is accepted without error."""
        StrainOnlyAdapter(NewtonianScaffoldSimulator(base_config))

    def test_rejects_non_conformant_object(self) -> None:
        """An object missing the joint protocol's surface is rejected up front, not at first use."""
        with pytest.raises(TypeError, match="JointStrainWitnessSimulator"):
            StrainOnlyAdapter(object())


class TestAdapterSatisfiesLegacyProtocol:
    """Existing gwmock-noise consumers only know the legacy NoiseSimulator protocol."""

    def test_isinstance_of_legacy_noise_simulator(self, base_config: NewtonianScaffoldConfig) -> None:
        """The adapter itself is recognized by the legacy protocol's runtime isinstance check."""
        adapter = StrainOnlyAdapter(NewtonianScaffoldSimulator(base_config))
        assert isinstance(adapter, NoiseSimulator)

    def test_attributes_mirror_the_wrapped_simulator(self, base_config: NewtonianScaffoldConfig) -> None:
        """duration/sampling_frequency/detectors/seed pass through from the wrapped simulator."""
        joint = NewtonianScaffoldSimulator(base_config)
        adapter = StrainOnlyAdapter(joint)
        assert adapter.duration == joint.duration
        assert adapter.sampling_frequency == joint.sampling_frequency
        assert adapter.detectors == joint.detectors
        assert adapter.seed == joint.seed


class TestAdapterOutputIsStrainOnly:
    """The adapter must never leak witness channels, channel_metadata, or provenance."""

    def test_generate_returns_only_strain_channels(self, base_config: NewtonianScaffoldConfig) -> None:
        """generate() returns exactly the requested detector names and nothing else."""
        adapter = StrainOnlyAdapter(NewtonianScaffoldSimulator(base_config))
        result = adapter.generate(base_config.duration, base_config.sampling_frequency, base_config.detectors, seed=5)
        assert set(result) == set(base_config.detectors)
        assert isinstance(result, dict)
        for array in result.values():
            assert isinstance(array, np.ndarray)

    def test_generate_stream_returns_only_strain_channels(self, base_config: NewtonianScaffoldConfig) -> None:
        """generate_stream() yields strain-only dicts, chunk after chunk."""
        adapter = StrainOnlyAdapter(NewtonianScaffoldSimulator(base_config))
        stream = adapter.generate_stream(0.5, base_config.sampling_frequency, base_config.detectors, seed=5)
        chunk = next(stream)
        assert set(chunk) == set(base_config.detectors)

    def test_adapter_output_matches_joint_strain(self, base_config: NewtonianScaffoldConfig) -> None:
        """The adapter's strain-only output is bit-identical to the joint realization's strain dict."""
        joint = NewtonianScaffoldSimulator(base_config)
        adapter = StrainOnlyAdapter(joint)

        realization = joint.generate_joint(
            base_config.duration, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=3
        )
        strain_only = adapter.generate(
            base_config.duration, base_config.sampling_frequency, base_config.detectors, seed=3
        )

        for name in base_config.detectors:
            np.testing.assert_array_equal(strain_only[name], realization.strain[name])
