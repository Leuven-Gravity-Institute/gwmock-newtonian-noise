"""Tests for NewtonianScaffoldSimulator.

Covers protocol conformance, channel mismatch, the streaming continuation
contract, and machine-readable units/provenance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    JointRealization,
    JointStrainWitnessSimulator,
)

from gwmock_newtonian.config import NewtonianScaffoldConfig
from gwmock_newtonian.simulator import BACKEND_NAME, NewtonianScaffoldSimulator

pytestmark = pytest.mark.unit


class TestProtocolConformance:
    """Structural conformance against the public gwmock_noise protocol."""

    def test_satisfies_joint_strain_witness_simulator(self, base_config: NewtonianScaffoldConfig) -> None:
        """A constructed simulator is recognized by the public protocol's runtime isinstance check."""
        simulator = NewtonianScaffoldSimulator(base_config)
        assert isinstance(simulator, JointStrainWitnessSimulator)

    def test_generate_joint_returns_joint_realization(self, base_config: NewtonianScaffoldConfig) -> None:
        """generate_joint returns the exact public JointRealization type."""
        simulator = NewtonianScaffoldSimulator(base_config)
        realization = simulator.generate_joint(
            base_config.duration, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=1
        )
        assert isinstance(realization, JointRealization)
        assert set(realization.strain) == set(base_config.detectors)
        assert set(realization.witness) == set(base_config.witnesses)


class TestChannelMismatch:
    """Requesting a channel set different from the one the simulator was configured with."""

    def test_unknown_detector_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """A detector name outside the configured set raises a clear error."""
        simulator = NewtonianScaffoldSimulator(base_config)
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(1.0, 64.0, ["NOT_A_DETECTOR"], base_config.witnesses)

    def test_unknown_witness_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """A witness name outside the configured set raises a clear error."""
        simulator = NewtonianScaffoldSimulator(base_config)
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(1.0, 64.0, base_config.detectors, ["NOT_A_WITNESS"])

    def test_empty_detectors_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """Requesting zero detectors at call time is rejected, matching the public dummy backend."""
        simulator = NewtonianScaffoldSimulator(base_config)
        with pytest.raises(ValueError, match="at least one detector"):
            simulator.generate_joint(1.0, 64.0, [], base_config.witnesses)

    def test_duplicate_requested_detector_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """A duplicated name in the requested detector list is rejected."""
        simulator = NewtonianScaffoldSimulator(base_config)
        duplicated = [base_config.detectors[0], base_config.detectors[0]]
        with pytest.raises(ValueError, match="duplicates"):
            simulator.generate_joint(1.0, 64.0, duplicated, base_config.witnesses)


class TestInvalidCallParameters:
    """duration/sampling_frequency are validated at call time, independent of config."""

    def test_non_positive_duration_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """A non-positive requested duration is refused."""
        simulator = NewtonianScaffoldSimulator(base_config)
        with pytest.raises(ValueError, match="duration"):
            simulator.generate_joint(0.0, 64.0, base_config.detectors, base_config.witnesses)

    def test_non_positive_sampling_frequency_rejected(self, base_config: NewtonianScaffoldConfig) -> None:
        """A non-positive requested sampling frequency is refused."""
        simulator = NewtonianScaffoldSimulator(base_config)
        with pytest.raises(ValueError, match="sampling_frequency"):
            simulator.generate_joint(1.0, 0.0, base_config.detectors, base_config.witnesses)


class TestSeededBatchAndStreamAgree:
    """The continuation contract: streamed chunks equal the corresponding batch draw.

    This only guards durations whose per-chunk ``round(duration *
    sampling_frequency)`` sample counts sum to the batch's own
    ``round(duration * sampling_frequency)`` count (as below: four 0.5s
    chunks at 64 Hz are 32 samples each, summing to the 128-sample 2.0s
    batch). A duration split that rounds differently per chunk than as a
    sum is outside what this test -- or the scaffold simulator's per-call
    ``round()`` -- guards; the public dummy backend this mirrors has the
    same limitation.
    """

    def test_stream_concatenation_matches_batch(self, base_config: NewtonianScaffoldConfig) -> None:
        """Four seeded 0.5s stream chunks concatenate to exactly the seeded 2.0s batch draw."""
        batch_simulator = NewtonianScaffoldSimulator(base_config)
        batch = batch_simulator.generate_joint(
            2.0, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=99
        )

        stream_simulator = NewtonianScaffoldSimulator(base_config)
        stream = stream_simulator.generate_joint_stream(
            0.5, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=99
        )
        chunks = [next(stream) for _ in range(4)]

        for name in base_config.detectors:
            concatenated = np.concatenate([chunk.strain[name] for chunk in chunks])
            np.testing.assert_array_equal(concatenated, batch.strain[name])
        for name in base_config.witnesses:
            concatenated = np.concatenate([chunk.witness[name] for chunk in chunks])
            np.testing.assert_array_equal(concatenated, batch.witness[name])

    def test_different_seeds_produce_different_realizations(self, base_config: NewtonianScaffoldConfig) -> None:
        """Sanity check that seeding actually changes the draw (guards a no-op RNG policy)."""
        simulator = NewtonianScaffoldSimulator(base_config)
        first = simulator.generate_joint(
            1.0, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=1
        )
        second = simulator.generate_joint(
            1.0, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=2
        )
        name = base_config.detectors[0]
        assert not np.array_equal(first.strain[name], second.strain[name])


class TestUnitsAndProvenanceAreMachineReadable:
    """Channel metadata and realization provenance must be typed and JSON-serializable."""

    def test_channel_metadata_has_typed_kind_and_domain(self, base_config: NewtonianScaffoldConfig) -> None:
        """Every channel's kind/domain/unit/sampling_frequency is typed, not a bare string dict."""
        simulator = NewtonianScaffoldSimulator(base_config)
        metadata = simulator.channel_metadata
        for name in base_config.detectors:
            assert metadata[name].kind == ChannelKind.STRAIN
            assert metadata[name].domain == ChannelDomain.TIME
            assert metadata[name].unit == base_config.strain_unit
            assert metadata[name].sampling_frequency == base_config.sampling_frequency
        for name in base_config.witnesses:
            assert metadata[name].kind == ChannelKind.WITNESS
            assert metadata[name].unit == base_config.witness_unit

    def test_provenance_is_json_serializable(self, base_config: NewtonianScaffoldConfig) -> None:
        """Provenance round-trips through json.dumps, so it survives being written to disk."""
        simulator = NewtonianScaffoldSimulator(base_config)
        realization = simulator.generate_joint(
            base_config.duration, base_config.sampling_frequency, base_config.detectors, base_config.witnesses, seed=1
        )
        serialized = json.dumps(realization.provenance)
        restored = json.loads(serialized)
        assert restored["backend"] == BACKEND_NAME
        assert restored["seed"] == 1
        assert restored["rng_bit_generator"] == "PCG64"
        assert restored["channel_order"] == list(base_config.detectors) + list(base_config.witnesses)

    def test_metadata_property_reflects_configured_channels(self, base_config: NewtonianScaffoldConfig) -> None:
        """The simulator-level metadata property names the implementation and configured channels."""
        simulator = NewtonianScaffoldSimulator(base_config)
        metadata = simulator.metadata
        assert metadata["implementation"] == BACKEND_NAME
        assert metadata["detectors"] == list(base_config.detectors)
        assert metadata["witnesses"] == list(base_config.witnesses)
