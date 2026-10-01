"""One conformance suite run against the public dummy and both of this package's backends.

Every backend is obtained the same way -- by name, through ``gwmock_noise``'s
public entry-point discovery -- and must pass exactly the same checks of the
``JointStrainWitnessSimulator`` contract: the realization and metadata shapes,
seeded determinism, stream continuity, a Hermitian positive-semidefinite
covariance over exactly the realized channels, JSON-native provenance and
metadata, and that no witness is disguised as a detector or carried as an array
in metadata.

The public dummy is ``gwmock_noise``'s own ``dummy_correlated`` backend; this
suite does not special-case it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from gwmock_noise import load_joint_backend
from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    JointCovariance,
    JointRealization,
    JointStrainWitnessSimulator,
)

from gwmock_newtonian.config import NewtonianScaffoldConfig

pytestmark = pytest.mark.unit

#: Samples per conformance chunk.
_CHUNK_SAMPLES = 64
#: Relative floor on eigenvalues of the normalized covariance slices.
_PSD_FLOOR = 1e-10


@dataclass(frozen=True)
class Backend:
    """A registered backend and how to build it."""

    name: str
    build: Callable[[int | None], JointStrainWitnessSimulator]
    sampling_frequency: float


def _dummy(seed: int | None) -> JointStrainWitnessSimulator:
    return load_joint_backend("dummy_correlated")(
        detectors=["H1", "L1"], witnesses=["SEIS1", "SEIS2"], sampling_frequency=64.0, seed=seed
    )


def _scaffold(seed: int | None) -> JointStrainWitnessSimulator:
    config = NewtonianScaffoldConfig(
        detectors=("ET1", "ET2"), witnesses=("SEIS1",), sampling_frequency=64.0, duration=2.0, seed=seed
    )
    return load_joint_backend("newtonian_scaffold")(config)


@pytest.fixture(params=["dummy_correlated", "newtonian_scaffold", "newtonian_rayleigh"])
def backend(request, joint_detector_config, joint_array) -> Backend:
    """Each registered backend, loaded through the public discovery surface."""
    if request.param == "dummy_correlated":
        return Backend(request.param, _dummy, 64.0)
    if request.param == "newtonian_scaffold":
        return Backend(request.param, _scaffold, 64.0)

    def physical(seed: int | None) -> JointStrainWitnessSimulator:
        return load_joint_backend("newtonian_rayleigh")(
            joint_detector_config, joint_array, sampling_frequency=16.0, segment_duration=2.0, seed=seed
        )

    return Backend(request.param, physical, 16.0)


def _chunk_duration(backend: Backend) -> float:
    return _CHUNK_SAMPLES / backend.sampling_frequency


def _generate(simulator, backend: Backend, seed: int | None = None, chunks: int = 1) -> JointRealization:
    return simulator.generate_joint(
        chunks * _chunk_duration(backend),
        backend.sampling_frequency,
        list(simulator.detectors),
        list(simulator.witnesses),
        seed=seed,
    )


#: Longest numeric sequence a metadata record may legitimately hold: a 3-vector
#: (position, orientation) or a ``[re, im]`` pair.
_MAX_METADATA_VECTOR = 3


def array_leaks(value: Any, path: str = "") -> list[str]:
    """Return the paths in a metadata/provenance tree that hold channel-like data.

    Flags any :class:`numpy.ndarray`, and any list or tuple of more than
    :data:`_MAX_METADATA_VECTOR` numbers. Sequences of names (channel orders)
    are not data and are walked, not flagged.
    """
    if isinstance(value, np.ndarray):
        return [f"{path or '<root>'}: ndarray{value.shape}"]
    if isinstance(value, dict):
        return [leak for key, item in value.items() for leak in array_leaks(item, f"{path}/{key}")]
    if isinstance(value, (list, tuple)):
        numeric = len(value) > _MAX_METADATA_VECTOR and all(
            isinstance(item, (int, float, complex)) and not isinstance(item, bool) for item in value
        )
        own = [f"{path}: {len(value)} numbers"] if numeric else []
        return own + [leak for index, item in enumerate(value) for leak in array_leaks(item, f"{path}[{index}]")]
    return []


class TestConformance:
    """The shared contract every joint backend must meet."""

    def test_satisfies_the_public_protocol(self, backend: Backend) -> None:
        """The runtime-checkable public protocol recognizes the backend."""
        assert isinstance(backend.build(1), JointStrainWitnessSimulator)

    def test_realization_shape_dtype_and_channel_sets(self, backend: Backend) -> None:
        """Strain and witness dicts hold exactly the requested channels as 1-D float64 arrays."""
        simulator = backend.build(1)
        realization = _generate(simulator, backend, seed=1)
        assert isinstance(realization, JointRealization)
        assert set(realization.strain) == set(simulator.detectors)
        assert set(realization.witness) == set(simulator.witnesses)
        for series in (*realization.strain.values(), *realization.witness.values()):
            assert series.dtype == np.float64
            assert series.shape == (_CHUNK_SAMPLES,)
            assert np.all(np.isfinite(series))

    def test_channel_metadata_is_typed_and_complete(self, backend: Backend) -> None:
        """Every realized channel has typed metadata with the right kind, domain, rate and dtype."""
        simulator = backend.build(1)
        realization = _generate(simulator, backend, seed=1)
        metadata = realization.channel_metadata
        assert set(metadata) == set(realization.strain) | set(realization.witness)
        for name, record in metadata.items():
            expected_kind = ChannelKind.STRAIN if name in realization.strain else ChannelKind.WITNESS
            assert record.name == name
            assert record.kind is expected_kind
            assert record.domain is ChannelDomain.TIME
            assert record.sampling_frequency == backend.sampling_frequency
            assert record.dtype == "float64"
            assert isinstance(record.unit, str)
            assert record.unit

    def test_seeded_generation_is_deterministic(self, backend: Backend) -> None:
        """Equal seeds give bit-identical realizations; different seeds differ."""
        first = _generate(backend.build(None), backend, seed=7)
        second = _generate(backend.build(None), backend, seed=7)
        other = _generate(backend.build(None), backend, seed=8)
        for name in first.strain:
            np.testing.assert_array_equal(first.strain[name], second.strain[name])
            assert not np.array_equal(first.strain[name], other.strain[name])
        for name in first.witness:
            np.testing.assert_array_equal(first.witness[name], second.witness[name])

    def test_stream_chunks_continue_the_batch(self, backend: Backend) -> None:
        """Four seeded stream chunks concatenate to the seeded batch over the same span."""
        batch = _generate(backend.build(None), backend, seed=21, chunks=4)
        simulator = backend.build(None)
        stream = simulator.generate_joint_stream(
            _chunk_duration(backend),
            backend.sampling_frequency,
            list(simulator.detectors),
            list(simulator.witnesses),
            seed=21,
        )
        chunks = [next(stream) for _ in range(4)]
        for name in batch.strain:
            np.testing.assert_array_equal(np.concatenate([c.strain[name] for c in chunks]), batch.strain[name])
        for name in batch.witness:
            np.testing.assert_array_equal(np.concatenate([c.witness[name] for c in chunks]), batch.witness[name])

    def test_covariance_is_hermitian_psd_over_the_realized_channels(self, backend: Backend) -> None:
        """The covariance indexes exactly the realized channels and is Hermitian PSD on an rfft grid."""
        simulator = backend.build(1)
        covariance = simulator.covariance()
        assert isinstance(covariance, JointCovariance)
        assert set(covariance.channel_order) == set(simulator.detectors) | set(simulator.witnesses)
        assert len(covariance.channel_order) == len(set(covariance.channel_order))
        frequencies = covariance.frequencies
        assert frequencies[0] == 0.0
        assert np.all(np.diff(frequencies) > 0.0)
        assert frequencies[-1] == pytest.approx(backend.sampling_frequency / 2.0)
        n = len(covariance.channel_order)
        assert covariance.matrix.shape == (frequencies.size, n, n)
        diagonal = np.real(np.diagonal(covariance.matrix, axis1=1, axis2=2))
        excited = np.all(diagonal > 0.0, axis=1)
        assert np.count_nonzero(excited) > frequencies.size // 2
        scale = 1.0 / np.sqrt(diagonal[excited])
        normalized = covariance.matrix[excited] * scale[:, :, None] * scale[:, None, :]
        hermitian_part = 0.5 * (normalized + np.conj(np.swapaxes(normalized, 1, 2)))
        assert np.max(np.abs(normalized - hermitian_part)) < 1e-12
        assert np.min(np.linalg.eigvalsh(hermitian_part)) > -_PSD_FLOOR

    def test_provenance_and_metadata_are_json_native(self, backend: Backend) -> None:
        """Provenance and metadata survive a strict ``json.dumps`` round trip and name the backend."""
        simulator = backend.build(3)
        realization = _generate(simulator, backend, seed=3)
        provenance = realization.provenance
        assert json.loads(json.dumps(provenance)) == provenance
        assert json.loads(json.dumps(simulator.metadata)) == simulator.metadata
        assert provenance["backend"] == backend.name.replace("dummy_correlated", "joint_dummy_correlated")
        assert provenance["seed"] == 3
        assert provenance["rng_bit_generator"] == "PCG64"
        assert provenance["package_version"]
        assert set(provenance["channel_order"]) == set(simulator.detectors) | set(simulator.witnesses)

    def test_changing_the_channel_set_is_refused(self, backend: Backend) -> None:
        """An unknown detector or witness name is refused rather than silently generated."""
        simulator = backend.build(1)
        fs = backend.sampling_frequency
        duration = _chunk_duration(backend)
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(duration, fs, ["NOT_A_DETECTOR"], list(simulator.witnesses), seed=1)
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(duration, fs, list(simulator.detectors), ["NOT_A_WITNESS"], seed=1)


class TestNoDisguisedWitnesses:
    """No witness is disguised as a detector name or stored as an array in metadata."""

    def test_witnesses_are_never_detector_names_or_strain(self, backend: Backend) -> None:
        """Detector and witness names are disjoint, and every witness is typed and delivered as a witness."""
        simulator = backend.build(1)
        realization = _generate(simulator, backend, seed=1)
        assert not set(simulator.detectors) & set(simulator.witnesses)
        assert not set(realization.strain) & set(simulator.witnesses)
        for name in simulator.witnesses:
            assert name in realization.witness
            assert realization.channel_metadata[name].kind is ChannelKind.WITNESS
        for name in simulator.detectors:
            assert realization.channel_metadata[name].kind is ChannelKind.STRAIN

    def test_no_channel_data_in_metadata_or_provenance(self, backend: Backend) -> None:
        """Metadata, provenance and channel metadata hold no arrays and no channel-length sequences."""
        simulator = backend.build(1)
        realization = _generate(simulator, backend, seed=1)
        for label, tree in (
            ("metadata", simulator.metadata),
            ("provenance", realization.provenance),
            ("channel_metadata", {name: vars(record) for name, record in realization.channel_metadata.items()}),
        ):
            assert array_leaks(tree) == [], label
        assert all(isinstance(name, str) for name in simulator.metadata["witnesses"])
        assert all(isinstance(name, str) for name in simulator.metadata["detectors"])

    def test_the_leak_detector_flags_arrays_and_numeric_lists(self) -> None:
        """The metadata check is not vacuous: it flags an ndarray and any short numeric series."""
        series = np.zeros(4)
        assert array_leaks({"witness": {"SEIS1": series}}) == ["/witness/SEIS1: ndarray(4,)"]
        assert array_leaks({"SEIS1": series.tolist()}) == ["/SEIS1: 4 numbers"]
        assert array_leaks({"nested": [{"samples": (1, 2.0, 3, 4)}]}) == ["/nested[0]/samples: 4 numbers"]
        legitimate = {"channel_order": ["ET1", "ET2", "ET3", "W0", "W1"], "position": [0.0, 1.0, 2.0], "c": [1.0, 0.0]}
        assert array_leaks(legitimate) == []


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

        from gwmock_newtonian.witness.array import WitnessArray

        disguised = WitnessArray(sensors=(replace(joint_array.sensors[0], name="ET1"), *joint_array.sensors[1:]))
        with pytest.raises(ValueError, match="disjoint"):
            load_joint_backend("newtonian_rayleigh")(
                joint_detector_config, disguised, sampling_frequency=16.0, segment_duration=2.0
            )
