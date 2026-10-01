"""Physical joint strain+witness backend for the public ``gwmock_noise`` protocol.

:class:`NewtonianJointSimulator` implements the optional
``gwmock_noise.JointStrainWitnessSimulator`` protocol on top of the physical
model: the three ET strains of :mod:`gwmock_newtonian.detector.response` and the
seismic witnesses of :mod:`gwmock_newtonian.witness`, all observing one shared
Rayleigh-wave field. It adds no physics of its own. Its realizations are
:func:`~gwmock_newtonian.witness.response.synthesize_joint` draws and its
covariance is :func:`~gwmock_newtonian.witness.response.joint_covariance`, both
from the one per-mode response matrix computed at construction.

The declared seeded process
---------------------------
A spectral-representation draw is periodic in its own duration, so a stream
cannot be one draw over an unbounded time. The process this backend declares is
therefore a sequence of independent, fixed-length segments: sample ``n`` lies
in segment ``n // segment_samples``, and segment ``k`` is the
:func:`~gwmock_newtonian.witness.response.synthesize_joint` draw seeded by
:func:`~gwmock_newtonian.rng.segment_seed_sequence` ``(entropy, k)``. Every
sample of the process is fixed by the root entropy alone, so a batch call and
any chunking of a stream over the same samples -- chunks inside a segment or
straddling segment boundaries -- return bit-identical arrays.

Within a segment the process has exactly the analytic covariance on the
segment's :func:`numpy.fft.rfftfreq` grid, which is what :meth:`covariance`
reports. Two segments are independent, so a window that straddles a segment
boundary sees a join between two independent draws rather than a continuation
of one; estimators should use segment-aligned windows.

Cross-spectrum convention: ``covariance().matrix[k, a, b]`` is estimated by
``(2 / (fs N)) E[conj(X_a) X_b]`` with ``X = numpy.fft.rfft(x)``, which is the
convention of ``scipy.signal.csd(x_a, x_b)``; see
:mod:`gwmock_newtonian.witness.response`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    ChannelMetadata,
    JointCovariance,
    JointRealization,
)
from numpy.typing import NDArray

from gwmock_newtonian.detector.response import DetectorResponseConfig
from gwmock_newtonian.provenance import JointRealizationProvenance, source_state
from gwmock_newtonian.rng import segment_seed_sequence
from gwmock_newtonian.version import __version__
from gwmock_newtonian.witness.array import ArrayManifest, WitnessArray
from gwmock_newtonian.witness.response import (
    JointStrainWitnessCovariance,
    joint_covariance,
    joint_mode_responses,
    synthesize_joint,
)

#: The backend identifier recorded in provenance/metadata, and the name this
#: class is registered under in ``[project.entry-points."gwmock_noise.joint_backends"]``.
BACKEND_NAME = "newtonian_rayleigh"
#: Physical model identifier recorded in provenance and metadata.
PHYSICAL_MODEL = "rayleigh_wave_newtonian_noise_et_triangle_with_seismic_witnesses"
#: Unit recorded for the three strain channels.
STRAIN_UNIT = "strain"
#: Relative tolerance within which ``duration * sampling_frequency`` must be an integer.
_SAMPLE_COUNT_RTOL = 1e-9
#: Fewest samples a segment can hold (a start and an end sample).
_MIN_SEGMENT_SAMPLES = 2


def _sample_count(duration: float, sampling_frequency: float, label: str) -> int:
    """Return ``duration * sampling_frequency`` as a positive integer sample count.

    A non-integer product is refused rather than rounded: rounding each chunk
    separately makes consecutive chunks drift off the batch sample grid.
    """
    if not duration > 0.0:
        raise ValueError(f"{label} must be greater than zero, got {duration}.")
    exact = duration * sampling_frequency
    count = round(exact)
    if count < 1 or not math.isclose(exact, count, rel_tol=_SAMPLE_COUNT_RTOL, abs_tol=_SAMPLE_COUNT_RTOL):
        raise ValueError(
            f"{label} * sampling_frequency must be a positive integer number of samples, "
            f"got {duration} * {sampling_frequency} = {exact}."
        )
    return count


def _sha256_array(array: NDArray[Any]) -> str:
    """Return the SHA-256 of an array's contiguous bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class NewtonianJointSimulator:
    """Physical joint strain+witness simulator implementing the public protocol.

    The detector channels are the triangle geometry's names and the witness
    channels the array's active channel names; both sets are fixed for the
    lifetime of an instance (later calls may reorder but not change them), as
    is the sampling frequency the response matrix was computed at.
    """

    def __init__(  # noqa: PLR0913 -- one parameter per physically distinct input.
        self,
        detector_config: DetectorResponseConfig,
        array: WitnessArray,
        *,
        sampling_frequency: float,
        segment_duration: float,
        seed: int | None = None,
        manifest_frequency: float = 1.0,
    ) -> None:
        """Compute the per-mode response and the analytic covariance of one segment.

        Args:
            detector_config: The detector response configuration; its field is
                the latent field the witnesses also observe.
            array: The witness array; at least one sensor must be active.
            sampling_frequency: Sampling frequency in hertz, strictly positive.
            segment_duration: Length in seconds of one segment of the declared
                process (see the module docstring); an integer number of at
                least two samples.
            seed: Integer seed of the declared process, or ``None`` for fresh
                OS entropy (recorded in provenance, so still replayable).
            manifest_frequency: Frequency at which the array manifest is evaluated.
        """
        if not sampling_frequency > 0.0:
            raise ValueError(f"sampling_frequency must be greater than zero, got {sampling_frequency}.")
        segment_samples = _sample_count(segment_duration, sampling_frequency, "segment_duration")
        if segment_samples < _MIN_SEGMENT_SAMPLES:
            raise ValueError("segment_duration must span at least two samples.")
        detector_names = detector_config.layout.geometry.names
        witness_names = array.channel_names
        if not witness_names:
            raise ValueError("array must have at least one active witness channel.")

        self._detector_config = detector_config
        self._array = array
        self._detector_names = tuple(detector_names)
        self._witness_names = tuple(witness_names)
        self._channel_order = (*self._detector_names, *self._witness_names)
        self._sampling_frequency = float(sampling_frequency)
        self._segment_duration = float(segment_duration)
        self._segment_samples = segment_samples

        frequencies = np.fft.rfftfreq(segment_samples, d=1.0 / self._sampling_frequency)
        self._responses = joint_mode_responses(frequencies, detector_config, array)
        self._analytic = joint_covariance(
            frequencies, detector_config, array, manifest_frequency, responses=self._responses
        )
        self._fingerprints = {
            "responses_sha256": _sha256_array(self._responses),
            "covariance_sha256": _sha256_array(self._analytic.matrix),
            "array_manifest_sha256": hashlib.sha256(
                json.dumps(self._analytic.array_manifest.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
        }
        self._source = source_state()

        self.detectors = list(self._detector_names)
        self.witnesses = list(self._witness_names)
        self.duration = self._segment_duration
        self.sampling_frequency = self._sampling_frequency
        self.seed = seed

        self._entropy: int | None = None
        self._cursor = 0
        self._cached_segment: tuple[int, NDArray[np.float64]] | None = None

    def _validate_request(self, sampling_frequency: float, detectors: list[str], witnesses: list[str]) -> None:
        """Validate one call's sampling frequency and channel lists against the fixed configuration."""
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if not witnesses:
            raise ValueError("witnesses must contain at least one witness channel.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicates.")
        if len(set(witnesses)) != len(witnesses):
            raise ValueError("witnesses must not contain duplicates.")
        if set(detectors) != set(self._detector_names):
            raise ValueError(
                "Changing the detector set after construction is unsupported "
                f"(configured: {sorted(self._detector_names)}, requested: {sorted(detectors)})."
            )
        if set(witnesses) != set(self._witness_names):
            raise ValueError(
                "Changing the witness set after construction is unsupported "
                f"(configured: {sorted(self._witness_names)}, requested: {sorted(witnesses)})."
            )
        if sampling_frequency != self._sampling_frequency:
            raise ValueError(
                "Changing the sampling frequency after construction is unsupported "
                f"(configured: {self._sampling_frequency} Hz, requested: {sampling_frequency} Hz)."
            )

    def _restart(self, seed: int | None) -> None:
        """Restart the declared process from sample zero under ``seed``."""
        self.seed = seed
        self._entropy = int(np.random.SeedSequence(seed).entropy)
        self._cursor = 0
        self._cached_segment = None

    def _segment(self, index: int) -> NDArray[np.float64]:
        """Return segment ``index`` of the declared process, ``(n_channel, segment_samples)``."""
        if self._cached_segment is not None and self._cached_segment[0] == index:
            return self._cached_segment[1]
        series = synthesize_joint(
            self._responses,
            self._detector_config,
            self._array,
            self._segment_duration,
            self._sampling_frequency,
            segment_seed_sequence(int(self._entropy), index),
        )
        stacked = np.stack([series[name] for name in self._channel_order])
        self._cached_segment = (index, stacked)
        return stacked

    def _draw(self, start: int, n_samples: int) -> NDArray[np.float64]:
        """Return samples ``start .. start + n_samples - 1`` of the declared process."""
        output = np.empty((len(self._channel_order), n_samples), dtype=float)
        filled = 0
        while filled < n_samples:
            segment, offset = divmod(start + filled, self._segment_samples)
            take = min(self._segment_samples - offset, n_samples - filled)
            output[:, filled : filled + take] = self._segment(segment)[:, offset : offset + take]
            filled += take
        return output

    def generate_joint(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> JointRealization:
        """Return the next ``duration`` seconds of the declared seeded process.

        An integer ``seed`` (or the first call) restarts the process at sample
        zero; ``seed=None`` on a started process continues from where the
        previous call stopped.
        """
        self._validate_request(sampling_frequency, detectors, witnesses)
        n_samples = _sample_count(duration, sampling_frequency, "duration")
        if seed is not None or self._entropy is None:
            self._restart(seed if seed is not None else self.seed)
        start = self._cursor
        samples = self._draw(start, n_samples)
        self._cursor = start + n_samples

        self.duration = duration
        self.detectors = list(detectors)
        self.witnesses = list(witnesses)

        index = {name: position for position, name in enumerate(self._channel_order)}
        strain = {name: samples[index[name]].copy() for name in self.detectors}
        witness = {name: samples[index[name]].copy() for name in self.witnesses}
        return JointRealization(
            strain=strain,
            witness=witness,
            channel_metadata=self.channel_metadata,
            provenance=self._provenance(start, n_samples).to_dict(),
        )

    def generate_joint_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> Iterator[JointRealization]:
        """Yield consecutive ``chunk_duration`` chunks of the declared seeded process.

        Concatenating the chunks reproduces, bit for bit, the batch
        :meth:`generate_joint` call over the combined duration from the same seed.
        """
        while True:
            yield self.generate_joint(chunk_duration, sampling_frequency, detectors, witnesses, seed)
            seed = None

    def covariance(self) -> JointCovariance:
        """Return the one-sided joint cross-spectral density on the segment's rfft grid.

        ``matrix[k, a, b]`` is in ``unit_a * unit_b / Hz``, with the channel
        units of :attr:`channel_metadata`.
        """
        return JointCovariance(
            frequencies=self._analytic.frequencies.copy(),
            matrix=self._analytic.matrix.copy(),
            channel_order=list(self._channel_order),
        )

    @property
    def analytic_covariance(self) -> JointStrainWitnessCovariance:
        """Return the typed joint covariance (blocks, null projection, manifest)."""
        return self._analytic

    @property
    def array_manifest(self) -> ArrayManifest:
        """Return the typed witness-array manifest the covariance was built with."""
        return self._analytic.array_manifest

    @property
    def segment_samples(self) -> int:
        """Return the number of samples in one segment of the declared process."""
        return self._segment_samples

    @property
    def channel_metadata(self) -> dict[str, ChannelMetadata]:
        """Return typed metadata for every strain and witness channel."""
        metadata: dict[str, ChannelMetadata] = {}
        for name in self._detector_names:
            metadata[name] = ChannelMetadata(
                name=name,
                kind=ChannelKind.STRAIN,
                unit=STRAIN_UNIT,
                domain=ChannelDomain.TIME,
                sampling_frequency=self._sampling_frequency,
                dtype="float64",
            )
        for sensor in self._array.active_sensors:
            metadata[sensor.name] = ChannelMetadata(
                name=sensor.name,
                kind=ChannelKind.WITNESS,
                unit=sensor.unit,
                domain=ChannelDomain.TIME,
                sampling_frequency=self._sampling_frequency,
                dtype="float64",
            )
        return metadata

    def _provenance(self, start: int, n_samples: int) -> JointRealizationProvenance:
        """Return the provenance of the samples ``start .. start + n_samples - 1``."""
        return JointRealizationProvenance(
            backend=BACKEND_NAME,
            source_commit=self._source.commit,
            source_dirty=self._source.dirty,
            physical_model=PHYSICAL_MODEL,
            seed=self.seed,
            seed_entropy=str(self._entropy),
            sampling_frequency=self._sampling_frequency,
            segment_duration=self._segment_duration,
            segment_samples=self._segment_samples,
            start_sample=start,
            n_samples=n_samples,
            channel_order=self._channel_order,
            detector_channels=self._detector_names,
            witness_channels=self._witness_names,
            **self._fingerprints,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata (same role as ``NoiseSimulator.metadata``)."""
        return {
            "implementation": BACKEND_NAME,
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "witnesses": list(self.witnesses),
            "seed": self.seed,
            BACKEND_NAME: {
                "physical_model": PHYSICAL_MODEL,
                "channel_order": list(self._channel_order),
                "segment_duration": self._segment_duration,
                "segment_samples": self._segment_samples,
                "package_version": __version__,
                "source_commit": self._source.commit,
                "source_dirty": self._source.dirty,
            },
        }


__all__ = ["BACKEND_NAME", "PHYSICAL_MODEL", "STRAIN_UNIT", "NewtonianJointSimulator"]
