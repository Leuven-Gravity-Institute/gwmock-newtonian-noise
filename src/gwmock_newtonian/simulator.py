"""Realization and stream API for the Newtonian-noise scaffold simulator.

Implements the public, optional ``gwmock_noise.JointStrainWitnessSimulator``
protocol with a physics-free placeholder statistical model. **No
Newtonian-noise or seismic physics is modelled here** -- this class exists
to conformance-test the public protocol, the RNG policy, the
channel/provenance types, and the compatibility adapter. The physical model
is the joint backend in :mod:`gwmock_newtonian.joint`.
"""

from __future__ import annotations

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

from gwmock_newtonian.config import NewtonianScaffoldConfig
from gwmock_newtonian.covariance import analytic_covariance, equicorrelation_matrix
from gwmock_newtonian.provenance import RealizationProvenance
from gwmock_newtonian.rng import create_generator

#: The backend identifier recorded in provenance/metadata, and the name this
#: class is registered under in ``[project.entry-points."gwmock_noise.joint_backends"]``.
BACKEND_NAME = "newtonian_scaffold"


class NewtonianScaffoldSimulator:
    """Physics-free scaffold implementing the public joint strain+witness protocol.

    Every channel is temporally white with a fixed equicorrelation structure
    between every channel pair (strain-strain, witness-witness, and
    strain-witness alike): it exercises and conformance-tests
    :class:`~gwmock_noise.JointStrainWitnessSimulator` without claiming any
    Newtonian-noise or seismic physics.
    """

    def __init__(self, config: NewtonianScaffoldConfig) -> None:
        """Initialize the scaffold simulator from a validated configuration.

        Args:
            config: Validated simulator configuration.
        """
        self._config = config
        self._detectors = list(config.detectors)
        self._witnesses = list(config.witnesses)
        self._channel_order = self._detectors + self._witnesses

        self.detectors = list(self._detectors)
        self.witnesses = list(self._witnesses)
        self.duration = config.duration
        self.sampling_frequency = config.sampling_frequency
        self.seed = config.seed

        self._sample_covariance = equicorrelation_matrix(self._channel_order, config.coupling)
        self._cholesky = np.linalg.cholesky(self._sample_covariance)
        self._rng: np.random.Generator | None = None

    @classmethod
    def from_config(cls, config: NewtonianScaffoldConfig) -> NewtonianScaffoldSimulator:
        """Construct a simulator from a validated :class:`~gwmock_newtonian.config.NewtonianScaffoldConfig`."""
        return cls(config)

    def _validate_channels(self, detectors: list[str], witnesses: list[str]) -> None:
        """Validate one call's requested channel lists against the configured, fixed sets."""
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if not witnesses:
            raise ValueError("witnesses must contain at least one witness channel.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicates.")
        if len(set(witnesses)) != len(witnesses):
            raise ValueError("witnesses must not contain duplicates.")
        if set(detectors) != set(self._detectors):
            raise ValueError(
                "Changing the detector set after construction is unsupported "
                f"(configured: {sorted(self._detectors)}, requested: {sorted(detectors)})."
            )
        if set(witnesses) != set(self._witnesses):
            raise ValueError(
                "Changing the witness set after construction is unsupported "
                f"(configured: {sorted(self._witnesses)}, requested: {sorted(witnesses)})."
            )

    def generate_joint(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> JointRealization:
        """Generate one simultaneous, correlated strain+witness realization."""
        self._validate_channels(detectors, witnesses)
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        if seed is not None or self._rng is None:
            self.seed = seed if seed is not None else self.seed
            self._rng = create_generator(self.seed)

        innovations = self._rng.standard_normal((n_samples, len(self._channel_order)))
        samples = innovations @ self._cholesky.T

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.witnesses = list(witnesses)

        index = {name: position for position, name in enumerate(self._channel_order)}
        strain = {name: samples[:, index[name]].copy() for name in self.detectors}
        witness = {name: samples[:, index[name]].copy() for name in self.witnesses}
        return JointRealization(
            strain=strain,
            witness=witness,
            channel_metadata=self.channel_metadata,
            provenance=self._provenance().to_dict(),
        )

    def generate_joint_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> Iterator[JointRealization]:
        """Yield simultaneous strain+witness chunks lazily, preserving RNG state.

        Continuation contract: after the first chunk, later chunks reuse the
        same :class:`~numpy.random.Generator` instance without reseeding, so
        concatenating consecutive chunks reproduces exactly the array a
        single :meth:`generate_joint` call over the combined duration would
        return.
        """
        while True:
            yield self.generate_joint(chunk_duration, sampling_frequency, detectors, witnesses, seed)
            seed = None

    def covariance(self) -> JointCovariance:
        """Return the scaffold's exact, closed-form one-sided cross-spectral density."""
        return analytic_covariance(self._channel_order, self._config.coupling, self.sampling_frequency)

    @property
    def channel_metadata(self) -> dict[str, ChannelMetadata]:
        """Return typed metadata for every configured strain and witness channel."""
        metadata: dict[str, ChannelMetadata] = {}
        for name in self._detectors:
            metadata[name] = ChannelMetadata(
                name=name,
                kind=ChannelKind.STRAIN,
                unit=self._config.strain_unit,
                domain=ChannelDomain.TIME,
                sampling_frequency=self.sampling_frequency,
                dtype="float64",
            )
        for name in self._witnesses:
            metadata[name] = ChannelMetadata(
                name=name,
                kind=ChannelKind.WITNESS,
                unit=self._config.witness_unit,
                domain=ChannelDomain.TIME,
                sampling_frequency=self.sampling_frequency,
                dtype="float64",
            )
        return metadata

    def _provenance(self) -> RealizationProvenance:
        """Return the structured provenance for the realization just generated."""
        return RealizationProvenance(
            backend=BACKEND_NAME,
            seed=self.seed,
            coupling=self._config.coupling,
            channel_order=list(self._channel_order),
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
                "coupling": self._config.coupling,
                "channel_order": list(self._channel_order),
                "physical_model": "none",
            },
        }


__all__ = ["BACKEND_NAME", "NewtonianScaffoldSimulator"]
