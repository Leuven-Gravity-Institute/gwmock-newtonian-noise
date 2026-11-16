"""Compatibility adapter: legacy strain-only view of a joint simulator.

Existing ``gwmock_noise.NoiseSimulator`` consumers only know about strain
channels. This adapter wraps any protocol-conformant
:class:`~gwmock_noise.JointStrainWitnessSimulator` and exposes it through the
legacy ``NoiseSimulator`` structural protocol, dropping witness channels,
``channel_metadata`` and ``provenance`` from the return value -- the wrapped
joint simulator itself is unchanged and still available for this project's own
pipeline to consume jointly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from gwmock_noise.simulators.joint_protocol import JointStrainWitnessSimulator


class StrainOnlyAdapter:
    """Adapts a :class:`~gwmock_noise.JointStrainWitnessSimulator` to the legacy ``NoiseSimulator`` protocol."""

    def __init__(self, joint_simulator: JointStrainWitnessSimulator) -> None:
        """Wrap a protocol-conformant joint simulator.

        Args:
            joint_simulator: A simulator satisfying
                ``gwmock_noise.JointStrainWitnessSimulator``.

        Raises:
            TypeError: If ``joint_simulator`` does not satisfy the protocol.
        """
        if not isinstance(joint_simulator, JointStrainWitnessSimulator):
            raise TypeError(
                "StrainOnlyAdapter requires a gwmock_noise.JointStrainWitnessSimulator-conformant "
                f"instance, got {type(joint_simulator)!r}."
            )
        self._joint = joint_simulator

    @property
    def duration(self) -> float:
        """Nominal generation duration in seconds."""
        return self._joint.duration

    @property
    def sampling_frequency(self) -> float:
        """Sampling frequency in hertz."""
        return self._joint.sampling_frequency

    @property
    def detectors(self) -> list[str]:
        """Strain channel names."""
        return self._joint.detectors

    @property
    def seed(self) -> int | None:
        """Seed of the wrapped joint simulator."""
        return self._joint.seed

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector strain arrays only, dropping witness channels."""
        realization = self._joint.generate_joint(duration, sampling_frequency, detectors, self._joint.witnesses, seed)
        return realization.strain

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield per-detector strain chunks only, dropping witness channels."""
        for realization in self._joint.generate_joint_stream(
            chunk_duration, sampling_frequency, detectors, self._joint.witnesses, seed
        ):
            yield realization.strain

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the wrapped joint simulator's metadata."""
        return self._joint.metadata


__all__ = ["StrainOnlyAdapter"]
