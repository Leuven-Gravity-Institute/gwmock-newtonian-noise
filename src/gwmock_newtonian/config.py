"""Validated configuration schema for the Newtonian-noise scaffold simulator.

This is a *structural* scaffold config only: ``coupling`` parameterizes a
placeholder equicorrelation statistical model, not a physical Newtonian-noise
coupling. The physical configuration schema for the seismic field and
detector/witness response is the physical joint backend's (:mod:`gwmock_newtonian.joint`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator


class NewtonianScaffoldConfig(BaseModel):
    """Validated configuration for :class:`~gwmock_newtonian.simulator.NewtonianScaffoldSimulator`.

    ``detectors`` and ``witnesses`` are ``tuple[str, ...]``, not ``list[str]``:
    ``frozen=True`` only blocks field *reassignment* (``config.detectors = ...``
    raises), it does not stop in-place mutation of a mutable container a frozen
    field happens to hold. A tuple has no mutating methods, so the frozen
    contract holds for the container itself, not just for the attribute name.

    Attributes:
        detectors: Strain channel names, non-empty and unique.
        witnesses: Witness channel names, non-empty, unique, and disjoint from
            ``detectors``.
        sampling_frequency: Sampling frequency in hertz, must be positive.
        duration: Nominal generation duration in seconds, must be positive.
        seed: Seed for the deterministic RNG policy (see
            :mod:`gwmock_newtonian.rng`), or ``None`` for a non-reproducible
            generator.
        coupling: Equicorrelation coefficient of the placeholder statistical
            model, applied between every channel pair. Must satisfy
            ``0.0 <= coupling < 1.0``.
        strain_unit: Unit string recorded for strain channels.
        witness_unit: Unit string recorded for witness channels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detectors: tuple[str, ...]
    witnesses: tuple[str, ...]
    sampling_frequency: float
    duration: float
    seed: int | None = None
    coupling: float = 0.3
    strain_unit: str = "strain"
    witness_unit: str = "counts"

    @field_validator("detectors", "witnesses")
    @classmethod
    def _non_empty_and_unique(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        """Reject an empty or duplicate-containing channel list."""
        if not value:
            raise ValueError(f"{info.field_name} must contain at least one channel name.")
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} must not contain duplicate channel names.")
        return value

    @field_validator("sampling_frequency", "duration")
    @classmethod
    def _positive(cls, value: float, info: ValidationInfo) -> float:
        """Reject a non-positive sampling frequency or duration."""
        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than zero.")
        return value

    @field_validator("coupling")
    @classmethod
    def _coupling_in_range(cls, value: float) -> float:
        """Reject a coupling coefficient that would make the covariance matrix singular or invalid."""
        if not (0.0 <= value < 1.0):
            raise ValueError("coupling must satisfy 0.0 <= coupling < 1.0.")
        return value

    @model_validator(mode="after")
    def _disjoint_channel_sets(self) -> NewtonianScaffoldConfig:
        """Reject a config where a name is used as both a detector and a witness."""
        overlap = set(self.detectors) & set(self.witnesses)
        if overlap:
            raise ValueError(f"detectors and witnesses must not share channel names: {sorted(overlap)}")
        return self


__all__ = ["NewtonianScaffoldConfig"]
