"""Config rejection tests for NewtonianScaffoldConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gwmock_newtonian_noise.config import NewtonianScaffoldConfig

pytestmark = pytest.mark.unit


def _config(**overrides):
    defaults = {
        "detectors": ["ET1", "ET2"],
        "witnesses": ["SEIS1"],
        "sampling_frequency": 64.0,
        "duration": 2.0,
        "seed": 1,
        "coupling": 0.3,
    }
    defaults.update(overrides)
    return NewtonianScaffoldConfig(**defaults)


class TestValidConfigIsAccepted:
    """The baseline config the rejection tests each perturb one field of."""

    def test_defaults_construct(self) -> None:
        """A well-formed config constructs without error."""
        config = _config()
        assert config.detectors == ("ET1", "ET2")
        assert config.witnesses == ("SEIS1",)

    def test_config_is_frozen(self) -> None:
        """A validated config cannot be mutated after construction."""
        config = _config()
        with pytest.raises(ValidationError):
            config.coupling = 0.5


class TestFrozenContainersAreReallyImmutable:
    """Container mutation, not just field reassignment, must be blocked.

    ``frozen=True`` blocks field *reassignment*; it does not, by itself, stop
    in-place mutation of a mutable container a frozen field happens to hold.
    ``detectors``/``witnesses`` are ``tuple[str, ...]`` specifically so the
    frozen contract holds for the container itself, not just the field name.
    """

    def test_caller_owned_list_mutation_does_not_reach_the_config(self) -> None:
        """Constructing from a caller-owned list copies it; mutating the caller's list afterwards is invisible."""
        caller_detectors = ["ET1", "ET2"]
        caller_witnesses = ["SEIS1"]
        config = _config(detectors=caller_detectors, witnesses=caller_witnesses)

        caller_detectors.append("MUTATED")
        caller_witnesses.append("MUTATED")

        assert "MUTATED" not in config.detectors
        assert "MUTATED" not in config.witnesses
        assert config.detectors == ("ET1", "ET2")
        assert config.witnesses == ("SEIS1",)

    def test_mutating_the_configs_own_detectors_container_raises(self) -> None:
        """The frozen contract must hold for the container itself, not just for `config.detectors = ...`."""
        config = _config()
        with pytest.raises(AttributeError):
            config.detectors.append("MUTABLE")

    def test_mutating_the_configs_own_witnesses_container_raises(self) -> None:
        """Same guarantee for `witnesses`."""
        config = _config()
        with pytest.raises(AttributeError):
            config.witnesses.append("MUTABLE")


class TestChannelListRejection:
    """Detectors/witnesses must be non-empty, unique, and disjoint."""

    def test_empty_detectors_rejected(self) -> None:
        """An empty detector list is not a valid scaffold config."""
        with pytest.raises(ValidationError, match="detectors"):
            _config(detectors=[])

    def test_empty_witnesses_rejected(self) -> None:
        """An empty witness list is not a valid scaffold config."""
        with pytest.raises(ValidationError, match="witnesses"):
            _config(witnesses=[])

    def test_duplicate_detectors_rejected(self) -> None:
        """A detector name repeated in the list is rejected."""
        with pytest.raises(ValidationError, match="duplicate"):
            _config(detectors=["ET1", "ET1"])

    def test_duplicate_witnesses_rejected(self) -> None:
        """A witness name repeated in the list is rejected."""
        with pytest.raises(ValidationError, match="duplicate"):
            _config(witnesses=["SEIS1", "SEIS1"])

    def test_overlapping_detector_and_witness_names_rejected(self) -> None:
        """A name used as both a detector and a witness is rejected."""
        with pytest.raises(ValidationError, match="not share channel names"):
            _config(detectors=["ET1", "SHARED"], witnesses=["SHARED"])


class TestNumericFieldRejection:
    """sampling_frequency, duration, and coupling each have a validity range."""

    @pytest.mark.parametrize("sampling_frequency", [0.0, -1.0, -64.0])
    def test_non_positive_sampling_frequency_rejected(self, sampling_frequency: float) -> None:
        """A non-positive sampling frequency is not physically meaningful."""
        with pytest.raises(ValidationError, match="sampling_frequency"):
            _config(sampling_frequency=sampling_frequency)

    @pytest.mark.parametrize("duration", [0.0, -1.0, -2.0])
    def test_non_positive_duration_rejected(self, duration: float) -> None:
        """A non-positive duration cannot produce any samples."""
        with pytest.raises(ValidationError, match="duration"):
            _config(duration=duration)

    @pytest.mark.parametrize("coupling", [-0.1, 1.0, 1.5])
    def test_out_of_range_coupling_rejected(self, coupling: float) -> None:
        """A coupling outside [0, 1) would make the covariance matrix invalid."""
        with pytest.raises(ValidationError, match="coupling"):
            _config(coupling=coupling)


class TestUnknownFieldsRejected:
    """The schema is closed: an unrecognized key is a config-authoring mistake, not silently ignored."""

    def test_unknown_field_rejected(self) -> None:
        """A typo'd or removed config key is refused rather than silently dropped."""
        with pytest.raises(ValidationError):
            _config(not_a_real_field=1)
