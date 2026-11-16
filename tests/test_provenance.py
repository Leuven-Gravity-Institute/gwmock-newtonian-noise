"""Tests for RealizationProvenance: frozen-container immutability and JSON shape.

Companion to the same class of fix in ``tests/test_config.py`` --
``RealizationProvenance`` is the second (and last, per an audit of the
package's pydantic models) frozen pydantic model in this package that held a mutable
``list[str]`` field (``channel_order``) despite ``frozen=True``.
"""

from __future__ import annotations

import pytest

from gwmock_newtonian_noise.provenance import PROVENANCE_SCHEMA_VERSION, RealizationProvenance

pytestmark = pytest.mark.unit


def _provenance(**overrides):
    defaults = {
        "backend": "newtonian_scaffold",
        "seed": 1,
        "coupling": 0.3,
        "channel_order": ["ET1", "ET2", "SEIS1"],
    }
    defaults.update(overrides)
    return RealizationProvenance(**defaults)


class TestDefaults:
    """Baseline construction and field defaults."""

    def test_defaults_construct(self) -> None:
        """A well-formed provenance record constructs without error."""
        provenance = _provenance()
        assert provenance.channel_order == ("ET1", "ET2", "SEIS1")
        assert provenance.schema_version == PROVENANCE_SCHEMA_VERSION
        assert provenance.physical_model == "none"


class TestFrozenContainerIsReallyImmutable:
    """Container mutation, not just field reassignment, must be blocked.

    ``frozen=True`` blocks field reassignment; ``channel_order`` is a tuple
    so the frozen contract also holds for the container itself.
    """

    def test_caller_owned_list_mutation_does_not_reach_the_provenance(self) -> None:
        """Constructing from a caller-owned list copies it; mutating the caller's list afterwards is invisible."""
        caller_order = ["ET1", "ET2"]
        provenance = _provenance(channel_order=caller_order)

        caller_order.append("MUTATED")

        assert "MUTATED" not in provenance.channel_order
        assert provenance.channel_order == ("ET1", "ET2")

    def test_mutating_the_provenance_own_container_raises(self) -> None:
        """The frozen contract must hold for the container itself, not just for `provenance.channel_order = ...`."""
        provenance = _provenance()
        with pytest.raises(AttributeError):
            provenance.channel_order.append("MUTABLE")


class TestToDictIsMachineReadable:
    """to_dict() feeds JointRealization.provenance, an implementation-defined dict[str, Any]."""

    def test_channel_order_serializes_to_a_plain_list(self) -> None:
        """JSON has no tuple type; the tuple field must still serialize to a plain, JSON-native list."""
        payload = _provenance().to_dict()
        assert payload["channel_order"] == ["ET1", "ET2", "SEIS1"]
        assert isinstance(payload["channel_order"], list)

    def test_to_dict_round_trips_through_json(self) -> None:
        """The whole payload survives json.dumps/json.loads, since it is written to disk verbatim."""
        import json

        payload = _provenance().to_dict()
        restored = json.loads(json.dumps(payload))
        assert restored == payload
