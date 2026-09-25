"""Machine-readable provenance for one joint realization.

``JointRealization.provenance`` is, in the public protocol, an
implementation-defined ``dict[str, Any]``. This module gives this package's
own realizations a versioned, typed shape instead of an ad hoc mapping, so a
downstream consumer (or this project's own pipeline) can parse it reliably.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gwmock_newtonian.rng import RNG_BIT_GENERATOR
from gwmock_newtonian.version import __version__

#: Schema version for :class:`RealizationProvenance`. Bump whenever a field is
#: added, removed, or its meaning changes.
PROVENANCE_SCHEMA_VERSION = 1


class RealizationProvenance(BaseModel):
    """Structured, JSON-serializable provenance for one joint realization.

    ``channel_order`` is ``tuple[str, ...]``, not ``list[str]``: ``frozen=True``
    only blocks field reassignment, not in-place mutation of a mutable
    container a frozen field happens to hold. A tuple has no mutating
    methods, so the frozen contract holds for the container itself.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = PROVENANCE_SCHEMA_VERSION
    backend: str
    package: str = "gwmock_newtonian"
    package_version: str = __version__
    seed: int | None
    rng_bit_generator: str = RNG_BIT_GENERATOR
    coupling: float
    channel_order: tuple[str, ...]
    physical_model: str = Field(
        default="none",
        description=(
            'Physical model identifier. "none" for the structural scaffold, which models no '
            "Newtonian-noise or seismic physics."
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` for ``JointRealization.provenance``."""
        return self.model_dump(mode="json")


__all__ = ["PROVENANCE_SCHEMA_VERSION", "RealizationProvenance"]
