"""Machine-readable provenance for one joint realization.

``JointRealization.provenance`` is, in the public protocol, an
implementation-defined ``dict[str, Any]``. This module gives this package's
own realizations a versioned, typed shape instead of an ad hoc mapping, so a
downstream consumer (or this project's own pipeline) can parse it reliably.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gwmock_noise.version import __version__ as _gwmock_noise_version
from pydantic import BaseModel, ConfigDict, Field

from gwmock_newtonian_noise.rng import RNG_BIT_GENERATOR
from gwmock_newtonian_noise.version import __version__

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
    package: str = "gwmock_newtonian_noise"
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


#: Schema version for :class:`JointRealizationProvenance`. Bump whenever a
#: field is added, removed, or its meaning changes.
JOINT_PROVENANCE_SCHEMA_VERSION = 1

#: Timeout in seconds for each git query made by :func:`source_state`.
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class SourceState:
    """Version-control state of the source tree the package was imported from.

    The installed ``package_version`` is fixed when the package is installed;
    an editable install keeps reporting it after the tree has moved on or been
    edited. This records the tree itself at the time it is queried.

    Attributes:
        commit: The checked-out commit, or ``None`` when the package was not
            imported from a git checkout of this package.
        dirty: Whether the checkout has uncommitted changes to tracked files or
            untracked, non-ignored files; ``None`` when ``commit`` is ``None``.
    """

    commit: str | None
    dirty: bool | None


def _git(git: str, repository: Path, *arguments: str) -> str:
    """Run one git query in ``repository`` and return its stripped standard output."""
    completed = subprocess.run(  # noqa: S603 -- fixed git argument list, no shell.
        [git, "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    )
    return completed.stdout.strip()


def source_state(package_dir: Path | None = None) -> SourceState:
    """Return the git commit and dirty flag of the tree this package lives in.

    The tree is identified as this package's own checkout only when its
    top level contains ``src/gwmock_newtonian_noise`` at ``package_dir``; an
    installed copy that happens to sit inside some other repository reports
    ``SourceState(None, None)`` rather than that repository's state.

    Args:
        package_dir: The package directory to inspect; defaults to the directory
            of the imported :mod:`gwmock_newtonian_noise` package.

    Returns:
        The :class:`SourceState`.
    """
    package_dir = (package_dir or Path(__file__).parent).resolve()
    git = shutil.which("git")
    unknown = SourceState(commit=None, dirty=None)
    if git is None:
        return unknown
    try:
        top_level = Path(_git(git, package_dir, "rev-parse", "--show-toplevel")).resolve()
        if top_level / "src" / "gwmock_newtonian_noise" != package_dir:
            return unknown
        commit = _git(git, top_level, "rev-parse", "HEAD")
        status = _git(git, top_level, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return unknown
    return SourceState(commit=commit, dirty=bool(status))


class JointRealizationProvenance(BaseModel):
    """Structured, JSON-serializable provenance for one physical joint realization.

    Records everything a realization depends on: the package version and the
    source tree's commit and dirty state, the seeded segmented process (root
    entropy, segment length and the samples drawn), and fingerprints of the
    per-mode response and analytic covariance the realization was synthesized
    from. Only scalars and short name lists appear; no channel data does.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = JOINT_PROVENANCE_SCHEMA_VERSION
    backend: str
    package: str = "gwmock_newtonian_noise"
    package_version: str = __version__
    gwmock_noise_version: str = _gwmock_noise_version
    source_commit: str | None
    source_dirty: bool | None
    physical_model: str
    seed: int | None
    seed_entropy: str = Field(
        description=(
            "Root SeedSequence entropy as a decimal string: equal to the seed when one is given, "
            "otherwise the OS entropy drawn, so even an unseeded realization can be replayed."
        )
    )
    rng_bit_generator: str = RNG_BIT_GENERATOR
    rng_stream: str = "SeedSequence(seed_entropy, spawn_key=(segment,)) per segment"
    sampling_frequency: float
    segment_duration: float
    segment_samples: int
    start_sample: int
    n_samples: int
    channel_order: tuple[str, ...]
    detector_channels: tuple[str, ...]
    witness_channels: tuple[str, ...]
    responses_sha256: str
    covariance_sha256: str
    array_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` for ``JointRealization.provenance``."""
        return self.model_dump(mode="json")


__all__ = [
    "JOINT_PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "JointRealizationProvenance",
    "RealizationProvenance",
    "SourceState",
    "source_state",
]
