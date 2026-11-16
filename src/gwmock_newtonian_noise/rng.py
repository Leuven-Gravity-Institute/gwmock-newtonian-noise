"""Deterministic RNG policy for gwmock-newtonian-noise scaffold generation.

Policy:

- The only supported bit generator is :class:`numpy.random.PCG64`, constructed
  through :func:`numpy.random.default_rng`. This matches the convention
  already recorded by ``gwmock_noise``'s public dummy backend provenance
  (``rng_bit_generator: "PCG64"``), so provenance recorded by both packages is
  directly comparable.
- A simulator seeded with an integer ``seed`` is fully deterministic: the same
  ``(config, seed)`` pair always produces bit-identical realizations, and a
  batch realization over duration ``T`` is bit-identical to the concatenation
  of streamed chunks covering the same ``T`` from the same seed (the
  continuation contract the public protocol requires), because both draw from
  one uninterrupted :class:`~numpy.random.Generator` bit stream in call order.
- The physical joint backend (:mod:`gwmock_newtonian_noise.joint`) declares a
  segmented process instead of one uninterrupted stream: segment ``k`` draws
  from :func:`segment_seed_sequence` ``(entropy, k)``, so any sample is fixed by
  the root entropy alone and batch and chunked calls agree for any chunking.
- A simulator seeded with ``None`` draws fresh entropy from the OS on every
  construction and is *not* reproducible; this is recorded verbatim
  (``seed: None``) in provenance so a consumer can tell reproducible and
  non-reproducible realizations apart. The physical joint backend also records
  the OS entropy it drew (``seed_entropy``), which replays the draw as a seed.
"""

from __future__ import annotations

import numpy as np

#: The only bit generator this package's RNG policy supports.
RNG_BIT_GENERATOR = "PCG64"


def create_generator(seed: int | np.random.SeedSequence | None) -> np.random.Generator:
    """Construct the shared innovation generator for one simulator instance.

    Args:
        seed: Integer seed or :class:`numpy.random.SeedSequence` for a
            reproducible generator, or ``None`` for a non-reproducible,
            OS-entropy-seeded generator.

    Returns:
        A :class:`numpy.random.Generator` backed by :data:`RNG_BIT_GENERATOR`.
    """
    return np.random.default_rng(seed)


def segment_seed_sequence(entropy: int, segment: int) -> np.random.SeedSequence:
    """Return the seed sequence of one fixed-length segment of a segmented stream.

    A segmented process (see :class:`~gwmock_newtonian_noise.joint.NewtonianJointSimulator`)
    draws segment ``k`` from the ``k``-th spawned child of
    ``SeedSequence(entropy)``. Every segment is reachable directly, without
    drawing the segments before it, so a batch call and any chunking of a
    stream over the same samples consume identical bit streams.

    Args:
        entropy: Root entropy: the integer seed, or the OS entropy a
            ``SeedSequence(None)`` drew, as recorded in provenance.
        segment: Zero-based segment index.

    Returns:
        The :class:`numpy.random.SeedSequence` of that segment.
    """
    if segment < 0:
        raise ValueError(f"segment must be non-negative, got {segment}.")
    return np.random.SeedSequence(entropy, spawn_key=(segment,))


__all__ = ["RNG_BIT_GENERATOR", "create_generator", "segment_seed_sequence"]
