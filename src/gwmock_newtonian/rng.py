"""Deterministic RNG policy for gwmock-newtonian scaffold generation.

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
- A simulator seeded with ``None`` draws fresh entropy from the OS on every
  construction and is *not* reproducible; this is recorded verbatim
  (``seed: None``) in provenance so a consumer can tell reproducible and
  non-reproducible realizations apart.
"""

from __future__ import annotations

import numpy as np

#: The only bit generator this package's RNG policy supports.
RNG_BIT_GENERATOR = "PCG64"


def create_generator(seed: int | None) -> np.random.Generator:
    """Construct the shared innovation generator for one simulator instance.

    Args:
        seed: Integer seed for a reproducible generator, or ``None`` for a
            non-reproducible, OS-entropy-seeded generator.

    Returns:
        A :class:`numpy.random.Generator` backed by :data:`RNG_BIT_GENERATOR`.
    """
    return np.random.default_rng(seed)


__all__ = ["RNG_BIT_GENERATOR", "create_generator"]
