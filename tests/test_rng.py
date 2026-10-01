"""Tests for the deterministic RNG policy."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian.rng import RNG_BIT_GENERATOR, create_generator, segment_seed_sequence

pytestmark = pytest.mark.unit


def test_bit_generator_is_pcg64() -> None:
    """The declared policy names PCG64; the constructed generator must actually use it."""
    assert RNG_BIT_GENERATOR == "PCG64"
    generator = create_generator(seed=0)
    assert type(generator.bit_generator).__name__ == RNG_BIT_GENERATOR


def test_same_seed_is_deterministic() -> None:
    """Two generators built from the same seed draw the identical sequence."""
    first = create_generator(seed=42)
    second = create_generator(seed=42)
    np.testing.assert_array_equal(first.standard_normal(16), second.standard_normal(16))


def test_different_seeds_diverge() -> None:
    """Two generators built from different seeds draw different sequences."""
    first = create_generator(seed=1)
    second = create_generator(seed=2)
    assert not np.array_equal(first.standard_normal(16), second.standard_normal(16))


def test_none_seed_still_produces_usable_generator() -> None:
    """A ``None`` seed is a valid, if non-reproducible, generator."""
    generator = create_generator(seed=None)
    draw = generator.standard_normal(4)
    assert draw.shape == (4,)


def test_continuation_matches_one_shot_draw() -> None:
    """Two sequential draws from one generator equal one draw of the combined size.

    This is the property the streaming continuation contract in
    :mod:`gwmock_newtonian.simulator` relies on.
    """
    continued = create_generator(seed=7)
    first_chunk = continued.standard_normal(5)
    second_chunk = continued.standard_normal(5)

    one_shot = create_generator(seed=7)
    combined = one_shot.standard_normal(10)

    np.testing.assert_array_equal(np.concatenate([first_chunk, second_chunk]), combined)


def test_segment_seed_sequences_are_independent_children_of_the_root() -> None:
    """Segment ``k`` is the ``k``-th spawned child of ``SeedSequence(entropy)``, reachable directly."""
    children = np.random.SeedSequence(5).spawn(3)
    for segment, child in enumerate(children):
        np.testing.assert_array_equal(segment_seed_sequence(5, segment).generate_state(4), child.generate_state(4))
    draws = [create_generator(segment_seed_sequence(5, k)).standard_normal(8) for k in range(3)]
    assert not np.array_equal(draws[0], draws[1])


def test_segment_seed_sequence_rejects_a_negative_segment() -> None:
    """There is no segment before the first."""
    with pytest.raises(ValueError, match="non-negative"):
        segment_seed_sequence(5, -1)
