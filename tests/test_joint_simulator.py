"""Tests for the physical joint strain+witness backend, end to end.

Covers the joint covariance of three strains and N witnesses (Hermitian,
positive-semidefinite, unit-consistent), agreement of covariance-domain and
realization-domain auto/cross spectra *and phases*, reproduction of the declared
seeded segmented process across batch/stream/chunk boundaries, recorded
version/dirty/provenance state, and generation through both the public
``gwmock_noise`` entry-point discovery and the direct analytic API.

Fixtures live in ``conftest.py`` and are UNANCHORED illustrative values.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
from gwmock_noise import available_joint_backend_names, load_joint_backend
from gwmock_noise.simulators.protocol import NoiseSimulator

import gwmock_newtonian
from gwmock_newtonian.adapter import StrainOnlyAdapter
from gwmock_newtonian.joint import BACKEND_NAME, STRAIN_UNIT, NewtonianJointSimulator
from gwmock_newtonian.provenance import SourceState, source_state
from gwmock_newtonian.rng import segment_seed_sequence
from gwmock_newtonian.witness.response import joint_covariance, joint_mode_responses, synthesize_joint

pytestmark = pytest.mark.unit

#: Realizations in the Monte-Carlo comparisons. The per-bin relative standard
#: error of a K-average periodogram is ``1 / sqrt(K) ~= 2.2%``.
_MC_REALIZATIONS = 2000
#: Per-bin sampling sigmas allowed on an auto-spectrum and on a cross phase.
_MC_SIGMAS = 5.0
#: Bound on the normalized cross residual ratio, whose expectation is ~1 (see
#: ``tests/test_witness_response.py`` for its derivation).
_MC_CROSS_RATIO_MAX = 3.0
#: Coherence squared above which a cross phase is scored; the per-entry sigma
#: already widens as coherence falls, so this only drops near-incoherent pairs.
_PHASE_COHERENCE_SQUARED = 0.1
#: Relative floor on eigenvalues of the normalized (coherence) matrix.
_PSD_FLOOR = 1e-10


def _all(simulator: NewtonianJointSimulator):
    """Return the configured detector and witness lists."""
    return list(simulator.detectors), list(simulator.witnesses)


def _stack(realization, order) -> np.ndarray:
    """Return a realization's channels stacked in ``order``."""
    merged = {**realization.strain, **realization.witness}
    return np.stack([merged[name] for name in order])


def _normalized(matrix: np.ndarray) -> np.ndarray:
    """Return ``D^-1/2 C D^-1/2`` per slice: scale-free across strain and witness units."""
    diagonal = np.real(np.diagonal(matrix, axis1=1, axis2=2))
    scale = 1.0 / np.sqrt(diagonal)
    return matrix * scale[:, :, None] * scale[:, None, :]


def _excited(simulator: NewtonianJointSimulator) -> np.ndarray:
    """Return the mask of bins where every channel carries power."""
    diagonal = np.real(np.diagonal(simulator.covariance().matrix, axis1=1, axis2=2))
    return np.all(diagonal > 0.0, axis=1)


def _empirical_csd(simulator: NewtonianJointSimulator, realizations: int) -> np.ndarray:
    """Return the K-average one-sided ``(2 / (fs N)) conj(X_a) X_b`` over seeded segments."""
    order = simulator.covariance().channel_order
    detectors, witnesses = _all(simulator)
    n_samples = simulator.segment_samples
    fs = simulator.sampling_frequency
    accumulator = None
    for seed in range(realizations):
        realization = simulator.generate_joint(simulator.duration, fs, detectors, witnesses, seed=seed)
        spectra = np.fft.rfft(_stack(realization, order), axis=1)
        term = np.einsum("af,bf->fab", spectra.conj(), spectra)
        accumulator = term if accumulator is None else accumulator + term
    return accumulator * (2.0 / (fs * n_samples)) / realizations


class TestJointCovariance:
    """The joint covariance of three strains and N witnesses."""

    def test_shape_and_order_are_three_strains_then_the_witnesses(self, joint_simulator) -> None:
        """The matrix indexes the three detector names first, then every active witness."""
        covariance = joint_simulator.covariance()
        assert covariance.channel_order == ["ET1", "ET2", "ET3", "W0", "W1", "W2", "W3"]
        n_freq = joint_simulator.segment_samples // 2 + 1
        assert covariance.matrix.shape == (n_freq, 7, 7)
        np.testing.assert_array_equal(
            covariance.frequencies,
            np.fft.rfftfreq(joint_simulator.segment_samples, d=1.0 / joint_simulator.sampling_frequency),
        )

    def test_every_slice_is_hermitian(self, joint_simulator) -> None:
        """Each slice equals its own conjugate transpose, relative to the square root of its diagonal products."""
        matrix = joint_simulator.covariance().matrix
        excited = _excited(joint_simulator)
        assert np.count_nonzero(excited) >= 10
        normalized = _normalized(matrix[excited])
        assert np.max(np.abs(normalized - np.conj(np.swapaxes(normalized, 1, 2)))) < 1e-12

    def test_every_slice_is_positive_semidefinite(self, joint_simulator) -> None:
        """The normalized slices have no eigenvalue below a round-off floor, and the DC slice is zero."""
        matrix = joint_simulator.covariance().matrix
        excited = _excited(joint_simulator)
        normalized = _normalized(matrix[excited])
        eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + np.conj(np.swapaxes(normalized, 1, 2))))
        assert np.min(eigenvalues) > -_PSD_FLOOR
        assert not excited[0]
        assert np.max(np.abs(matrix[0])) == 0.0

    def test_strain_witness_coherence_is_nontrivial(self, joint_simulator) -> None:
        """The cross block carries both coherence and complex phase, so the agreement tests can fail."""
        normalized = _normalized(joint_simulator.covariance().matrix[_excited(joint_simulator)])
        cross = normalized[:, :3, 3:]
        assert np.max(np.abs(cross) ** 2) > 0.3
        assert np.max(np.abs(np.angle(cross[np.abs(cross) ** 2 > _PHASE_COHERENCE_SQUARED]))) > 0.3

    def test_channel_units_follow_each_sensor(self, joint_simulator) -> None:
        """Strain is dimensionless; each witness reports the SI unit of its measured quantity."""
        units = {name: meta.unit for name, meta in joint_simulator.channel_metadata.items()}
        assert units == {
            "ET1": STRAIN_UNIT,
            "ET2": STRAIN_UNIT,
            "ET3": STRAIN_UNIT,
            "W0": "m",
            "W1": "m/s",
            "W2": "m/s^2",
            "W3": "m",
        }

    def test_a_unit_change_rescales_only_its_row_and_column(
        self, joint_detector_config, joint_array, joint_array_factory
    ) -> None:
        """Recalibrating W2 into counts (x 2000) scales its row/column by 2000 and nothing else.

        This is the unit-consistency statement: entry ``(a, b)`` is in
        ``unit_a * unit_b / Hz``, so changing one channel's unit by a factor
        ``g`` must multiply exactly that channel's row and column by ``g`` (its
        auto entry by ``g**2``) and leave every other entry -- the strain block
        included -- bit-for-bit as it was.
        """
        gain = 2000.0
        reference = NewtonianJointSimulator(
            joint_detector_config, joint_array_factory(), sampling_frequency=16.0, segment_duration=2.0, seed=1
        )
        counts = NewtonianJointSimulator(
            joint_detector_config,
            joint_array_factory(
                calibration=gain + 0.0j,
                output_unit="counts",
                self_noise_psd=lambda f: gain**2 * joint_array.sensors[2].self_noise(f),
            ),
            sampling_frequency=16.0,
            segment_duration=2.0,
            seed=1,
        )
        assert counts.channel_metadata["W2"].unit == "counts"
        index = counts.covariance().channel_order.index("W2")
        scale = np.ones(7)
        scale[index] = gain
        expected = reference.covariance().matrix * scale[None, :, None] * scale[None, None, :]
        np.testing.assert_allclose(counts.covariance().matrix, expected, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(counts.covariance().matrix[:, :3, :3], reference.covariance().matrix[:, :3, :3])

    def test_sample_variance_equals_the_integrated_psd(self, joint_simulator) -> None:
        """Time-domain variance in ``unit**2`` matches ``sum_k C_aa(f_k) df`` in ``unit**2 / Hz * Hz``.

        For the spectral-representation draw, ``E[x**2] = sum_{k>=1} C_aa(f_k) df``
        exactly (every positive bin, Nyquist included, contributes ``C df``). A
        K-average over seeded segments of N samples has relative standard error
        well under ``sqrt(2 / (K N)) * sqrt(N) = sqrt(2 / K)`` per channel, ~3%
        for K = 2000; 0.15 is a 5-sigma bound fixed before measuring.
        """
        detectors, witnesses = _all(joint_simulator)
        covariance = joint_simulator.covariance()
        df = joint_simulator.sampling_frequency / joint_simulator.segment_samples
        expected = np.real(np.diagonal(covariance.matrix, axis1=1, axis2=2))[1:].sum(axis=0) * df
        variance = np.zeros(7)
        for seed in range(_MC_REALIZATIONS):
            realization = joint_simulator.generate_joint(2.0, 16.0, detectors, witnesses, seed=seed)
            variance += np.mean(_stack(realization, covariance.channel_order) ** 2, axis=1)
        variance /= _MC_REALIZATIONS
        np.testing.assert_allclose(variance, expected, rtol=0.15, atol=0.0)


@pytest.fixture(scope="module")
def ensemble(joint_prototype):
    """Return a backend and its K-average empirical cross-spectral matrix over seeded segments."""
    simulator = copy.deepcopy(joint_prototype)
    return simulator, _empirical_csd(simulator, _MC_REALIZATIONS)


class TestRealizationAgreesWithCovariance:
    """Covariance-domain and realization-domain auto/cross spectra and phases agree."""

    @staticmethod
    def _interior(simulator) -> np.ndarray:
        """Return excited bins strictly between DC and Nyquist."""
        mask = _excited(simulator)
        mask[-1] = False
        return mask

    def test_the_estimator_is_scipy_csd(self, joint_simulator) -> None:
        """The empirical estimator is ``scipy.signal.csd(x_a, x_b)`` -- an external convention anchor."""
        order = joint_simulator.covariance().channel_order
        realization = joint_simulator.generate_joint(2.0, 16.0, *_all(joint_simulator), seed=4)
        samples = _stack(realization, order)
        spectra = np.fft.rfft(samples, axis=1)
        ours = np.einsum("af,bf->fab", spectra.conj(), spectra) * (2.0 / (16.0 * samples.shape[1]))
        interior = self._interior(joint_simulator)
        for a in range(len(order)):
            for b in range(len(order)):
                _, reference = scipy.signal.csd(
                    samples[a],
                    samples[b],
                    fs=16.0,
                    window="boxcar",
                    nperseg=samples.shape[1],
                    detrend=False,
                    scaling="density",
                )
                np.testing.assert_allclose(ours[interior, a, b], reference[interior], rtol=1e-9, atol=0.0)

    def test_auto_spectra_agree_per_bin(self, ensemble) -> None:
        """Every channel's averaged periodogram is within 5 per-bin sigmas of its analytic PSD."""
        simulator, empirical = ensemble
        analytic = simulator.covariance().matrix
        interior = self._interior(simulator)
        ratio = (
            np.real(np.diagonal(empirical, axis1=1, axis2=2))[interior]
            / np.real(np.diagonal(analytic, axis1=1, axis2=2))[interior]
        )
        worst = float(np.max(np.abs(ratio - 1.0)))
        assert worst < _MC_SIGMAS / np.sqrt(_MC_REALIZATIONS), f"worst auto relative error {worst:.4f}"

    def test_cross_spectra_agree_in_magnitude(self, ensemble) -> None:
        """The cross residual normalized by its sampling variance is ~1, not ~K * coherence**2."""
        simulator, empirical = ensemble
        analytic = simulator.covariance().matrix
        interior = self._interior(simulator)
        diagonal = np.real(np.diagonal(analytic, axis1=1, axis2=2))
        variance = diagonal[:, :, None] * diagonal[:, None, :] / _MC_REALIZATIONS
        off_diagonal = ~np.eye(analytic.shape[1], dtype=bool)
        for block, rows, columns in (
            ("strain-witness", slice(0, 3), slice(3, None)),
            ("all", slice(None), slice(None)),
        ):
            residual = np.abs(empirical - analytic)[interior][:, rows, columns] ** 2
            mask = off_diagonal[rows, columns][None, :, :]
            ratio = float(np.sum(residual * mask) / np.sum(variance[interior][:, rows, columns] * mask))
            assert ratio < _MC_CROSS_RATIO_MAX, f"{block} cross residual ratio {ratio:.3f}"

    def test_cross_phases_agree(self, ensemble) -> None:
        """Every coherent cross phase is within 5 sigmas of ``sqrt((1 - g2) / (2 K g2))``.

        This is the test the timing offsets in the fixture exist for: the
        conjugate estimator ``X_a conj(X_b)`` flips every phase and fails it.
        """
        simulator, empirical = ensemble
        analytic = simulator.covariance().matrix
        interior = self._interior(simulator)
        coherence_squared = np.abs(_normalized(analytic[interior])) ** 2
        off_diagonal = ~np.eye(analytic.shape[1], dtype=bool)[None, :, :]
        scored = (coherence_squared > _PHASE_COHERENCE_SQUARED) & (coherence_squared < 1.0 - 1e-9) & off_diagonal
        assert np.count_nonzero(scored[:, :3, 3:]) >= 20
        phase_error = np.abs(np.angle(empirical[interior] * np.conj(analytic[interior])))[scored]
        g2 = coherence_squared[scored]
        sigma = np.sqrt((1.0 - g2) / (2.0 * _MC_REALIZATIONS * g2))
        worst = float(np.max(phase_error / sigma))
        assert worst < _MC_SIGMAS, f"worst cross phase error {worst:.2f} sigma"

    def test_the_conjugate_estimator_is_rejected(self, ensemble) -> None:
        """The phase check discriminates: conjugating the empirical matrix fails it by a wide margin."""
        simulator, empirical = ensemble
        analytic = simulator.covariance().matrix
        interior = self._interior(simulator)
        coherence_squared = np.abs(_normalized(analytic[interior])) ** 2
        off_diagonal = ~np.eye(analytic.shape[1], dtype=bool)[None, :, :]
        scored = (coherence_squared > _PHASE_COHERENCE_SQUARED) & (coherence_squared < 1.0 - 1e-9) & off_diagonal
        phase_error = np.abs(np.angle(np.conj(empirical[interior]) * np.conj(analytic[interior])))[scored]
        g2 = coherence_squared[scored]
        sigma = np.sqrt((1.0 - g2) / (2.0 * _MC_REALIZATIONS * g2))
        assert float(np.max(phase_error / sigma)) > 10 * _MC_SIGMAS


class TestDeclaredSeededProcess:
    """Batch, stream and chunk boundaries reproduce one declared seeded process."""

    def _batch(self, simulator, duration: float, seed: int = 11) -> np.ndarray:
        realization = simulator.generate_joint(duration, 16.0, *_all(simulator), seed=seed)
        return _stack(realization, simulator.covariance().channel_order)

    def test_stream_chunks_concatenate_to_the_batch(self, joint_simulator, joint_prototype) -> None:
        """Segment-aligned, sub-segment and segment-straddling chunkings all equal the batch draw."""
        batch = self._batch(joint_simulator, 6.0)
        order = joint_simulator.covariance().channel_order
        for chunk_duration, count in ((2.0, 3), (0.5, 12), (0.75, 8), (3.0, 2)):
            simulator = copy.deepcopy(joint_prototype)
            stream = simulator.generate_joint_stream(chunk_duration, 16.0, *_all(simulator), seed=11)
            chunks = np.concatenate([_stack(next(stream), order) for _ in range(count)], axis=1)
            np.testing.assert_array_equal(chunks, batch, err_msg=f"chunk_duration={chunk_duration}")

    def test_irregular_continuation_calls_equal_the_batch(self, joint_simulator, joint_prototype) -> None:
        """Unequal ``seed=None`` continuation calls, down to one sample, tile the batch exactly."""
        batch = self._batch(joint_simulator, 6.0)
        simulator = copy.deepcopy(joint_prototype)
        order = simulator.covariance().channel_order
        durations = [0.0625, 1.9375, 0.0625, 2.5, 1.4375]
        pieces = [_stack(simulator.generate_joint(durations[0], 16.0, *_all(simulator), seed=11), order)]
        pieces += [_stack(simulator.generate_joint(d, 16.0, *_all(simulator)), order) for d in durations[1:]]
        np.testing.assert_array_equal(np.concatenate(pieces, axis=1), batch)

    def test_segment_k_is_the_direct_seeded_draw(self, joint_simulator, joint_detector_config, joint_array) -> None:
        """Segment ``k`` of the process is ``synthesize_joint`` seeded by ``segment_seed_sequence(seed, k)``."""
        batch = self._batch(joint_simulator, 6.0, seed=11)
        frequencies = np.fft.rfftfreq(32, d=1.0 / 16.0)
        responses = joint_mode_responses(frequencies, joint_detector_config, joint_array)
        order = joint_simulator.covariance().channel_order
        for segment in (0, 2):
            direct = synthesize_joint(
                responses, joint_detector_config, joint_array, 2.0, 16.0, segment_seed_sequence(11, segment)
            )
            expected = np.stack([direct[name] for name in order])
            np.testing.assert_array_equal(batch[:, 32 * segment : 32 * (segment + 1)], expected)

    def test_segments_are_distinct_draws(self, joint_simulator) -> None:
        """Consecutive segments are not repeats of one periodic draw."""
        batch = self._batch(joint_simulator, 4.0)
        assert not np.array_equal(batch[:, :32], batch[:, 32:])

    def test_a_seed_restarts_and_none_continues(self, joint_simulator) -> None:
        """An explicit seed restarts at sample zero; ``seed=None`` continues the started process."""
        order = joint_simulator.covariance().channel_order
        first = _stack(joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator), seed=5), order)
        following = _stack(joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator)), order)
        restarted = _stack(joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator), seed=5), order)
        np.testing.assert_array_equal(restarted, first)
        assert not np.array_equal(following, first)
        other = _stack(joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator), seed=6), order)
        assert not np.array_equal(other, first)

    def test_unseeded_process_is_replayable_from_recorded_entropy(self, joint_prototype) -> None:
        """``seed=None`` draws OS entropy, records it, and that entropy replays the draw as a seed."""
        unseeded = copy.deepcopy(joint_prototype)
        unseeded.seed = None
        order = unseeded.covariance().channel_order
        realization = unseeded.generate_joint(3.0, 16.0, *_all(unseeded))
        assert realization.provenance["seed"] is None
        entropy = int(realization.provenance["seed_entropy"])
        replay = copy.deepcopy(joint_prototype)
        np.testing.assert_array_equal(
            _stack(replay.generate_joint(3.0, 16.0, *_all(replay), seed=entropy), order), _stack(realization, order)
        )

    def test_provenance_records_the_sample_window(self, joint_simulator) -> None:
        """Each chunk's provenance names the samples of the process it holds."""
        stream = joint_simulator.generate_joint_stream(0.75, 16.0, *_all(joint_simulator), seed=3)
        windows = [(chunk["start_sample"], chunk["n_samples"]) for chunk in (next(stream).provenance for _ in range(4))]
        assert windows == [(0, 12), (12, 12), (24, 12), (36, 12)]

    @pytest.mark.parametrize("duration", [0.03, 1.0 / 3.0])
    def test_non_integer_sample_counts_are_refused(self, joint_simulator, duration: float) -> None:
        """A duration that is not a whole number of samples is refused, not rounded off the grid."""
        with pytest.raises(ValueError, match="integer number of samples"):
            joint_simulator.generate_joint(duration, 16.0, *_all(joint_simulator), seed=1)

    @pytest.mark.parametrize(
        ("detectors", "witnesses", "message"),
        [
            ([], ["W0", "W1", "W2", "W3"], "at least one detector"),
            (["ET1", "ET2", "ET3"], [], "at least one witness"),
            (["ET1", "ET1", "ET2", "ET3"], ["W0", "W1", "W2", "W3"], "detectors must not contain duplicates"),
            (["ET1", "ET2", "ET3"], ["W0", "W0", "W1", "W2", "W3"], "witnesses must not contain duplicates"),
        ],
    )
    def test_malformed_channel_requests_are_refused(self, joint_simulator, detectors, witnesses, message) -> None:
        """Empty or duplicated channel lists are refused before anything is drawn."""
        with pytest.raises(ValueError, match=message):
            joint_simulator.generate_joint(1.0, 16.0, detectors, witnesses, seed=1)

    def test_the_typed_covariance_carries_the_null_projection(self, joint_simulator) -> None:
        """The typed covariance is the protocol matrix, with the null kept as a labelled projection."""
        typed = joint_simulator.analytic_covariance
        np.testing.assert_array_equal(typed.matrix, joint_simulator.covariance().matrix)
        u = typed.geometry.null_vector()
        np.testing.assert_array_equal(typed.null_projection(), np.einsum("i,fij,j->f", u, typed.strain_block(), u))

    def test_changing_the_sampling_frequency_is_refused(self, joint_simulator) -> None:
        """The response matrix is fixed at one sampling frequency."""
        with pytest.raises(ValueError, match="sampling frequency"):
            joint_simulator.generate_joint(1.0, 32.0, *_all(joint_simulator), seed=1)

    def test_reordered_channel_requests_are_accepted(self, joint_simulator, joint_prototype) -> None:
        """Requesting the same channels in another order returns the same per-channel data."""
        detectors, witnesses = _all(joint_simulator)
        forward = joint_simulator.generate_joint(1.0, 16.0, detectors, witnesses, seed=8)
        reverse_simulator = copy.deepcopy(joint_prototype)
        reverse = reverse_simulator.generate_joint(1.0, 16.0, detectors[::-1], witnesses[::-1], seed=8)
        for name in detectors:
            np.testing.assert_array_equal(reverse.strain[name], forward.strain[name])
        for name in witnesses:
            np.testing.assert_array_equal(reverse.witness[name], forward.witness[name])


class TestProvenanceState:
    """Version, dirty and provenance state is recorded with every realization."""

    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(repository), *arguments],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _repository(self, root: Path) -> Path:
        """Create a committed git repository with this package's ``src`` layout; return the package dir."""
        package = root / "src" / "gwmock_newtonian"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        self._git(root, "init", "-q")
        self._git(root, "add", ".")
        self._git(
            root,
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "init",
        )
        return package

    def test_realization_provenance_records_version_and_tree_state(self, joint_simulator) -> None:
        """Provenance carries the package version and the source tree's commit and dirty flag."""
        provenance = joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator), seed=1).provenance
        restored = json.loads(json.dumps(provenance))
        assert restored == provenance
        assert provenance["backend"] == BACKEND_NAME
        assert provenance["package_version"] == gwmock_newtonian.__version__
        assert provenance["gwmock_noise_version"]
        package_dir = Path(gwmock_newtonian.__file__).resolve().parent
        top_level = Path(self._git(package_dir, "rev-parse", "--show-toplevel"))
        assert provenance["source_commit"] == self._git(top_level, "rev-parse", "HEAD")
        assert provenance["source_dirty"] is bool(self._git(top_level, "status", "--porcelain"))
        assert joint_simulator.metadata[BACKEND_NAME]["source_commit"] == provenance["source_commit"]

    def test_source_state_tracks_a_clean_then_dirty_tree(self, tmp_path: Path) -> None:
        """A clean checkout reports its commit; a tracked edit or an untracked file marks it dirty."""
        package = self._repository(tmp_path)
        commit = self._git(tmp_path, "rev-parse", "HEAD")
        assert source_state(package) == SourceState(commit=commit, dirty=False)
        (package / "extra.py").write_text("")
        assert source_state(package) == SourceState(commit=commit, dirty=True)
        (package / "extra.py").unlink()
        (package / "__init__.py").write_text("x = 1\n")
        assert source_state(package) == SourceState(commit=commit, dirty=True)

    def test_source_state_ignores_trees_that_are_not_this_package(self, tmp_path: Path) -> None:
        """No git tree, or an unrelated repository containing an installed copy, reports unknown."""
        plain = tmp_path / "plain" / "gwmock_newtonian"
        plain.mkdir(parents=True)
        assert source_state(plain) == SourceState(commit=None, dirty=None)
        self._repository(tmp_path / "other")
        installed = tmp_path / "other" / "site-packages" / "gwmock_newtonian"
        installed.mkdir(parents=True)
        assert source_state(installed) == SourceState(commit=None, dirty=None)

    def test_fingerprints_identify_the_configuration(self, joint_simulator, joint_detector_config, joint_array_factory):
        """The covariance fingerprint is the matrix's own hash, and a different array changes it."""
        import hashlib

        provenance = joint_simulator.generate_joint(1.0, 16.0, *_all(joint_simulator), seed=1).provenance
        matrix = joint_simulator.covariance().matrix
        assert provenance["covariance_sha256"] == hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
        other = NewtonianJointSimulator(
            joint_detector_config,
            joint_array_factory(timing_offset=0.01),
            sampling_frequency=16.0,
            segment_duration=2.0,
            seed=1,
        )
        other_provenance = other.generate_joint(1.0, 16.0, *_all(other), seed=1).provenance
        for key in ("responses_sha256", "covariance_sha256", "array_manifest_sha256"):
            assert other_provenance[key] != provenance[key], key


class TestGenerationPaths:
    """Generation through ``gwmock_noise`` entry-point discovery and through the direct analytic API."""

    def test_the_backend_is_discovered_by_gwmock_noise(self) -> None:
        """The public discovery surface lists and loads this backend from package metadata alone."""
        assert {"dummy_correlated", "newtonian_scaffold", BACKEND_NAME} <= set(available_joint_backend_names())
        assert load_joint_backend(BACKEND_NAME) is NewtonianJointSimulator

    def test_discovered_backend_reproduces_the_direct_class(self, joint_detector_config, joint_array) -> None:
        """A simulator built from the discovered entry point draws bit-identically to the direct class."""
        discovered = load_joint_backend(BACKEND_NAME)(
            joint_detector_config, joint_array, sampling_frequency=16.0, segment_duration=2.0, seed=9
        )
        direct = NewtonianJointSimulator(
            joint_detector_config, joint_array, sampling_frequency=16.0, segment_duration=2.0, seed=9
        )
        order = direct.covariance().channel_order
        np.testing.assert_array_equal(
            _stack(discovered.generate_joint(5.0, 16.0, *_all(discovered)), order),
            _stack(direct.generate_joint(5.0, 16.0, *_all(direct)), order),
        )
        np.testing.assert_array_equal(discovered.covariance().matrix, direct.covariance().matrix)

    def test_covariance_is_the_direct_analytic_joint_covariance(
        self, joint_simulator, joint_detector_config, joint_array
    ) -> None:
        """``covariance()`` equals the direct ``joint_covariance`` API on the segment grid, bit for bit."""
        frequencies = np.fft.rfftfreq(32, d=1.0 / 16.0)
        direct = joint_covariance(frequencies, joint_detector_config, joint_array)
        covariance = joint_simulator.covariance()
        np.testing.assert_array_equal(covariance.frequencies, direct.frequencies)
        np.testing.assert_array_equal(covariance.matrix, direct.matrix)
        assert tuple(covariance.channel_order) == direct.channel_order
        assert joint_simulator.array_manifest.to_dict() == direct.array_manifest.to_dict()

    def test_strain_only_adapter_serves_legacy_consumers(self, joint_simulator, joint_prototype) -> None:
        """The public-protocol adapter exposes the physical backend as a legacy strain-only ``NoiseSimulator``."""
        adapter = StrainOnlyAdapter(joint_simulator)
        assert isinstance(adapter, NoiseSimulator)
        strain = adapter.generate(4.0, 16.0, list(joint_simulator.detectors), seed=12)
        joint = copy.deepcopy(joint_prototype).generate_joint(4.0, 16.0, *_all(joint_simulator), seed=12)
        assert set(strain) == {"ET1", "ET2", "ET3"}
        for name, series in strain.items():
            np.testing.assert_array_equal(series, joint.strain[name])
        stream = StrainOnlyAdapter(copy.deepcopy(joint_prototype)).generate_stream(
            1.0, 16.0, list(joint_simulator.detectors), seed=12
        )
        streamed = [next(stream) for _ in range(4)]
        for name in strain:
            np.testing.assert_array_equal(np.concatenate([chunk[name] for chunk in streamed]), strain[name])


class TestConstruction:
    """Construction-time validation of the physical backend."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sampling_frequency": 0.0, "segment_duration": 2.0}, "sampling_frequency"),
            ({"sampling_frequency": 16.0, "segment_duration": 0.03}, "integer number of samples"),
            ({"sampling_frequency": 16.0, "segment_duration": 0.0625}, "at least two samples"),
            ({"sampling_frequency": 16.0, "segment_duration": -1.0}, "greater than zero"),
        ],
    )
    def test_rejects_an_invalid_sample_grid(self, joint_detector_config, joint_array, kwargs, message) -> None:
        """The segment must be a whole number of at least two samples at a positive rate."""
        with pytest.raises(ValueError, match=message):
            NewtonianJointSimulator(joint_detector_config, joint_array, **kwargs)

    def test_rejects_an_array_with_no_active_witness(self, joint_detector_config, joint_array) -> None:
        """A joint backend needs at least one witness channel."""
        empty = joint_array.with_missing(joint_array.channel_names)
        with pytest.raises(ValueError, match="active witness"):
            NewtonianJointSimulator(joint_detector_config, empty, sampling_frequency=16.0, segment_duration=2.0)

    def test_joint_covariance_rejects_mismatched_responses(self, joint_detector_config, joint_array) -> None:
        """Precomputed responses on another grid are refused, not silently combined."""
        responses = joint_mode_responses(np.fft.rfftfreq(16, d=1.0 / 16.0), joint_detector_config, joint_array)
        with pytest.raises(ValueError, match="responses must have shape"):
            joint_covariance(np.fft.rfftfreq(32, d=1.0 / 16.0), joint_detector_config, joint_array, responses=responses)
