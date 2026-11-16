"""Joint strain--witness and witness--witness responses, covariances and realizations.

This module ties the witness sensors of
:mod:`gwmock_newtonian_noise.witness.array` to the *same* latent Rayleigh-wave field
that the detector response of :mod:`gwmock_newtonian_noise.detector.response` already
consumes. It does not re-implement the detector: the strain part of every
per-mode response is exactly :func:`~gwmock_newtonian_noise.detector.response.mode_strain`
with the production closed-form plane-wave kernel, so the strain block of the
joint covariance reproduces
:func:`~gwmock_newtonian_noise.detector.response.strain_covariance` entry for entry.

Every channel -- the three detector strains and every active witness -- is a
linear functional of one shared set of latent Rayleigh modes, labelled by
(frequency, azimuth). The per-mode response matrix ``V[f, phi, c]`` is the
response of channel ``c`` to one unit of surface vertical amplitude in mode
``(f, phi)``. The analytic one-sided covariances are the non-negative weighted
sums

    C_ab(f) = sum_phi w_phi S_u(f) V[f, phi, a] conj(V[f, phi, b]),

with ``w_phi`` the normalized azimuth weights and ``S_u(f)`` the one-sided
surface-displacement PSD -- the same construction as the field's and the
detector's covariance, so the joint matrix is Hermitian positive-semidefinite by
construction. Witness self-noise and correlated sensor noise are added to the
witness block only: they are independent of the field and of the strains.

Cross-spectrum sign convention: a channel's time series is
``Re(c exp(-i omega t))`` for a complex mode coefficient ``c``, so the
:func:`numpy.fft.rfft` of a sampled channel is proportional to ``conj(c)``.
``C_ab = E[c_a conj(c_b)]`` is therefore estimated from realizations by
``(2 / (fs N)) E[conj(X_a) X_b]`` -- the convention of
``scipy.signal.csd(x_a, x_b)`` -- and *not* by ``X_a conj(X_b)``, which is its
complex conjugate. The two agree only where the cross-spectrum is real, which an
isotropic field with untimed sensors makes it; a timing offset or an anisotropic
field exposes the difference.

Sampled realizations use the spectral-representation method already used by
:func:`~gwmock_newtonian_noise.seismic.field.realize`: one independent random-phase
complex amplitude is drawn per excited (frequency, azimuth) cell and shared by
every channel, and the deterministic per-mode response alone sets the
correlations. :func:`synthesize_joint` also injects correlated Gaussian sensor
noise with the array's own covariance. A Monte-Carlo average of the realized
periodogram therefore converges to the analytic covariance; the test suite
compares them within a sampling-theory tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian_noise.detector.geometry import TriangleGeometry
from gwmock_newtonian_noise.detector.response import DetectorResponseConfig, _plane_wave_kernel, mode_strain
from gwmock_newtonian_noise.detector.response import strain_covariance as _detector_strain_covariance
from gwmock_newtonian_noise.rng import create_generator
from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig, steering_vectors
from gwmock_newtonian_noise.witness.array import ArrayManifest, WitnessArray
from gwmock_newtonian_noise.witness.array import effective_rank as _effective_rank
from gwmock_newtonian_noise.witness.array import participation_ratio as _participation_ratio

#: Fewest samples a realization can produce: at least a start and end sample.
_MIN_REALIZATION_SAMPLES = 2


def _surface_psd(field: RayleighWaveFieldConfig, frequency: float) -> float:
    """Return the field's one-sided surface vertical displacement PSD at one frequency."""
    value = float(np.asarray(field.surface_displacement_psd(np.array([frequency])), dtype=float)[0])
    if value < 0.0:
        raise ValueError(f"surface_displacement_psd returned a negative value ({value}) at f={frequency}.")
    return value


def _is_excited(field: RayleighWaveFieldConfig, frequency: float) -> bool:
    """Return whether the field carries any power at one strictly-positive frequency."""
    return frequency > 0.0 and _surface_psd(field, frequency) > 0.0


def _azimuth_count(field: RayleighWaveFieldConfig) -> int:
    """Return the number of azimuth quadrature cells."""
    return int(field.azimuth_grid().size)


def witness_mode_responses(
    frequencies: NDArray[np.floating],
    field: RayleighWaveFieldConfig,
    array: WitnessArray,
) -> NDArray[np.complex128]:
    """Return each active witness's complex per-azimuth response to a unit mode.

    The response of a sensor is its full transfer (calibration, instrument
    response, quantity transfer, timing) times the projection of the field's
    displacement steering on its sensing axis. Bins where the field's surface
    PSD is exactly zero are returned as zero: no mode is excited there.

    Args:
        frequencies: Frequency grid in hertz, shape ``(n_freq,)``.
        field: The latent seismic field.
        array: The witness array.

    Returns:
        A complex ``(n_freq, n_azimuth, n_active)`` array.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    sensors = array.active_sensors
    n_sensors = len(sensors)
    n_azimuth = _azimuth_count(field)
    responses = np.zeros((frequencies.size, n_azimuth, n_sensors), dtype=complex)
    if n_sensors == 0:
        return responses
    orientations = np.stack([sensor.unit_orientation for sensor in sensors])
    positions = [sensor.position for sensor in sensors]
    for index, frequency in enumerate(frequencies):
        if not _is_excited(field, float(frequency)):
            continue
        steering = steering_vectors(positions, float(frequency), field)
        combined = np.zeros((n_azimuth, n_sensors), dtype=complex)
        for axis, component in enumerate(("x", "y", "z")):
            combined += steering[component] * orientations[:, axis][None, :]
        transfers = np.array([sensor.transfer(float(frequency)) for sensor in sensors])
        responses[index] = combined * transfers[None, :]
    return responses


def joint_mode_responses(
    frequencies: NDArray[np.floating],
    detector_config: DetectorResponseConfig,
    array: WitnessArray,
) -> NDArray[np.complex128]:
    """Return the strain-plus-witness per-azimuth response to a unit mode.

    Channel order is the detector geometry's names followed by the array's
    active channel names. The strain block is produced by the detector
    response (:func:`~gwmock_newtonian_noise.detector.response.mode_strain` with the
    closed-form plane-wave kernel); the witness block by
    :func:`witness_mode_responses` on the same field.

    Args:
        frequencies: Frequency grid in hertz, shape ``(n_freq,)``.
        detector_config: The detector response configuration (its field is the
            shared latent field the witnesses also observe).
        array: The witness array.

    Returns:
        A complex ``(n_freq, n_azimuth, 3 + n_active)`` array.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    field = detector_config.field
    n_azimuth = _azimuth_count(field)
    n_detector = detector_config.layout.geometry.n_detectors
    n_witness = len(array.active_sensors)
    responses = np.zeros((frequencies.size, n_azimuth, n_detector + n_witness), dtype=complex)
    witness = witness_mode_responses(frequencies, field, array)
    azimuths = field.azimuth_grid()
    for index, frequency in enumerate(frequencies):
        if not _is_excited(field, float(frequency)):
            continue
        kernel = _plane_wave_kernel(detector_config, float(frequency))
        for azimuth_index, azimuth in enumerate(azimuths):
            responses[index, azimuth_index, :n_detector] = mode_strain(
                detector_config, float(frequency), float(azimuth), kernel
            )
    responses[:, :, n_detector:] = witness
    return responses


def _assemble_covariance(
    responses: NDArray[np.complex128],
    frequencies: NDArray[np.floating],
    field: RayleighWaveFieldConfig,
) -> NDArray[np.complex128]:
    """Assemble the field-induced one-sided covariance from per-mode responses."""
    frequencies = np.asarray(frequencies, dtype=float)
    n_channel = responses.shape[2]
    matrix = np.zeros((frequencies.size, n_channel, n_channel), dtype=complex)
    weights = field.azimuth_weights()
    for index, frequency in enumerate(frequencies):
        if not _is_excited(field, float(frequency)):
            continue
        response = responses[index]
        matrix[index] = _surface_psd(field, float(frequency)) * np.einsum(
            "ai,aj,a->ij", response, response.conj(), weights
        )
    return matrix


def _noise_factor(covariance: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return a factor ``L`` with ``L @ L.conj().T == covariance`` via eigen-decomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.conj().T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


@dataclass(frozen=True)
class WitnessCovariance:
    """One-sided covariance of an array's active witness channels.

    Attributes:
        frequencies: Frequency grid in hertz, shape ``(n_freq,)``.
        matrix: ``(n_freq, n, n)`` Hermitian one-sided covariance, including noise.
        channel_order: Active channel names matching ``matrix``'s axes.
        array_manifest: The array manifest at the manifest reference frequency.
    """

    frequencies: NDArray[np.float64]
    matrix: NDArray[np.complex128]
    channel_order: tuple[str, ...]
    array_manifest: ArrayManifest

    def effective_rank(self, rtol: float = 1e-9) -> NDArray[np.int64]:
        """Return the effective rank of the field-plus-noise covariance per frequency.

        This ranks :attr:`matrix`, which includes the seismic field, so a
        noiseless array still has a nonzero rank here. It is not the manifest's
        :attr:`~gwmock_newtonian_noise.witness.array.ArrayManifest.noise_effective_rank`,
        which ranks the noise matrix alone.

        Returns:
            Integer ranks, shape ``(n_freq,)``.
        """
        return np.array([_effective_rank(slice_matrix, rtol=rtol) for slice_matrix in self.matrix], dtype=np.int64)

    def participation_ratio(self) -> NDArray[np.float64]:
        """Return the participation ratio of the field-plus-noise covariance per frequency.

        This is computed over :attr:`matrix`, which includes the seismic field,
        so a noiseless array still has a nonzero ratio here. It is not the
        manifest's
        :attr:`~gwmock_newtonian_noise.witness.array.ArrayManifest.noise_participation_ratio`,
        which is computed over the noise matrix alone.

        Returns:
            Participation ratios, shape ``(n_freq,)``.
        """
        return np.array([_participation_ratio(slice_matrix) for slice_matrix in self.matrix], dtype=float)


@dataclass(frozen=True)
class JointStrainWitnessCovariance:
    """One-sided joint covariance of the three strains and the active witnesses.

    Attributes:
        frequencies: Frequency grid in hertz, shape ``(n_freq,)``.
        matrix: ``(n_freq, 3 + n, 3 + n)`` Hermitian one-sided covariance.
        channel_order: Detector names then witness names.
        detector_channel_order: The three strain channel names.
        witness_channel_order: The active witness channel names.
        geometry: The triangle geometry, so the null stays a labelled projection.
        array_manifest: The array manifest at the manifest reference frequency.
    """

    frequencies: NDArray[np.float64]
    matrix: NDArray[np.complex128]
    channel_order: tuple[str, ...]
    detector_channel_order: tuple[str, ...]
    witness_channel_order: tuple[str, ...]
    geometry: TriangleGeometry
    array_manifest: ArrayManifest

    @property
    def n_detector(self) -> int:
        """Number of strain channels (three)."""
        return len(self.detector_channel_order)

    def strain_block(self) -> NDArray[np.complex128]:
        """Return the ``(n_freq, 3, 3)`` strain sub-covariance."""
        return self.matrix[:, : self.n_detector, : self.n_detector]

    def witness_block(self) -> NDArray[np.complex128]:
        """Return the ``(n_freq, n, n)`` witness sub-covariance."""
        return self.matrix[:, self.n_detector :, self.n_detector :]

    def cross_block(self) -> NDArray[np.complex128]:
        """Return the ``(n_freq, 3, n)`` strain--witness cross-covariance."""
        return self.matrix[:, : self.n_detector, self.n_detector :]

    def effective_rank(self, rtol: float = 1e-9) -> NDArray[np.int64]:
        """Return the effective rank of each full joint slice, shape ``(n_freq,)``."""
        return np.array([_effective_rank(slice_matrix, rtol=rtol) for slice_matrix in self.matrix], dtype=np.int64)

    def null_projection(self) -> NDArray[np.complex128]:
        """Return the strain null-channel PSD ``u^T C_strain u`` per frequency."""
        u = self.geometry.null_vector()
        return np.einsum("i,fij,j->f", u, self.strain_block(), u)


def witness_covariance(
    frequencies: NDArray[np.floating],
    field: RayleighWaveFieldConfig,
    array: WitnessArray,
    manifest_frequency: float = 1.0,
) -> WitnessCovariance:
    """Return the witness-only one-sided covariance including array noise.

    Args:
        frequencies: Frequency grid in hertz.
        field: The latent seismic field.
        array: The witness array.
        manifest_frequency: Frequency at which the manifest is evaluated.

    Returns:
        The :class:`WitnessCovariance`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    responses = witness_mode_responses(frequencies, field, array)
    matrix = _assemble_covariance(responses, frequencies, field)
    n_witness = len(array.active_sensors)
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0 or n_witness == 0:
            continue
        matrix[index] += array.noise_covariance(float(frequency))
    return WitnessCovariance(
        frequencies=frequencies,
        matrix=matrix,
        channel_order=array.channel_names,
        array_manifest=array.manifest(reference_frequency=manifest_frequency),
    )


def joint_covariance(
    frequencies: NDArray[np.floating],
    detector_config: DetectorResponseConfig,
    array: WitnessArray,
    manifest_frequency: float = 1.0,
    *,
    responses: NDArray[np.complex128] | None = None,
) -> JointStrainWitnessCovariance:
    """Return the joint strain--witness one-sided covariance including array noise.

    Noise is added to the witness block only: it is independent of the field
    and of the strain channels.

    Args:
        frequencies: Frequency grid in hertz.
        detector_config: The detector response configuration (its field is the
            shared latent field).
        array: The witness array.
        manifest_frequency: Frequency at which the array manifest is evaluated.
        responses: Precomputed :func:`joint_mode_responses` on ``frequencies``
            for the same configuration and array, or ``None`` to compute them.
            Passing the matrix a realization is synthesized from makes the
            covariance and the realization share one response by construction.

    Returns:
        The :class:`JointStrainWitnessCovariance`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    field = detector_config.field
    geometry = detector_config.layout.geometry
    detector_names = geometry.names
    witness_names = array.channel_names
    if set(detector_names) & set(witness_names):
        raise ValueError(
            f"detector and witness channel names must be disjoint, got {detector_names} and {witness_names}."
        )
    n_detector = geometry.n_detectors
    n_witness = len(witness_names)
    if responses is None:
        responses = joint_mode_responses(frequencies, detector_config, array)
    expected = (frequencies.size, _azimuth_count(field), n_detector + n_witness)
    if responses.shape != expected:
        raise ValueError(f"responses must have shape {expected}, got {responses.shape}.")
    matrix = _assemble_covariance(responses, frequencies, field)
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0 or n_witness == 0:
            continue
        matrix[index, n_detector:, n_detector:] += array.noise_covariance(float(frequency))
    return JointStrainWitnessCovariance(
        frequencies=frequencies,
        matrix=matrix,
        channel_order=(*detector_names, *witness_names),
        detector_channel_order=detector_names,
        witness_channel_order=witness_names,
        geometry=geometry,
        array_manifest=array.manifest(reference_frequency=manifest_frequency),
    )


def synthesize_joint(  # noqa: PLR0913, PLR0917 -- one parameter per physically distinct input.
    responses: NDArray[np.complex128],
    detector_config: DetectorResponseConfig,
    array: WitnessArray,
    duration: float,
    sampling_frequency: float,
    seed: int | np.random.SeedSequence | None,
) -> dict[str, NDArray[np.float64]]:
    """Synthesize a joint strain--witness realization from precomputed responses.

    One independent random-phase complex amplitude is drawn per excited
    (frequency, azimuth) cell and shared by every channel, exactly as in
    :func:`~gwmock_newtonian_noise.seismic.field.realize`; correlated Gaussian sensor
    noise with the array covariance is then added to the witness channels.

    Args:
        responses: The ``(n_freq, n_azimuth, n_channel)`` per-mode responses
            from :func:`joint_mode_responses`, on the same frequency grid that
            ``duration`` and ``sampling_frequency`` produce.
        detector_config: The detector response configuration.
        array: The witness array.
        duration: Realization duration in seconds, strictly positive.
        sampling_frequency: Sampling frequency in hertz, strictly positive.
        seed: Integer seed or :class:`numpy.random.SeedSequence` for
            reproducibility, or ``None``.

    Returns:
        A dict mapping every channel name to a real ``(n_samples,)`` time series.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}.")
    if sampling_frequency <= 0.0:
        raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}.")
    n_samples = round(duration * sampling_frequency)
    if n_samples < _MIN_REALIZATION_SAMPLES:
        raise ValueError("duration and sampling_frequency must produce at least two samples.")
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    if responses.shape[0] != frequencies.size:
        raise ValueError(
            f"responses have {responses.shape[0]} frequency bins but the realization grid has {frequencies.size}."
        )
    field = detector_config.field
    geometry = detector_config.layout.geometry
    n_detector = geometry.n_detectors
    witness_names = array.channel_names
    n_witness = len(witness_names)
    channel_order = (*geometry.names, *witness_names)

    rng = create_generator(seed)
    weights = field.azimuth_weights()
    n_azimuth = _azimuth_count(field)
    frequency_resolution = sampling_frequency / n_samples
    times = np.arange(n_samples) / sampling_frequency
    series = np.zeros((len(channel_order), n_samples), dtype=float)
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0:
            continue
        phase = np.exp(-1j * 2.0 * np.pi * frequency * times)
        if _is_excited(field, float(frequency)):
            surface_psd = _surface_psd(field, float(frequency))
            theta = rng.uniform(0.0, 2.0 * np.pi, size=n_azimuth)
            amplitude = np.sqrt(2.0 * surface_psd * weights * frequency_resolution) * np.exp(1j * theta)
            coefficients = amplitude @ responses[index]
            series += np.real(np.outer(coefficients, phase))
        if n_witness == 0:
            continue
        noise_covariance = array.noise_covariance(float(frequency))
        if not np.any(noise_covariance):
            continue
        factor = _noise_factor(noise_covariance)
        white = (rng.standard_normal(n_witness) + 1j * rng.standard_normal(n_witness)) / np.sqrt(2.0)
        noise_coefficients = np.sqrt(2.0 * frequency_resolution) * (factor @ white)
        series[n_detector:] += np.real(np.outer(noise_coefficients, phase))
    return {name: series[index] for index, name in enumerate(channel_order)}


def realize_joint(
    detector_config: DetectorResponseConfig,
    array: WitnessArray,
    duration: float,
    sampling_frequency: float,
    seed: int | np.random.SeedSequence | None,
) -> dict[str, NDArray[np.float64]]:
    """Synthesize a joint strain--witness realization, computing the responses.

    A convenience wrapper over :func:`joint_mode_responses` and
    :func:`synthesize_joint`; call those directly to reuse one response matrix
    across many seeds.

    Args:
        detector_config: The detector response configuration.
        array: The witness array.
        duration: Realization duration in seconds, strictly positive.
        sampling_frequency: Sampling frequency in hertz, strictly positive.
        seed: Integer seed or :class:`numpy.random.SeedSequence` for
            reproducibility, or ``None``.

    Returns:
        A dict mapping every channel name to a real ``(n_samples,)`` time series.
    """
    n_samples = round(duration * sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    responses = joint_mode_responses(frequencies, detector_config, array)
    return synthesize_joint(responses, detector_config, array, duration, sampling_frequency, seed)


def joint_strain_covariance_block(
    frequencies: NDArray[np.floating], detector_config: DetectorResponseConfig
) -> NDArray[np.complex128]:
    """Return the detector's own strain covariance matrix (the detector response).

    This is a thin read-through to
    :func:`~gwmock_newtonian_noise.detector.response.strain_covariance`, exposed so
    callers can confirm the strain block of :func:`joint_covariance` is the
    unmodified detector operator.

    Args:
        frequencies: Frequency grid in hertz.
        detector_config: The detector response configuration.

    Returns:
        The ``(n_freq, 3, 3)`` one-sided strain covariance.
    """
    return _detector_strain_covariance(frequencies, detector_config).matrix


__all__ = [
    "JointStrainWitnessCovariance",
    "WitnessCovariance",
    "joint_covariance",
    "joint_mode_responses",
    "joint_strain_covariance_block",
    "realize_joint",
    "synthesize_joint",
    "witness_covariance",
    "witness_mode_responses",
]
