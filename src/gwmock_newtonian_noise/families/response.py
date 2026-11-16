"""Generic strain, witness and joint responses of any field family.

Given a :class:`~gwmock_newtonian_noise.families.base.FieldFamily`, this module builds
exactly what :mod:`gwmock_newtonian_noise.witness.response` builds for the minimal
Rayleigh field, with the family's modes in place of the azimuth cells:

- the strain response of mode ``m``: the Newtonian acceleration of every test
  mass (plus the cavity-wall term, if a :class:`CavityModel` is given) through
  the layout's suspension and explicit arm contraction
  (:meth:`~gwmock_newtonian_noise.detector.masses.TestMassLayout.strain_from_accelerations`);
- the witness response of mode ``m``: the displacement at each active sensor
  projected on its sensing axis, times the sensor's full transfer;
- the one-sided joint covariance ``C(f) = sum_m P_m(f) V_m V_m^dagger`` plus the
  array noise in the witness block, Hermitian positive-semidefinite by
  construction and returned as the existing
  :class:`~gwmock_newtonian_noise.witness.response.JointStrainWitnessCovariance`;
- seeded realizations by the same spectral-representation draw (one independent
  random phase per excited mode, shared by every channel) and the same
  ``Re(c exp(-i omega t))`` channel convention, so the cross-spectrum convention
  of :mod:`gwmock_newtonian_noise.witness.response` carries over unchanged.

Cavity term. A test mass in an underground hall sits in a cavity: the medium
that would fill it is missing. For a cavity small compared with the seismic
wavelength the missing material, displaced by the local free-field displacement
``u(r0)``, is a dipole layer on the cavity wall whose field at the centre of a
spherical cavity is ``-(4 pi / 3) G rho0 u(r0)``, independent of the radius at
leading order; the bulk
density that the cavity removes contributes only at relative order
``(k a)^2``. Added to the infinite-medium P coupling ``4 pi G rho0 u_P`` this
gives ``(8 pi / 3) G rho0 u_P - (4 pi / 3) G rho0 u_S``, the cavity coefficient
pair reported for underground body waves by Harms (2019, Living Reviews in
Relativity 22, 6) and Badaracco and Harms (2019, CQG, DOI
10.1088/1361-6382/ab28c1). Those texts were not re-read for this
implementation; the test suite anchors the term, sign included, to a direct
real-space sum of the bulk outside the cavity plus the wall layer, whose
difference from the model scales as the expected ``(k a)^2 / 10``. The term is
applied to every family the same way, from that family's own displacement and
density at the test mass -- which is how Love and SH waves, which source no
field-intrinsic gravity, still reach the strain. Not modelled: scattering of the
seismic field by the cavity itself, non-spherical halls and tunnels.

UNANCHORED: the cavity radius and whether there is a cavity at all are site
geometry; :class:`FamilyJointModel` requires the caller to state it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian_noise.detector.masses import TestMassLayout
from gwmock_newtonian_noise.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian_noise.families.base import FieldFamily, as_positions, require_positive_frequency
from gwmock_newtonian_noise.rng import create_generator
from gwmock_newtonian_noise.witness.array import WitnessArray
from gwmock_newtonian_noise.witness.response import JointStrainWitnessCovariance, _noise_factor

#: Fewest samples a realization can produce: at least a start and end sample.
_MIN_REALIZATION_SAMPLES = 2

#: Leading-order cavity-wall coupling of a small spherical cavity, in units of ``G rho0``.
CAVITY_WALL_COEFFICIENT = -4.0 * np.pi / 3.0


@dataclass(frozen=True)
class CavityModel:
    """A small spherical cavity centred on every test mass.

    Attributes:
        radius: Cavity radius in metres (UNANCHORED). Enters only the validity
            checks: the leading-order wall term is radius-independent, and is
            accurate while ``k * radius << 1`` for the shortest wavelength used.
            Every test mass must lie deeper than ``radius`` so the cavity is
            inside the medium (below the plane ``z = 0``).
    """

    radius: float

    def __post_init__(self) -> None:
        """Reject a non-positive radius."""
        if not (np.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError(f"radius must be positive and finite, got {self.radius}.")

    def acceleration(
        self, displacement: NDArray[np.complex128], density: NDArray[np.float64], positions: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Return the cavity-wall acceleration ``-(4 pi/3) G rho0 u`` per mode and position.

        Args:
            displacement: ``(n_modes, n_positions, 3)`` free-field displacement.
            density: ``(n_positions,)`` background density at each position.
            positions: ``(n_positions, 3)`` cavity centres.

        Returns:
            The ``(n_modes, n_positions, 3)`` acceleration in metres/second^2 per metre.
        """
        if np.any(positions[:, 2] <= self.radius):
            raise ValueError(f"every cavity centre must lie deeper than the cavity radius {self.radius} m.")
        return CAVITY_WALL_COEFFICIENT * GRAVITATIONAL_CONSTANT * density[None, :, None] * displacement


@dataclass(frozen=True)
class FamilyJointModel:
    """A field family observed by the ET test masses and a witness array.

    Attributes:
        family: The latent seismic field.
        layout: Test masses, suspensions and the triangle geometry.
        array: The witness array.
        cavity: The cavity around every test mass, or ``None`` for none. No
            default: whether the test masses sit in cavities is site geometry the
            caller must state.
    """

    family: FieldFamily
    layout: TestMassLayout
    array: WitnessArray
    cavity: CavityModel | None


def newtonian_accelerations(model: FamilyJointModel, frequency: float) -> NDArray[np.complex128]:
    """Return the ``(n_modes, n_unique_positions, 3)`` total Newtonian acceleration of the test masses."""
    positions = as_positions(model.layout.unique_positions())
    acceleration = model.family.acceleration(positions, frequency)
    if model.cavity is not None:
        displacement = model.family.displacement(positions, frequency)
        acceleration = acceleration + model.cavity.acceleration(
            displacement, np.asarray(model.family.density_at(positions), dtype=float), positions
        )
    return acceleration


def family_strain_responses(model: FamilyJointModel, frequency: float) -> NDArray[np.complex128]:
    """Return the ``(n_modes, 3)`` complex strain of every mode per unit mode amplitude."""
    frequency = require_positive_frequency(frequency)
    positions = model.layout.unique_positions()
    accelerations = newtonian_accelerations(model, frequency)
    omega = 2.0 * np.pi * frequency
    strains = np.empty((accelerations.shape[0], model.layout.geometry.n_detectors), dtype=complex)
    for mode, per_position in enumerate(accelerations):
        by_mass = model.layout.broadcast_by_position(
            {position: per_position[index] for index, position in enumerate(positions)}
        )
        strains[mode] = model.layout.strain_from_accelerations(by_mass, omega)
    return strains


def family_witness_responses(model: FamilyJointModel, frequency: float) -> NDArray[np.complex128]:
    """Return the ``(n_modes, n_active)`` response of every active witness per unit mode amplitude."""
    frequency = require_positive_frequency(frequency)
    sensors = model.array.active_sensors
    if not sensors:
        return np.zeros((model.family.n_modes, 0), dtype=complex)
    positions = as_positions([sensor.position for sensor in sensors])
    displacement = model.family.displacement(positions, frequency)
    orientations = np.stack([sensor.unit_orientation for sensor in sensors])
    transfers = np.array([sensor.transfer(frequency) for sensor in sensors])
    return np.einsum("mpc,pc->mp", displacement, orientations) * transfers[None, :]


def family_joint_mode_responses(
    frequencies: NDArray[np.floating], model: FamilyJointModel
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Return the per-mode joint responses and mode powers on a frequency grid.

    Channel order is the detector names followed by the active witness names.
    Non-positive frequencies and frequencies where every mode has zero power get
    all-zero rows.

    Args:
        frequencies: Frequency grid in hertz, shape ``(n_freq,)``.
        model: The joint model.

    Returns:
        ``(responses, power)``: ``(n_freq, n_modes, 3 + n_active)`` complex
        responses and ``(n_freq, n_modes)`` non-negative mode powers.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    n_modes = model.family.n_modes
    n_detector = model.layout.geometry.n_detectors
    n_channel = n_detector + len(model.array.active_sensors)
    responses = np.zeros((frequencies.size, n_modes, n_channel), dtype=complex)
    power = np.zeros((frequencies.size, n_modes), dtype=float)
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0:
            continue
        mode_power = np.asarray(model.family.mode_power(float(frequency)), dtype=float)
        if mode_power.shape != (n_modes,) or np.any(~np.isfinite(mode_power)) or np.any(mode_power < 0.0):
            raise ValueError(f"mode_power must be {n_modes} finite non-negative values at f={frequency}.")
        if not np.any(mode_power > 0.0):
            continue
        power[index] = mode_power
        responses[index, :, :n_detector] = family_strain_responses(model, float(frequency))
        responses[index, :, n_detector:] = family_witness_responses(model, float(frequency))
    return responses, power


def family_joint_covariance(
    frequencies: NDArray[np.floating],
    model: FamilyJointModel,
    manifest_frequency: float = 1.0,
    *,
    responses: tuple[NDArray[np.complex128], NDArray[np.float64]] | None = None,
) -> JointStrainWitnessCovariance:
    """Return the one-sided joint strain--witness covariance of a family, including array noise.

    Args:
        frequencies: Frequency grid in hertz.
        model: The joint model.
        manifest_frequency: Frequency at which the array manifest is evaluated.
        responses: Precomputed :func:`family_joint_mode_responses` on ``frequencies``,
            or ``None`` to compute them.

    Returns:
        The :class:`~gwmock_newtonian_noise.witness.response.JointStrainWitnessCovariance`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    geometry = model.layout.geometry
    detector_names = geometry.names
    witness_names = model.array.channel_names
    if set(detector_names) & set(witness_names):
        raise ValueError(
            f"detector and witness channel names must be disjoint, got {detector_names} and {witness_names}."
        )
    mode_responses, power = family_joint_mode_responses(frequencies, model) if responses is None else responses
    n_detector = geometry.n_detectors
    expected = (frequencies.size, model.family.n_modes, n_detector + len(witness_names))
    if mode_responses.shape != expected or power.shape != expected[:2]:
        raise ValueError(f"responses must have shape {expected} and power {expected[:2]}.")
    matrix = np.einsum("fmi,fmj,fm->fij", mode_responses, mode_responses.conj(), power)
    for index, frequency in enumerate(frequencies):
        if frequency > 0.0 and witness_names:
            matrix[index, n_detector:, n_detector:] += model.array.noise_covariance(float(frequency))
    return JointStrainWitnessCovariance(
        frequencies=frequencies,
        matrix=matrix,
        channel_order=(*detector_names, *witness_names),
        detector_channel_order=detector_names,
        witness_channel_order=witness_names,
        geometry=geometry,
        array_manifest=model.array.manifest(reference_frequency=manifest_frequency),
    )


def synthesize_family_joint(  # noqa: PLR0913, PLR0917 -- one parameter per physically distinct input.
    responses: NDArray[np.complex128],
    power: NDArray[np.float64],
    model: FamilyJointModel,
    duration: float,
    sampling_frequency: float,
    seed: int | np.random.SeedSequence | None,
) -> dict[str, NDArray[np.float64]]:
    """Synthesize a seeded joint realization of a family from precomputed responses.

    Args:
        responses: ``(n_freq, n_modes, n_channel)`` from :func:`family_joint_mode_responses`
            on the ``numpy.fft.rfftfreq`` grid of ``duration`` and ``sampling_frequency``.
        power: The matching ``(n_freq, n_modes)`` mode powers.
        model: The joint model.
        duration: Realization duration in seconds, strictly positive.
        sampling_frequency: Sampling frequency in hertz, strictly positive.
        seed: Integer seed or :class:`numpy.random.SeedSequence`, or ``None``.

    Returns:
        A dict mapping every channel name to a real ``(n_samples,)`` time series.
    """
    if duration <= 0.0 or sampling_frequency <= 0.0:
        raise ValueError(f"duration and sampling_frequency must be positive, got {duration}, {sampling_frequency}.")
    n_samples = round(duration * sampling_frequency)
    if n_samples < _MIN_REALIZATION_SAMPLES:
        raise ValueError("duration and sampling_frequency must produce at least two samples.")
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    if responses.shape[0] != frequencies.size or power.shape != responses.shape[:2]:
        raise ValueError(
            f"responses/power must cover the {frequencies.size} realization bins, got {responses.shape}, {power.shape}."
        )
    geometry = model.layout.geometry
    n_detector = geometry.n_detectors
    witness_names = model.array.channel_names
    channel_order = (*geometry.names, *witness_names)
    rng = create_generator(seed)
    resolution = sampling_frequency / n_samples
    times = np.arange(n_samples) / sampling_frequency
    series = np.zeros((len(channel_order), n_samples), dtype=float)
    for index, frequency in enumerate(frequencies):
        if frequency <= 0.0:
            continue
        phase = np.exp(-1j * 2.0 * np.pi * frequency * times)
        if np.any(power[index] > 0.0):
            theta = rng.uniform(0.0, 2.0 * np.pi, size=power.shape[1])
            amplitude = np.sqrt(2.0 * power[index] * resolution) * np.exp(1j * theta)
            series += np.real(np.outer(amplitude @ responses[index], phase))
        if not witness_names:
            continue
        noise_covariance = model.array.noise_covariance(float(frequency))
        if not np.any(noise_covariance):
            continue
        white = (rng.standard_normal(len(witness_names)) + 1j * rng.standard_normal(len(witness_names))) / np.sqrt(2.0)
        series[n_detector:] += np.real(
            np.outer(np.sqrt(2.0 * resolution) * (_noise_factor(noise_covariance) @ white), phase)
        )
    return {name: series[index] for index, name in enumerate(channel_order)}


__all__ = [
    "CAVITY_WALL_COEFFICIENT",
    "CavityModel",
    "FamilyJointModel",
    "family_joint_covariance",
    "family_joint_mode_responses",
    "family_strain_responses",
    "family_witness_responses",
    "newtonian_accelerations",
    "synthesize_family_joint",
]
