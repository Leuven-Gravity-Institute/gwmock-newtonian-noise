"""Controlled witness-sensor arrays, noise models and manifests.

This module groups :class:`~gwmock_newtonian.witness.sensor.WitnessSensor`
objects into arrays, attaches the *array-level* noise covariance (self-noise is
a per-sensor property; its correlation across sensors is not), handles missing
channels, and produces an immutable manifest recording geometry and
calibration.

Nested benchmark arrays
-----------------------

:func:`nested_disk_array` builds arrays from a **fixed** set of ``max_sensors``
master sites (a sunflower/Vogel spiral filling a disk of radius ``aperture``)
taken in a fixed *site order*; the family member for ``N`` is the first ``N``
sites of that order. The caller must name the order, as a
:class:`NestedArrayMode`:

- :attr:`NestedArrayMode.FIXED_APERTURE` orders the sites by farthest-point
  sampling seeded at the outermost site. The outermost site is present at every
  ``N``, so the realized radius is the same for every ``N``: the master
  sequence's outer radius ``aperture * sqrt(1 - 1 / (2 * max_sensors))``, which
  falls short of the ceiling by a relative ``1 / (2 * max_sensors)`` at most.
  Changing ``N`` changes the sensor density inside that fixed span, not the
  span. This is the mode an ``N`` scan should use.
- :attr:`NestedArrayMode.GROWING_SPAN` takes the master sequence in its own
  radially increasing order, so the realized radius grows with ``N`` (for
  ``aperture=800`` and ``max_sensors=12`` it is about 163 m at ``N=1``, 432 m at
  ``N=4`` and 783 m at ``N=12``). An ``N`` scan in this mode confounds sensor
  density with baseline; it is kept for studies that want exactly that.

In both modes the nesting rule is exact and testable:

- positions are nested, ``positions(N) == positions(N_max)[:N]``, and every
  position is one of the fixed master sites;
- the aperture ceiling ``aperture`` is a fixed parameter of the family, not a
  function of ``N``;
- every per-sensor property -- measured quantity, sensing axis rule, response,
  calibration, self-noise PSD, timing offset -- is fixed and identical for every
  ``N``.

Changing ``N`` therefore changes only which of the fixed sites are active and
the aggregate array quality; the latter is made explicit, not silent, by the
manifest's raw count, effective rank and participation ratio. The manifest also
records the mode and the *realized* horizontal radius and span next to the
``aperture`` ceiling, so a change of span with ``N`` is never hidden behind a
constant ceiling.

Effective rank versus raw count
-------------------------------

:func:`effective_rank` and :func:`participation_ratio` are computed from the
array's ``N x N`` one-sided noise covariance. A diagonal (independent) noise
model gives rank ``N``; a noise model whose channels are strongly correlated at
a length scale much larger than the array aperture is numerically rank one
despite ``N`` raw channels. The manifest reports the raw count and the effective
rank *separately* so the two are never conflated.

All numeric fixtures reachable from this module's tests are UNANCHORED
illustrative values, not a representative ET sensor deployment.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.witness.sensor import (
    ComplexResponse,
    PSDCallable,
    SensorQuantity,
    WitnessSensor,
    unit_response,
)

#: Golden angle used by the sunflower aperture-filling sequence, in radians.
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

#: Number of horizontal coordinates in an array centre ``(x, y)``.
_HORIZONTAL_DIMENSIONS = 2

#: Default relative eigenvalue threshold for :func:`effective_rank`.
_DEFAULT_RANK_RTOL = 1e-9


class NestedArrayMode(StrEnum):
    """The site order a nested disk array takes its first ``N`` sites from.

    See the module docstring for the contract each mode promises.
    """

    #: Farthest-point order seeded at the outermost site: fixed realized span.
    FIXED_APERTURE = "fixed_aperture"
    #: The master sequence's radially increasing order: span grows with ``N``.
    GROWING_SPAN = "growing_span"


def _horizontal_radius(positions: Sequence[tuple[float, float, float]], center: tuple[float, float]) -> float:
    """Return the largest horizontal distance of ``positions`` from ``center``; ``0.0`` when empty."""
    if not positions:
        return 0.0
    horizontal = np.array(positions, dtype=float)[:, :2] - np.asarray(center, dtype=float)
    return float(np.max(np.hypot(horizontal[:, 0], horizontal[:, 1])))


def _horizontal_span(positions: Sequence[tuple[float, float, float]]) -> float:
    """Return the largest horizontal distance between two of ``positions``; ``0.0`` for fewer than two."""
    if not positions:
        return 0.0
    horizontal = np.array(positions, dtype=float)[:, :2]
    difference = horizontal[:, None, :] - horizontal[None, :, :]
    return float(np.max(np.hypot(difference[..., 0], difference[..., 1])))


def _hermitian_eigenvalues(matrix: NDArray[np.complexfloating]) -> NDArray[np.float64]:
    """Return the real eigenvalues of the Hermitian part of ``matrix``."""
    array = np.asarray(matrix, dtype=complex)
    hermitian = 0.5 * (array + array.conj().T)
    return np.linalg.eigvalsh(hermitian)


def effective_rank(matrix: NDArray[np.complexfloating], rtol: float = _DEFAULT_RANK_RTOL) -> int:
    """Return the number of eigenvalues of ``matrix`` above ``rtol`` times the largest.

    Args:
        matrix: A square Hermitian matrix, typically a one-sided noise
            covariance.
        rtol: Relative threshold; an eigenvalue counts when it exceeds
            ``rtol * max_eigenvalue``. Must be non-negative.

    Returns:
        The effective rank, an integer in ``[0, n]``. A zero matrix has rank zero.
    """
    if rtol < 0.0:
        raise ValueError(f"rtol must be non-negative, got {rtol}.")
    eigenvalues = _hermitian_eigenvalues(matrix)
    if eigenvalues.size == 0:
        return 0
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        return 0
    return int(np.count_nonzero(eigenvalues > rtol * largest))


def participation_ratio(matrix: NDArray[np.complexfloating]) -> float:
    """Return the participation ratio ``(sum L)^2 / sum L^2`` of the eigenvalues.

    This is a continuous rank measure: ``n`` for ``n`` equal eigenvalues and
    ``1`` for a rank-one matrix. It complements the integer
    :func:`effective_rank`.

    Args:
        matrix: A square Hermitian positive-semidefinite matrix.

    Returns:
        The participation ratio, in ``[0, n]``; ``0`` for a zero matrix.
    """
    eigenvalues = np.clip(_hermitian_eigenvalues(matrix), 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0
    return float(total**2 / np.sum(eigenvalues**2))


class NoiseModel(ABC):
    """One-sided noise covariance across an array's active sensor channels.

    A noise model may either rebuild its covariance from the sensors it is
    handed (as :class:`DiagonalNoise` and :class:`CorrelatedNoise` do), or be
    defined against a fixed declared channel order and select the active
    sensors out of it (as :class:`ExplicitNoise` does). The latter advertises
    that order through :attr:`declared_channel_order`, which
    :class:`WitnessArray` validates against its declared sensors.
    """

    @property
    def declared_channel_order(self) -> tuple[str, ...] | None:
        """Return the fixed channel order this model is defined against, or ``None``.

        ``None`` means the model derives its covariance entirely from the
        sensors it is given; a non-``None`` order means sensors are selected by
        name out of that fixed order.
        """
        return None

    @abstractmethod
    def covariance(self, frequency: float, sensors: Sequence[WitnessSensor]) -> NDArray[np.complex128]:
        """Return the ``(n, n)`` one-sided noise covariance at one frequency.

        Args:
            frequency: Frequency in hertz, strictly positive.
            sensors: The active sensors, in channel order.

        Returns:
            A Hermitian positive-semidefinite ``(n, n)`` covariance.
        """


@dataclass(frozen=True)
class DiagonalNoise(NoiseModel):
    """Independent per-sensor noise: the self-noise PSDs on the diagonal."""

    def covariance(self, frequency: float, sensors: Sequence[WitnessSensor]) -> NDArray[np.complex128]:
        """Return ``diag(self_noise_psd_i(f))`` for the active sensors."""
        diagonal = np.array([sensor.self_noise(frequency) for sensor in sensors], dtype=float)
        return np.diag(diagonal).astype(complex)


@dataclass(frozen=True)
class CorrelatedNoise(NoiseModel):
    """Self-noise correlated across sensors by a Gaussian spatial kernel.

    The covariance is ``diag(sqrt(S)) @ rho @ diag(sqrt(S))`` with
    ``rho_ij = exp(-d_ij**2 / (2 * correlation_length**2))`` and ``d_ij`` the
    distance between sensors ``i`` and ``j``. A Gaussian kernel matrix is
    positive semidefinite for any point set, so the result is a valid noise
    covariance by construction. At ``correlation_length`` far larger than the
    aperture the matrix is numerically rank one; far smaller than the sensor
    spacing it reduces to :class:`DiagonalNoise`.

    Attributes:
        correlation_length: Spatial correlation length in metres, strictly
            positive.
    """

    correlation_length: float

    def __post_init__(self) -> None:
        """Reject a non-positive correlation length."""
        if not (self.correlation_length > 0.0):
            raise ValueError(f"correlation_length must be positive, got {self.correlation_length}.")

    def covariance(self, frequency: float, sensors: Sequence[WitnessSensor]) -> NDArray[np.complex128]:
        """Return the correlated self-noise covariance at one frequency."""
        diagonal = np.array([sensor.self_noise(frequency) for sensor in sensors], dtype=float)
        positions = np.array([sensor.position for sensor in sensors], dtype=float)
        if positions.size == 0:
            return np.zeros((0, 0), dtype=complex)
        squared = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
        correlation = np.exp(-squared / (2.0 * self.correlation_length**2))
        sqrt_diagonal = np.sqrt(diagonal)
        return (sqrt_diagonal[:, None] * correlation * sqrt_diagonal[None, :]).astype(complex)


@dataclass(frozen=True)
class ExplicitNoise(NoiseModel):
    """A caller-supplied noise covariance defined against a fixed channel order.

    The callable returns the full ``(M, M)`` covariance over the declared array
    in ``channel_order``. When an array is permuted or has missing channels, the
    model selects the active sensors **by name** out of that fixed order and
    returns the corresponding principal submatrix, so the noise follows sensor
    identity under both permutations and masking. The declared order must be a
    permutation of the array's sensors; :class:`WitnessArray` enforces this.

    Attributes:
        channel_order: The sensor names the callable's matrix is ordered by,
            forming the declared array order.
        covariance_callable: Callable mapping a frequency in hertz to the
            ``(M, M)`` one-sided noise covariance in ``channel_order``. It is
            checked for the right shape and for Hermiticity, but the caller owns
            positive-semidefiniteness.
    """

    channel_order: tuple[str, ...]
    covariance_callable: Callable[[float], NDArray[np.complexfloating]]

    def __post_init__(self) -> None:
        """Reject a non-callable noise source or a repeated channel name."""
        if not callable(self.covariance_callable):
            raise TypeError("ExplicitNoise.covariance_callable must be callable.")
        if not self.channel_order:
            raise ValueError("ExplicitNoise.channel_order must be non-empty.")
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError(f"ExplicitNoise.channel_order must be unique, got {self.channel_order}.")

    @property
    def declared_channel_order(self) -> tuple[str, ...] | None:
        """Return the fixed declared channel order the covariance is defined against."""
        return self.channel_order

    def covariance(self, frequency: float, sensors: Sequence[WitnessSensor]) -> NDArray[np.complex128]:
        """Return the declared covariance's active principal submatrix, in sensor order.

        Args:
            frequency: Frequency in hertz, strictly positive.
            sensors: The active sensors, in channel order; they must all appear
                in :attr:`channel_order`.

        Returns:
            The ``(len(sensors), len(sensors))`` principal submatrix in the
            order of ``sensors``.
        """
        matrix = np.asarray(self.covariance_callable(frequency), dtype=complex)
        expected = (len(self.channel_order), len(self.channel_order))
        if matrix.shape != expected:
            raise ValueError(
                f"explicit noise covariance must have shape {expected} for channel_order "
                f"{self.channel_order}, got {matrix.shape}."
            )
        if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
            raise ValueError("explicit noise covariance must be Hermitian.")
        index = {name: position for position, name in enumerate(self.channel_order)}
        missing = [sensor.name for sensor in sensors if sensor.name not in index]
        if missing:
            raise ValueError(f"explicit noise channel_order {self.channel_order} does not include sensors {missing}.")
        selection = [index[sensor.name] for sensor in sensors]
        return matrix[np.ix_(selection, selection)]


@dataclass(frozen=True)
class SensorManifest:
    """A per-sensor record of geometry, measured quantity and calibration."""

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    quantity: SensorQuantity
    unit: str
    calibration: complex
    timing_offset: float
    response_name: str
    response_at_reference: complex
    self_noise_psd_at_reference: float
    present: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary; complex values become ``[re, im]``."""
        return {
            "name": self.name,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "quantity": str(self.quantity),
            "unit": self.unit,
            "calibration": [self.calibration.real, self.calibration.imag],
            "timing_offset": self.timing_offset,
            "response_name": self.response_name,
            "response_at_reference": [self.response_at_reference.real, self.response_at_reference.imag],
            "self_noise_psd_at_reference": self.self_noise_psd_at_reference,
            "present": self.present,
        }


@dataclass(frozen=True)
class ArrayManifest:
    """Immutable record of an array's geometry, calibration and rank summary.

    Attributes:
        aperture: Fixed aperture radius ceiling in metres, or ``None`` for an
            explicit array.
        nesting_rule: Human-readable statement of the nesting rule.
        mode: The nested-array mode (a :class:`NestedArrayMode` value), or
            ``"explicit"`` for an explicit array.
        center: Horizontal centre ``(x, y)`` in metres that ``realized_radius``
            is measured from.
        realized_radius: Largest horizontal distance in metres of a declared
            sensor from ``center``; the realized counterpart of ``aperture``.
        realized_span: Largest horizontal distance in metres between two
            declared sensors; ``0.0`` for fewer than two.
        active_realized_radius: As ``realized_radius``, over the present sensors
            only.
        active_realized_span: As ``realized_span``, over the present sensors
            only.
        reference_frequency: Frequency in hertz at which per-sensor response and
            self-noise, and the rank summary, are evaluated.
        raw_sensor_count: Number of sensors declared, including missing ones.
        active_sensor_count: Number of present (non-missing) sensors.
        raw_channel_count: Number of active sensor channels.
        effective_rank: Effective rank of the active noise covariance.
        participation_ratio: Participation ratio of the active noise covariance.
        rank_rtol: Relative eigenvalue threshold used for ``effective_rank``.
        sensors: Per-sensor records, in array order.
    """

    aperture: float | None
    nesting_rule: str
    mode: str
    center: tuple[float, float]
    realized_radius: float
    realized_span: float
    active_realized_radius: float
    active_realized_span: float
    reference_frequency: float
    raw_sensor_count: int
    active_sensor_count: int
    raw_channel_count: int
    effective_rank: int
    participation_ratio: float
    rank_rtol: float
    sensors: tuple[SensorManifest, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the whole manifest."""
        return {
            "aperture": self.aperture,
            "nesting_rule": self.nesting_rule,
            "mode": self.mode,
            "center": list(self.center),
            "realized_radius": self.realized_radius,
            "realized_span": self.realized_span,
            "active_realized_radius": self.active_realized_radius,
            "active_realized_span": self.active_realized_span,
            "reference_frequency": self.reference_frequency,
            "raw_sensor_count": self.raw_sensor_count,
            "active_sensor_count": self.active_sensor_count,
            "raw_channel_count": self.raw_channel_count,
            "effective_rank": self.effective_rank,
            "participation_ratio": self.participation_ratio,
            "rank_rtol": self.rank_rtol,
            "sensors": [sensor.to_dict() for sensor in self.sensors],
        }


@dataclass(frozen=True)
class WitnessArray:
    """A set of witness sensors, an array noise model, and a missing-channel mask.

    Attributes:
        sensors: All declared sensors, in channel order.
        noise: Array-level noise model; defaults to independent self-noise.
        missing: Names of sensors treated as missing (dropped from every
            covariance, realization and channel list).
        aperture: Fixed aperture radius ceiling in metres, or ``None`` for an
            explicit array.
        nesting_rule: Human-readable statement of the nesting rule.
        mode: The nested-array mode (a :class:`NestedArrayMode` value), or
            ``"explicit"`` for an explicit array.
        center: Horizontal centre ``(x, y)`` in metres that the realized radius
            is measured from.
    """

    sensors: tuple[WitnessSensor, ...]
    noise: NoiseModel = field(default_factory=DiagonalNoise)
    missing: tuple[str, ...] = ()
    aperture: float | None = None
    nesting_rule: str = "explicit"
    mode: str = "explicit"
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate unique sensor names, the missing mask and the aperture."""
        names = [sensor.name for sensor in self.sensors]
        if len(set(names)) != len(names):
            raise ValueError(f"sensor names must be unique, got {names}.")
        unknown = set(self.missing) - set(names)
        if unknown:
            raise ValueError(f"missing sensor names are not in the array: {sorted(unknown)}.")
        if len(set(self.missing)) != len(self.missing):
            raise ValueError(f"missing sensor names must be unique, got {self.missing}.")
        if self.aperture is not None and not (self.aperture > 0.0):
            raise ValueError(f"aperture must be positive, got {self.aperture}.")
        if len(self.center) != _HORIZONTAL_DIMENSIONS or not all(math.isfinite(value) for value in self.center):
            raise ValueError(f"center must be a finite (x, y) pair, got {self.center}.")
        if not isinstance(self.noise, NoiseModel):
            raise TypeError(f"noise must be a NoiseModel, got {type(self.noise)!r}.")
        declared = self.noise.declared_channel_order
        if declared is not None and set(declared) != set(names):
            raise ValueError(
                f"explicit noise channel_order {declared} must be a permutation of the declared array sensors {names}."
            )

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the active channel names, in array order."""
        return tuple(sensor.name for sensor in self.active_sensors)

    @property
    def raw_sensor_count(self) -> int:
        """Return the number of declared sensors, including missing ones."""
        return len(self.sensors)

    @property
    def active_sensors(self) -> tuple[WitnessSensor, ...]:
        """Return the present (non-missing) sensors, in array order."""
        if not self.missing:
            return self.sensors
        missing = set(self.missing)
        return tuple(sensor for sensor in self.sensors if sensor.name not in missing)

    def positions(self) -> tuple[tuple[float, float, float], ...]:
        """Return every declared sensor position, in array order."""
        return tuple(sensor.position for sensor in self.sensors)

    def realized_radius(self, *, active_only: bool = False) -> float:
        """Return the largest horizontal distance in metres of a sensor from :attr:`center`.

        Args:
            active_only: If ``True`` use the present sensors only; otherwise
                every declared sensor.

        Returns:
            The realized radius; ``0.0`` for an empty sensor set.
        """
        sensors = self.active_sensors if active_only else self.sensors
        return _horizontal_radius([sensor.position for sensor in sensors], self.center)

    def realized_span(self, *, active_only: bool = False) -> float:
        """Return the largest horizontal distance in metres between two sensors.

        Args:
            active_only: If ``True`` use the present sensors only; otherwise
                every declared sensor.

        Returns:
            The realized span; ``0.0`` for fewer than two sensors.
        """
        sensors = self.active_sensors if active_only else self.sensors
        return _horizontal_span([sensor.position for sensor in sensors])

    def noise_covariance(self, frequency: float, *, active_only: bool = True) -> NDArray[np.complex128]:
        """Return the array noise covariance at one frequency.

        Args:
            frequency: Frequency in hertz, strictly positive.
            active_only: If ``True`` use the present sensors only; otherwise
                include missing sensors as well.

        Returns:
            A Hermitian positive-semidefinite ``(n, n)`` one-sided covariance.
        """
        sensors = self.active_sensors if active_only else self.sensors
        return self.noise.covariance(frequency, sensors)

    def with_missing(self, missing: tuple[str, ...]) -> WitnessArray:
        """Return a copy with the given missing-channel mask."""
        return WitnessArray(
            sensors=self.sensors,
            noise=self.noise,
            missing=tuple(missing),
            aperture=self.aperture,
            nesting_rule=self.nesting_rule,
            mode=self.mode,
            center=self.center,
        )

    def permuted(self, order: Sequence[int]) -> WitnessArray:
        """Return a copy whose sensors are reordered by ``order``.

        Args:
            order: A permutation of ``range(len(sensors))``.

        Returns:
            The reordered array; the missing mask (by name) is unchanged.
        """
        order_array = np.asarray(order)
        if sorted(order_array.tolist()) != list(range(len(self.sensors))):
            raise ValueError(f"order must be a permutation of range({len(self.sensors)}), got {order}.")
        sensors = tuple(self.sensors[index] for index in order_array.tolist())
        return WitnessArray(
            sensors=sensors,
            noise=self.noise,
            missing=self.missing,
            aperture=self.aperture,
            nesting_rule=self.nesting_rule,
            mode=self.mode,
            center=self.center,
        )

    def manifest(self, reference_frequency: float = 1.0, rank_rtol: float = _DEFAULT_RANK_RTOL) -> ArrayManifest:
        """Return the array manifest at one reference frequency.

        Args:
            reference_frequency: Frequency in hertz at which per-sensor response,
                self-noise and the rank summary are evaluated.
            rank_rtol: Relative eigenvalue threshold for :func:`effective_rank`.

        Returns:
            The :class:`ArrayManifest`.
        """
        if reference_frequency <= 0.0:
            raise ValueError(f"reference_frequency must be positive, got {reference_frequency}.")
        active_names = set(self.channel_names)
        records = tuple(
            SensorManifest(
                name=sensor.name,
                position=sensor.position,
                orientation=tuple(float(value) for value in sensor.unit_orientation),
                quantity=sensor.quantity,
                unit=sensor.unit,
                calibration=complex(sensor.calibration),
                timing_offset=sensor.timing_offset,
                response_name=getattr(sensor.response, "__name__", repr(sensor.response)),
                response_at_reference=complex(sensor.response(reference_frequency)),
                self_noise_psd_at_reference=sensor.self_noise(reference_frequency),
                present=sensor.name in active_names,
            )
            for sensor in self.sensors
        )
        noise = self.noise_covariance(reference_frequency)
        return ArrayManifest(
            aperture=self.aperture,
            nesting_rule=self.nesting_rule,
            mode=self.mode,
            center=self.center,
            realized_radius=self.realized_radius(),
            realized_span=self.realized_span(),
            active_realized_radius=self.realized_radius(active_only=True),
            active_realized_span=self.realized_span(active_only=True),
            reference_frequency=reference_frequency,
            raw_sensor_count=self.raw_sensor_count,
            active_sensor_count=len(self.active_sensors),
            raw_channel_count=len(self.active_sensors),
            effective_rank=effective_rank(noise, rtol=rank_rtol),
            participation_ratio=participation_ratio(noise),
            rank_rtol=rank_rtol,
            sensors=records,
        )


def sunflower_positions(
    max_sensors: int,
    aperture: float,
    center: tuple[float, float] = (0.0, 0.0),
    depth: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return the fixed sunflower (Vogel) sequence filling a disk of radius ``aperture``.

    The ``i``-th of ``max_sensors`` points is at radius
    ``aperture * sqrt((i + 0.5) / max_sensors)`` and azimuth ``i * golden_angle``.
    Every point lies strictly inside ``aperture``, and the sequence is fixed
    once ``max_sensors`` and ``aperture`` are chosen, so any prefix is nested.

    Args:
        max_sensors: Length of the master sequence, at least one.
        aperture: Aperture radius in metres, strictly positive.
        center: Horizontal centre ``(x, y)`` in metres.
        depth: Common ``z`` depth in metres (positive downward).

    Returns:
        The master sequence as a tuple of ``(x, y, z)`` positions.
    """
    if max_sensors < 1:
        raise ValueError(f"max_sensors must be at least 1, got {max_sensors}.")
    if not (aperture > 0.0):
        raise ValueError(f"aperture must be positive, got {aperture}.")
    index = np.arange(max_sensors, dtype=float)
    radius = aperture * np.sqrt((index + 0.5) / max_sensors)
    angle = index * _GOLDEN_ANGLE
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return tuple((float(xi), float(yi), float(depth)) for xi, yi in zip(x, y, strict=True))


def nested_site_order(
    max_sensors: int,
    aperture: float,
    mode: NestedArrayMode | str,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, ...]:
    """Return the order in which a nested disk array activates the master sites.

    The ``N``-sensor member of the family occupies the master sites
    ``order[:N]``, where the master sites are
    ``sunflower_positions(max_sensors, aperture, center)``.

    Args:
        max_sensors: Number of master sites, at least one.
        aperture: Aperture radius ceiling in metres, strictly positive.
        mode: :attr:`NestedArrayMode.FIXED_APERTURE` for the farthest-point
            order seeded at the outermost site (ties broken by the lowest site
            index), or :attr:`NestedArrayMode.GROWING_SPAN` for the identity
            order.
        center: Horizontal centre ``(x, y)`` in metres.

    Returns:
        A permutation of ``range(max_sensors)``.
    """
    mode = NestedArrayMode(mode)
    positions = sunflower_positions(max_sensors, aperture, center=center)
    if mode is NestedArrayMode.GROWING_SPAN:
        return tuple(range(max_sensors))
    horizontal = np.array(positions, dtype=float)[:, :2]
    offset = horizontal - np.asarray(center, dtype=float)
    first = int(np.argmax(np.hypot(offset[:, 0], offset[:, 1])))
    order = [first]
    distance = np.hypot(*(horizontal - horizontal[first]).T)
    distance[first] = -np.inf
    while len(order) < max_sensors:
        chosen = int(np.argmax(distance))
        order.append(chosen)
        distance = np.minimum(distance, np.hypot(*(horizontal - horizontal[chosen]).T))
        distance[order] = -np.inf
    return tuple(order)


#: How each mode's site order is described in the nesting rule.
_SITE_ORDER_DESCRIPTION = {
    NestedArrayMode.FIXED_APERTURE: "farthest-point order from the outermost site (realized span fixed across N)",
    NestedArrayMode.GROWING_SPAN: "radially increasing sequence order (realized span grows with N)",
}


def _radial_orientation(
    position: tuple[float, float, float], center: tuple[float, float]
) -> tuple[float, float, float]:
    """Return the horizontal unit vector from ``center`` to ``position``."""
    dx = position[0] - center[0]
    dy = position[1] - center[1]
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (1.0, 0.0, 0.0)
    return (dx / norm, dy / norm, 0.0)


def nested_disk_array(  # noqa: PLR0913 -- one parameter per physically distinct input.
    n_sensors: int,
    *,
    aperture: float,
    max_sensors: int,
    mode: NestedArrayMode | str,
    depth: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    quantity: SensorQuantity = SensorQuantity.DISPLACEMENT,
    response: ComplexResponse | None = None,
    calibration: complex = 1.0 + 0.0j,
    self_noise_psd: PSDCallable | None = None,
    timing_offset: float = 0.0,
    output_unit: str | None = None,
    orientation: str | tuple[float, float, float] = "radial",
    noise: NoiseModel | None = None,
    name_prefix: str = "W",
) -> WitnessArray:
    """Build a nested disk array: the first ``n_sensors`` master sites in the mode's site order.

    Every sensor shares the same per-sensor quality (quantity, response,
    calibration, self-noise PSD, timing offset, and either a fixed axis or the
    radial axis rule), so ``n_sensors`` changes only how many of the fixed
    sites are active -- never the aperture ceiling or any per-sensor property.
    Whether it changes the realized span depends on ``mode``, which has no
    default so that an ``N`` scan never gets a growing span silently.

    Args:
        n_sensors: Number of active sensors, ``1 <= n_sensors <= max_sensors``.
        aperture: Fixed aperture radius ceiling in metres, strictly positive.
        max_sensors: Number of fixed master sites, at least ``n_sensors``.
        mode: The site order, see :func:`nested_site_order`.
            :attr:`NestedArrayMode.FIXED_APERTURE` keeps the realized radius
            fixed across ``N``; :attr:`NestedArrayMode.GROWING_SPAN` lets it
            grow with ``N``.
        depth: Common ``z`` depth in metres (positive downward).
        center: Horizontal centre ``(x, y)`` in metres.
        quantity: Measured kinematic quantity, shared by every sensor.
        response: Complex instrument response, shared by every sensor; ``None``
            uses the identity.
        calibration: Complex calibration, shared by every sensor.
        self_noise_psd: Self-noise PSD, shared by every sensor; ``None`` is noiseless.
        timing_offset: Timing offset in seconds, shared by every sensor.
        output_unit: Output unit, shared by every sensor; ``None`` uses the SI
            unit of ``quantity``.
        orientation: ``"radial"`` (default) for the horizontal outward axis at
            each position, or an explicit unit axis shared by every sensor.
        noise: Array noise model; defaults to :class:`DiagonalNoise`.
        name_prefix: Identifier prefix; the ``i``-th site in the order is named
            ``f"{prefix}{i}"``, so a name denotes the same site for every ``N``.

    Returns:
        The :class:`WitnessArray`.
    """
    mode = NestedArrayMode(mode)
    if n_sensors < 1:
        raise ValueError(f"n_sensors must be at least 1, got {n_sensors}.")
    if n_sensors > max_sensors:
        raise ValueError(f"n_sensors must not exceed max_sensors, got {n_sensors} > {max_sensors}.")
    positions = sunflower_positions(max_sensors, aperture, center=center, depth=depth)
    order = nested_site_order(max_sensors, aperture, mode, center=center)
    sensors = []
    for index in range(n_sensors):
        position = positions[order[index]]
        if orientation == "radial":
            axis = _radial_orientation(position, center)
        elif isinstance(orientation, str):
            raise ValueError(f"unknown orientation rule {orientation!r}; use 'radial' or a 3-vector.")
        else:
            axis = tuple(float(value) for value in orientation)
        sensors.append(
            WitnessSensor(
                name=f"{name_prefix}{index}",
                position=position,
                orientation=axis,
                quantity=quantity,
                response=response if response is not None else unit_response,
                calibration=calibration,
                self_noise_psd=self_noise_psd,
                timing_offset=timing_offset,
                output_unit=output_unit,
            )
        )
    return WitnessArray(
        sensors=tuple(sensors),
        noise=noise if noise is not None else DiagonalNoise(),
        aperture=aperture,
        nesting_rule=(
            f"{mode.value}: first N sites, in {_SITE_ORDER_DESCRIPTION[mode]}, of the fixed "
            f"{max_sensors}-site sunflower sequence filling aperture {aperture} m; "
            "per-sensor quantity, axis rule, response, calibration, self-noise and timing fixed across N"
        ),
        mode=mode.value,
        center=(float(center[0]), float(center[1])),
    )


__all__ = [
    "ArrayManifest",
    "CorrelatedNoise",
    "DiagonalNoise",
    "ExplicitNoise",
    "NestedArrayMode",
    "NoiseModel",
    "SensorManifest",
    "WitnessArray",
    "effective_rank",
    "nested_disk_array",
    "nested_site_order",
    "participation_ratio",
    "sunflower_positions",
]
