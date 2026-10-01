"""Seismic witness sensors observing the same latent field as the detector response.

A witness sensor is a linear, complex, frequency-dependent functional of the
*latent* Rayleigh-wave field that the detector/test-mass response is also built
from (:mod:`gwmock_newtonian.seismic.field`,
:mod:`gwmock_newtonian.detector.response`). It samples the ground motion at a
position along a sensing axis and turns it into an output through an explicit
chain of factors. Nothing in this module re-derives the field: the per-mode
response is assembled from the field's own ``steering_vectors`` primitive.

The response chain, in order, is

    y(f) = calibration * response(f) * quantity_transfer(f) * timing(f) * (n . u(p, f)),

where

- ``n . u(p, f)`` is the field displacement at the sensor position ``p``
  projected on the unit sensing axis ``n`` (the field's own steering vectors);
- ``quantity_transfer`` turns a displacement into the measured kinematic
  quantity. The field module uses the time convention ``exp(-i * omega * t)``
  (see :func:`~gwmock_newtonian.seismic.field.realize`), so for
  ``omega = 2 * pi * f`` a displacement ``x`` gives a velocity
  ``-i * omega * x`` and an acceleration ``-omega**2 * x``;
- ``response`` is any caller-supplied complex transfer in output units per
  metre of displacement along the axis (a seismometer's electronics/instrument
  response, a sensor's non-flat passband, ...);
- ``calibration`` is a complex scalar amplitude and phase calibration;
- ``timing`` is ``exp(+i * omega * tau)`` for a documented timing offset ``tau``
  in seconds: the channel records ``x(t - tau)``, i.e. it is *late* by ``tau``,
  which multiplies the spectrum by ``exp(+i * omega * tau)``.

Self-noise is a one-sided PSD carried on the sensor; the *correlations* between
sensor noise channels live in
:mod:`gwmock_newtonian.witness.array`'s noise models, because a covariance
between sensors is an array-level object.

All numeric values used by callers in this package's tests are UNANCHORED
illustrative fixtures, not a representative ET sensor set. Positions use the
field convention (metres, ``z`` positive downward from the free surface);
sensor outputs are in the caller's output unit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

#: One-sided power spectral density callable: strictly-positive frequency in
#: hertz to a non-negative value in ``output_unit**2 / hertz``.
PSDCallable = Callable[[float], float]

#: Complex instrument response: frequency in hertz to a complex transfer in
#: output units per metre of field displacement along the sensing axis.
ComplexResponse = Callable[[float], complex]


class SensorQuantity(StrEnum):
    """The kinematic quantity a witness sensor measures.

    The transfer from field displacement to the reported quantity is stated in
    :func:`quantity_transfer`; it is fixed by the field's ``exp(-i * omega * t)``
    time convention.
    """

    DISPLACEMENT = "displacement"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


#: SI unit of each kinematic quantity, the output unit of a sensor whose
#: calibration and instrument response are dimensionless.
QUANTITY_SI_UNITS: dict[SensorQuantity, str] = {
    SensorQuantity.DISPLACEMENT: "m",
    SensorQuantity.VELOCITY: "m/s",
    SensorQuantity.ACCELERATION: "m/s^2",
}


def unit_response(frequency: float) -> complex:
    """Return the identity instrument response, ``1`` at every frequency."""
    del frequency
    return 1.0 + 0.0j


def constant_response(value: complex) -> ComplexResponse:
    """Return a frequency-independent complex instrument response.

    Args:
        value: The complex response in output units per metre.

    Returns:
        A callable returning ``value`` at every frequency.
    """

    def _response(frequency: float) -> complex:
        del frequency
        return complex(value)

    _response.__name__ = f"constant_response({value!r})"
    return _response


def quantity_transfer(quantity: SensorQuantity, frequency: float) -> complex:
    """Return the displacement-to-quantity transfer at one frequency.

    With the field's ``exp(-i * omega * t)`` convention and
    ``omega = 2 * pi * frequency``, differentiation multiplies a Fourier
    coefficient by ``-i * omega``:

    - ``DISPLACEMENT``: ``1``;
    - ``VELOCITY``: ``-i * omega``;
    - ``ACCELERATION``: ``-omega**2``.

    Args:
        quantity: The measured kinematic quantity.
        frequency: Frequency in hertz, strictly positive.

    Returns:
        The complex transfer in ``(quantity unit) / metre``.
    """
    if frequency <= 0.0:
        raise ValueError(f"frequency must be positive, got {frequency}.")
    omega = 2.0 * np.pi * frequency
    if quantity is SensorQuantity.DISPLACEMENT:
        return 1.0 + 0.0j
    if quantity is SensorQuantity.VELOCITY:
        return -1j * omega
    if quantity is SensorQuantity.ACCELERATION:
        return complex(-(omega**2))
    raise ValueError(f"unknown sensor quantity {quantity!r}.")


@dataclass(frozen=True)
class WitnessSensor:
    """One seismic witness sensor observing the latent Rayleigh-wave field.

    Attributes:
        name: Unique identifier within an array.
        position: ``(x, y, z)`` position in metres, ``z`` positive downward,
            matching the field convention.
        orientation: Non-zero sensing axis ``n``; it is normalized on
            construction. A multi-axis instrument is modelled as several
            co-located sensors with different axes.
        quantity: Which kinematic quantity the sensor reports.
        response: Complex instrument response (output per metre of displacement
            along the axis); defaults to the identity.
        calibration: Complex scalar amplitude and phase calibration.
        self_noise_psd: One-sided self-noise PSD in ``output_unit**2 / hertz``,
            or ``None`` for a noiseless sensor. Must return non-negative values.
        timing_offset: Documented timing offset ``tau`` in seconds; the channel
            records ``x(t - tau)`` and its spectrum carries ``exp(+i * omega * tau)``.
        output_unit: Unit of the channel's samples, or ``None`` for the SI unit
            of ``quantity`` (see :data:`QUANTITY_SI_UNITS`). Set it whenever
            ``calibration`` or ``response`` is not dimensionless, e.g.
            ``"counts"`` for a response in counts per metre.
    """

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float] = (0.0, 0.0, 1.0)
    quantity: SensorQuantity = SensorQuantity.DISPLACEMENT
    response: ComplexResponse = unit_response
    calibration: complex = 1.0 + 0.0j
    self_noise_psd: PSDCallable | None = None
    timing_offset: float = 0.0
    output_unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the identifier, geometry, response and self-noise callables."""
        if not self.name:
            raise ValueError("WitnessSensor.name must be non-empty.")
        if len(self.position) != 3:  # noqa: PLR2004
            raise ValueError(f"WitnessSensor.position must have three components, got {self.position}.")
        orientation = np.asarray(self.orientation, dtype=float)
        if orientation.shape != (3,):
            raise ValueError(f"WitnessSensor.orientation must have three components, got {self.orientation}.")
        if float(np.linalg.norm(orientation)) == 0.0:
            raise ValueError("WitnessSensor.orientation must be a non-zero axis.")
        if not callable(self.response):
            raise TypeError("WitnessSensor.response must be callable.")
        if self.self_noise_psd is not None and not callable(self.self_noise_psd):
            raise TypeError("WitnessSensor.self_noise_psd must be callable or None.")
        if not np.isfinite(self.timing_offset):
            raise ValueError(f"WitnessSensor.timing_offset must be finite, got {self.timing_offset}.")
        if self.output_unit is not None and not self.output_unit:
            raise ValueError("WitnessSensor.output_unit must be non-empty or None.")

    @property
    def unit(self) -> str:
        """Return the unit of the channel's samples."""
        return self.output_unit if self.output_unit is not None else QUANTITY_SI_UNITS[self.quantity]

    @property
    def unit_orientation(self) -> np.ndarray:
        """Return the sensing axis normalized to unit length, shape ``(3,)``."""
        axis = np.asarray(self.orientation, dtype=float)
        return axis / float(np.linalg.norm(axis))

    def self_noise(self, frequency: float) -> float:
        """Return the one-sided self-noise PSD at one frequency, or ``0`` if noiseless.

        Args:
            frequency: Frequency in hertz, strictly positive.

        Returns:
            A non-negative one-sided PSD in ``output_unit**2 / hertz``.
        """
        if frequency <= 0.0:
            raise ValueError(f"frequency must be positive, got {frequency}.")
        if self.self_noise_psd is None:
            return 0.0
        value = float(self.self_noise_psd(frequency))
        if value < 0.0:
            raise ValueError(f"self_noise_psd returned a negative value ({value}) at f={frequency}.")
        return value

    def transfer(self, frequency: float) -> complex:
        """Return the full complex displacement-to-output transfer at one frequency.

        The transfer is
        ``calibration * response(f) * quantity_transfer(quantity, f) * exp(+i * omega * tau)``,
        with ``omega = 2 * pi * frequency`` and ``tau`` the timing offset.

        Args:
            frequency: Frequency in hertz, strictly positive.

        Returns:
            A complex transfer in output units per metre of displacement along
            the sensing axis.
        """
        if frequency <= 0.0:
            raise ValueError(f"frequency must be positive, got {frequency}.")
        omega = 2.0 * np.pi * frequency
        timing = np.exp(1j * omega * self.timing_offset)
        return complex(
            complex(self.calibration)
            * complex(self.response(frequency))
            * quantity_transfer(self.quantity, frequency)
            * timing
        )


__all__ = [
    "QUANTITY_SI_UNITS",
    "ComplexResponse",
    "PSDCallable",
    "SensorQuantity",
    "WitnessSensor",
    "constant_response",
    "quantity_transfer",
    "unit_response",
]
