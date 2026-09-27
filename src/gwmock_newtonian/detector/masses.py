"""Physical test masses, suspension geometry and the strain conversion.

A detector's strain is built from the displacements of its beamsplitter and its
two far (end) test masses. This module makes that structure explicit:

- :class:`TestMass` is a physical mass with an identifier and a position.
- :class:`Suspension` converts a test-mass acceleration into a displacement
  (a free mass at low frequency gives ``x = -a / omega**2``; Harms, J. (2019),
  "Terrestrial Gravity Fluctuations," Living Reviews in Relativity, 22, 6,
  Secs. 2.3-2.4), and may be replaced by any caller-supplied complex response.
- :class:`TestMassLayout` assigns a mass to each beamsplitter and arm-end role
  and enforces the shared-structure bookkeeping: a mass is a single physical
  object with a single position and therefore a single realization, reused by
  every channel that references it.

The layout is tied to a :class:`~gwmock_newtonian.detector.geometry.TriangleGeometry`
and validated against it, so the arm directions and signs used for the strain
conversion come from one source of truth rather than being re-typed here.

The arm-role validation is a **sanity separator**, not an engineering accuracy
claim and not externally anchored: the fixtures in this package are synthetic
and illustrative, the documented per-role ``offsets`` are sub-metre position
perturbations (the fixtures use 0.1-0.14 m), and the explicit metre caps below
separate those from a swapped end or a wrong arm, which displace the projection
by of order the arm length. See ``_MAX_ARM_PROJECTION_OFFSET_METRES`` and
``_MAX_ARM_LENGTH_OFFSET_METRES``.

Strain is dimensionless (metres per metre); masses are in kilograms when a
physical mass is supplied, positions in metres, accelerations in metres/second^2
and suspension responses in metres per (metre/second^2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gwmock_newtonian.detector.geometry import N_DETECTORS, TriangleGeometry

#: Maximum admissible longitudinal displacement of an arm end, in metres: the
#: amount by which the projection of ``end - beamsplitter`` onto the declared
#: outward unit arm may differ from the arm length. This is a documented SANITY
#: separator, not an engineering accuracy claim and not externally anchored: the
#: documented per-role ``offsets`` are sub-metre position perturbations (the
#: fixtures use 0.1-0.14 m), while a swapped end or a mass on a different arm
#: moves the projection by of order the arm length -- ``L / 2`` = 5000 m on the
#: 10 km fixture. The cap is absolute and independent of ``L``: a test-mass
#: placement offset is a length, not a fraction of the arm.
_MAX_ARM_PROJECTION_OFFSET_METRES = 0.5

#: Maximum admissible physical arm-length error, in metres. This bounds a
#: transverse/off-axis displacement of an end: a purely perpendicular shift ``t``
#: changes ``||end - beamsplitter||`` by ``t**2 / (2 * L)``, so this cap admits an
#: off-axis displacement up to ``sqrt(2 * cap * L)`` -- about 100 m at
#: ``L`` = 10 km, growing as ``sqrt(L)``. Absolute, like the projection cap above.
_MAX_ARM_LENGTH_OFFSET_METRES = 0.5

#: Caller-supplied suspension response: acceleration (m/s^2) -> displacement (m)
#: as a function of angular frequency (rad/s), returning a complex scalar.
SuspensionResponse = Callable[[float], complex]


@dataclass(frozen=True)
class Suspension:
    """Displacement response of a suspended test mass to a test-mass acceleration.

    Attributes:
        response: Callable mapping an angular frequency ``omega`` in rad/s to
            the complex displacement per unit acceleration, in seconds^2. Its
            default is the free-mass response ``-1 / omega**2``.
        name: Human-readable identifier for provenance.
    """

    response: SuspensionResponse = None  # type: ignore[assignment]
    name: str = "free_mass"

    def __post_init__(self) -> None:
        """Install the free-mass response when no callable is supplied."""
        if self.response is None:
            object.__setattr__(self, "response", free_mass_response)
        if not callable(self.response):
            raise TypeError("Suspension.response must be callable.")

    def displacement(
        self, acceleration: NDArray[np.floating] | NDArray[np.complexfloating], angular_frequency: float
    ) -> NDArray[np.complex128]:
        """Return the complex displacement for an acceleration at one angular frequency."""
        if angular_frequency <= 0.0:
            raise ValueError(f"angular_frequency must be positive, got {angular_frequency}.")
        value = complex(self.response(angular_frequency))
        return value * np.asarray(acceleration, dtype=complex)


def free_mass_response(angular_frequency: float) -> complex:
    """Return the free-mass displacement per unit acceleration, ``-1 / omega**2``.

    This is the low-frequency limit of a test mass driven above its pendulum
    resonance (Harms 2019, Sec. 2.3).  ``omega`` is in rad/s and the result is
    in seconds^2.
    """
    if angular_frequency <= 0.0:
        raise ValueError(f"angular_frequency must be positive, got {angular_frequency}.")
    return -1.0 / (angular_frequency**2)


@dataclass(frozen=True)
class TestMass:
    """A single physical test mass.

    Attributes:
        identifier: Unique identifier within a layout.
        position: ``(x, y, z)`` position in metres, ``z`` positive downward from
            the free surface, matching the seismic-field convention.
    """

    __test__ = False

    identifier: str
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Reject an empty identifier or a wrong-length position."""
        if not self.identifier:
            raise ValueError("TestMass.identifier must be non-empty.")
        if len(self.position) != 3:  # noqa: PLR2004
            raise ValueError(f"TestMass.position must have three components, got {self.position}.")

    @property
    def coordinates(self) -> NDArray[np.float64]:
        """Return the position as a length-3 float array."""
        return np.asarray(self.position, dtype=float)


@dataclass(frozen=True)
class DetectorOptics:
    """Which physical masses a detector's beamsplitter and two arms use.

    Attributes:
        name: Channel name, matching the geometry's detector name.
        beamsplitter: Identifier of the beamsplitter test mass.
        positive_end: Identifier of the far mass on the positive-arm vertex.
        negative_end: Identifier of the far mass on the negative-arm vertex.
    """

    name: str
    beamsplitter: str
    positive_end: str
    negative_end: str

    @property
    def mass_identifiers(self) -> tuple[str, str, str]:
        """Return ``(beamsplitter, positive_end, negative_end)`` in channel order."""
        return (self.beamsplitter, self.positive_end, self.negative_end)


@dataclass(frozen=True)
class TestMassLayout:
    """Physical test masses, their sharing structure and the detectors that use them.

    Attributes:
        geometry: The triangle geometry the layout is validated against.
        masses: All physical test masses.
        optics: Per-detector mass assignment, one entry per geometry detector, in
            channel order.
        suspension: Suspension response applied to every mass.
    """

    geometry: TriangleGeometry
    masses: tuple[TestMass, ...]
    optics: tuple[DetectorOptics, ...]
    suspension: Suspension = Suspension()
    suspensions: dict[str, Suspension] | None = None

    __test__ = False

    def __post_init__(self) -> None:
        """Validate mass uniqueness, detector coverage and geometry consistency."""
        identifiers = [mass.identifier for mass in self.masses]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"test-mass identifiers must be unique, got {identifiers}.")
        if len(self.optics) != N_DETECTORS:
            raise ValueError(f"optics must have {N_DETECTORS} entries, got {len(self.optics)}.")
        if self.suspensions is not None:
            unknown = set(self.suspensions) - set(identifiers)
            if unknown:
                raise ValueError(f"per-mass suspensions reference unknown masses {sorted(unknown)}.")
        by_id = {mass.identifier: mass for mass in self.masses}
        expected_names = self.geometry.names
        for index, optics in enumerate(self.optics):
            if optics.name != expected_names[index]:
                raise ValueError(
                    f"optics[{index}].name {optics.name!r} does not match geometry name {expected_names[index]!r}."
                )
            for role, identifier in zip(
                ("beamsplitter", "positive_end", "negative_end"), optics.mass_identifiers, strict=True
            ):
                if identifier not in by_id:
                    raise ValueError(f"detector {optics.name!r} {role} references unknown test mass {identifier!r}.")
            if len(set(optics.mass_identifiers)) != N_DETECTORS:
                raise ValueError(
                    f"detector {optics.name!r} must use three distinct test masses, got {optics.mass_identifiers}."
                )
            self._validate_arm_geometry(index, optics, by_id)

    def suspension_for(self, identifier: str) -> Suspension:
        """Return the suspension response governing the given test mass."""
        if self.suspensions is not None and identifier in self.suspensions:
            return self.suspensions[identifier]
        return self.suspension

    def _validate_arm_geometry(self, index: int, optics: DetectorOptics, by_id: dict[str, TestMass]) -> None:
        """Check each assigned mass lies on its declared arm, within the documented caps.

        The longitudinal projection of the end onto the declared outward unit arm
        must match ``arm_length`` within ``_MAX_ARM_PROJECTION_OFFSET_METRES``, and
        the physical arm length within ``_MAX_ARM_LENGTH_OFFSET_METRES``. The
        projection cap is the sanity separator that tells a detector's two ends
        apart: a swapped end or a mass assigned to a different arm changes the
        projection by of order the arm length (``L / 2``), while the documented
        per-role ``offsets`` are sub-metre position perturbations. The length cap
        admits a transverse/perpendicular offset -- the documented ``offsets``
        model used to build a null-visible response -- up to the off-axis radius
        ``sqrt(2 * cap * L)``. These are documented sanity caps, not an engineering
        accuracy claim; see the module docstring.
        """
        positive, negative = self.geometry.outward_arm_vectors(index)
        beamsplitter = by_id[optics.beamsplitter].coordinates
        arm_length = self.geometry.arm_length
        for role, outward, identifier in (
            ("positive_end", positive, optics.positive_end),
            ("negative_end", negative, optics.negative_end),
        ):
            delta = by_id[identifier].coordinates - beamsplitter
            length = float(np.linalg.norm(delta))
            projection = float(delta @ outward)
            if (
                abs(projection - arm_length) > _MAX_ARM_PROJECTION_OFFSET_METRES
                or abs(length - arm_length) > _MAX_ARM_LENGTH_OFFSET_METRES
            ):
                raise ValueError(
                    f"detector {optics.name!r} {role} does not sit on its declared arm: arm length {length:.6g} m "
                    f"and projection {projection:.6g} m must both match the geometry arm length {arm_length:.6g} m "
                    f"within {_MAX_ARM_PROJECTION_OFFSET_METRES:g} m."
                )

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Return every test-mass identifier, in layout order."""
        return tuple(mass.identifier for mass in self.masses)

    def mass(self, identifier: str) -> TestMass:
        """Return the test mass with the given identifier."""
        for candidate in self.masses:
            if candidate.identifier == identifier:
                return candidate
        raise KeyError(f"unknown test mass {identifier!r}.")

    def unique_positions(self) -> tuple[tuple[float, float, float], ...]:
        """Return the unique test-mass positions, in first-seen layout order.

        A single physical position receives exactly one realization; every mass
        at that position reuses it.
        """
        seen: dict[tuple[float, float, float], None] = {}
        for candidate in self.masses:
            seen.setdefault(candidate.position, None)
        return tuple(seen)

    def position_sharing(self) -> dict[tuple[float, float, float], tuple[str, ...]]:
        """Map each unique position to the identifiers of every mass there."""
        groups: dict[tuple[float, float, float], list[str]] = {}
        for candidate in self.masses:
            groups.setdefault(candidate.position, []).append(candidate.identifier)
        return {position: tuple(identifiers) for position, identifiers in groups.items()}

    def broadcast_by_position(
        self, values: dict[tuple[float, float, float], NDArray[np.floating]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return per-mass values from per-position values, sharing each realization.

        Args:
            values: Mapping from a unique position to its realization.

        Returns:
            Mapping from every mass identifier to the realization at its
            position. Two co-located masses receive the identical array object,
            so a shared realization is never generated twice.
        """
        result: dict[str, NDArray[np.float64]] = {}
        for candidate in self.masses:
            if candidate.position not in values:
                raise KeyError(f"no realization supplied for position {candidate.position}.")
            result[candidate.identifier] = values[candidate.position]
        return result

    def strain_from_mass_displacements(
        self, displacements: dict[str, NDArray[np.floating] | NDArray[np.complexfloating]]
    ) -> NDArray[np.complex128]:
        """Return the three complex strains from per-mass displacement vectors.

        The strain functional is complex-linear, so a spectral (complex)
        displacement is passed through without taking a real part; callers that
        need a time-domain strain take the real part after combining modes.

        Args:
            displacements: Mapping from mass identifier to a length-3 displacement.

        Returns:
            A length-3 complex strain array.
        """
        stacked = np.empty((N_DETECTORS, 3, 3), dtype=complex)
        for index, optics in enumerate(self.optics):
            for role, identifier in enumerate(optics.mass_identifiers):
                if identifier not in displacements:
                    raise KeyError(f"no displacement supplied for test mass {identifier!r}.")
                value = np.asarray(displacements[identifier])
                if value.shape != (3,):
                    raise ValueError(f"displacement for {identifier!r} must have shape (3,), got {value.shape}.")
                stacked[index, role] = value.astype(complex)
        return self.geometry.strain_from_test_mass_displacements(stacked).astype(complex)

    def strain_from_accelerations(
        self, accelerations: dict[str, NDArray[np.floating] | NDArray[np.complexfloating]], angular_frequency: float
    ) -> NDArray[np.complex128]:
        """Return the three complex strains from per-mass accelerations at one angular frequency.

        Each acceleration is converted to a displacement by the suspension
        response (complex, so a phase or a pendulum resonance can be supplied),
        then contracted with the detector's explicit arm vectors.
        """
        displacements = {
            identifier: self.suspension_for(identifier).displacement(
                np.asarray(acceleration, dtype=complex), angular_frequency
            )
            for identifier, acceleration in accelerations.items()
        }
        return self.strain_from_mass_displacements(displacements)


def shared_vertex_layout(
    geometry: TriangleGeometry,
    suspension: Suspension | None = None,
    prefix: str = "M",
    test_mass_depth: float = 0.0,
) -> TestMassLayout:
    """Build the physical shared-vertex ET layout: one test mass per triangle vertex.

    Each vertex hosts a single mass that serves as the beamsplitter of the
    detector sited there and as the far-end mass of the two detectors that arm
    into it. This is the layout in which every channel genuinely shares physical
    masses.

    Args:
        geometry: Triangle geometry.
        suspension: Suspension response; defaults to a free mass.
        prefix: Identifier prefix.
        test_mass_depth: Depth of every test mass below the free surface in
            metres (``z`` positive downward). Non-zero by default is not
            assumed; the caller supplies the physical value, and a strictly
            positive depth avoids the free-surface kernel singularity.
    """
    masses = tuple(
        TestMass(
            identifier=f"{prefix}{index}",
            position=(
                float(geometry.vertices[index][0]),
                float(geometry.vertices[index][1]),
                float(test_mass_depth),
            ),
        )
        for index in range(geometry.n_detectors)
    )
    optics = tuple(
        DetectorOptics(
            name=definition.name,
            beamsplitter=f"{prefix}{definition.beamsplitter_vertex}",
            positive_end=f"{prefix}{definition.positive_vertex}",
            negative_end=f"{prefix}{definition.negative_vertex}",
        )
        for definition in geometry.detectors
    )
    return TestMassLayout(
        geometry=geometry,
        masses=masses,
        optics=optics,
        suspension=suspension if suspension is not None else Suspension(),
    )


def independent_mass_layout(  # noqa: PLR0913, PLR0917 -- one parameter per physically distinct input.
    geometry: TriangleGeometry,
    suspension: Suspension | None = None,
    offsets: dict[str, Sequence[float]] | None = None,
    prefix: str = "M",
    test_mass_depth: float = 0.0,
    suspensions_by_role: dict[str, Suspension] | None = None,
) -> TestMassLayout:
    """Build a layout with a distinct mass per (detector, role), all at the vertices.

    Every detector gets its own beamsplitter and two far-end masses, so no mass
    is shared between channels. With zero offsets and a single suspension
    response the co-located masses still produce an exactly null-invisible
    displacement field; a non-zero per-role offset (metres) or a role-dependent
    suspension makes the Newtonian response null-visible.

    Args:
        geometry: Triangle geometry.
        suspension: Default suspension response for every mass; defaults to a free mass.
        offsets: Optional mapping from role name (``"beamsplitter"``,
            ``"positive_end"``, ``"negative_end"``) to a length-3 offset in
            metres added to every mass in that role.
        prefix: Identifier prefix.
        test_mass_depth: Depth of every test mass below the free surface in metres.
        suspensions_by_role: Optional mapping from role name to the suspension
            governing every mass in that role, overriding ``suspension``; this is
            how a physically null-visible response is built without moving masses.

    Returns:
        The :class:`TestMassLayout`.
    """
    offsets = offsets or {}
    unknown = set(offsets) - {"beamsplitter", "positive_end", "negative_end"}
    if unknown:
        raise ValueError(f"unknown offset roles {sorted(unknown)}.")
    suspensions_by_role = suspensions_by_role or {}
    unknown_suspensions = set(suspensions_by_role) - {"beamsplitter", "positive_end", "negative_end"}
    if unknown_suspensions:
        raise ValueError(f"unknown suspension roles {sorted(unknown_suspensions)}.")
    masses: list[TestMass] = []
    optics: list[DetectorOptics] = []
    suspensions: dict[str, Suspension] = {}
    for index, definition in enumerate(geometry.detectors):
        roles = {
            "beamsplitter": definition.beamsplitter_vertex,
            "positive_end": definition.positive_vertex,
            "negative_end": definition.negative_vertex,
        }
        identifiers: dict[str, str] = {}
        for role, vertex in roles.items():
            identifier = f"{prefix}{index}_{role}"
            base = geometry.vertices[vertex].copy()
            base[2] = float(test_mass_depth)
            position = base + np.asarray(offsets.get(role, (0.0, 0.0, 0.0)), dtype=float)
            masses.append(TestMass(identifier=identifier, position=tuple(float(value) for value in position)))
            identifiers[role] = identifier
            if role in suspensions_by_role:
                suspensions[identifier] = suspensions_by_role[role]
        optics.append(
            DetectorOptics(
                name=definition.name,
                beamsplitter=identifiers["beamsplitter"],
                positive_end=identifiers["positive_end"],
                negative_end=identifiers["negative_end"],
            )
        )
    return TestMassLayout(
        geometry=geometry,
        masses=tuple(masses),
        optics=tuple(optics),
        suspension=suspension if suspension is not None else Suspension(),
        suspensions=suspensions or None,
    )


def collect_positions(layouts: Iterable[TestMassLayout]) -> tuple[tuple[float, float, float], ...]:
    """Return the union of unique positions across several layouts, in first-seen order."""
    seen: dict[tuple[float, float, float], None] = {}
    for layout in layouts:
        for position in layout.unique_positions():
            seen.setdefault(position, None)
    return tuple(seen)


__all__ = [
    "DetectorOptics",
    "Suspension",
    "SuspensionResponse",
    "TestMass",
    "TestMassLayout",
    "collect_positions",
    "free_mass_response",
    "independent_mass_layout",
    "shared_vertex_layout",
]
