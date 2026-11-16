"""Explicit Einstein-Telescope triangle geometry and ideal gravitational-wave response.

The three ET interferometers form an equilateral triangle: each detector sits at
one vertex and its two outward arms run along two of the three shared physical
sides, so a single detector output is the differential response of two arms.
This module fixes the geometry, the outward arm vectors, the per-detector signs
and the resulting detector tensors explicitly -- no channel identity is ever
inferred from a name.

Geometry and the cyclic detector-tensor convention follow the original
triple-Michelson concept: Freise, A., Chelkowski, S., Hild, S., Del Pozzo, W.,
Perreca, A. and Vecchio, A. (2009), "Triple Michelson interferometer for a
third-generation gravitational wave detector," Classical and Quantum Gravity,
26(8), 085012, arXiv:0804.1036. For outward unit arm vectors ``a_k`` and
``b_k``, detector ``k``'s tensor is

    d_k = (a_k (x) a_k - b_k (x) b_k) / 2,

and the ideal long-wavelength metric response is ``s_k = d_k : h``. In the closed
triangle the three tensors sum to zero, so ``s_1 + s_2 + s_3 = 0`` and the
normalized null vector ``u = (1, 1, 1) / sqrt(3)`` is an exact left null vector
of the low-frequency response. Which vertex each detector uses, and which two
arms it reads, is data in this module, not an assumption.

Coordinates are right-handed with ``x`` and ``y`` horizontal and ``z`` vertical
(the free-surface normal); a strain channel is dimensionless, arm vectors are
dimensionless, and positions/arm length are in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Number of vertices (equivalently physical arms and detectors) in the triangle.
N_VERTICES = 3

#: Number of strain channels in the triangle.
N_DETECTORS = 3


def _unit(vector: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return ``vector`` scaled to unit length, rejecting a vanishing vector."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero-length vector.")
    return vector / norm


@dataclass(frozen=True)
class DetectorDefinition:
    """One Michelson detector: a beamsplitter vertex and its two signed outward arms.

    Attributes:
        name: Channel name.
        beamsplitter_vertex: Index of the vertex the beamsplitter sits at.
        positive_vertex: Index of the vertex at the far end of the positive arm.
        negative_vertex: Index of the vertex at the far end of the negative arm.
    """

    name: str
    beamsplitter_vertex: int
    positive_vertex: int
    negative_vertex: int


@dataclass(frozen=True)
class TriangleGeometry:
    """Explicit geometry of the closed three-arm ET triangle and its detectors.

    Attributes:
        vertices: ``(3, 3)`` array of vertex positions in metres.
        arm_length: Common arm length ``L`` in metres, strictly positive.
        detectors: The three :class:`DetectorDefinition` entries, in channel order.
    """

    vertices: NDArray[np.float64]
    arm_length: float
    detectors: tuple[DetectorDefinition, DetectorDefinition, DetectorDefinition]

    def __post_init__(self) -> None:
        """Validate vertex positions, positive arm length and detector indices."""
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.shape != (N_VERTICES, 3):
            raise ValueError(f"vertices must have shape ({N_VERTICES}, 3), got {vertices.shape}.")
        if not (self.arm_length > 0.0):
            raise ValueError(f"arm_length must be positive, got {self.arm_length}.")
        if len(self.detectors) != N_DETECTORS:
            raise ValueError(f"detectors must have {N_DETECTORS} entries, got {len(self.detectors)}.")
        names = [detector.name for detector in self.detectors]
        if len(set(names)) != N_DETECTORS:
            raise ValueError(f"detector names must be unique, got {names}.")
        for detector in self.detectors:
            indices = {detector.beamsplitter_vertex, detector.positive_vertex, detector.negative_vertex}
            if len(indices) != N_VERTICES or min(indices) < 0 or max(indices) >= N_VERTICES:
                raise ValueError(
                    "each detector needs three distinct vertices in "
                    f"[0, {N_VERTICES}), got bs={detector.beamsplitter_vertex}, "
                    f"positive={detector.positive_vertex}, negative={detector.negative_vertex}."
                )

    @property
    def names(self) -> tuple[str, str, str]:
        """Return the three channel names in channel order."""
        return (self.detectors[0].name, self.detectors[1].name, self.detectors[2].name)

    @property
    def n_detectors(self) -> int:
        """Number of strain channels (always :data:`N_DETECTORS`)."""
        return N_DETECTORS

    def outward_arm_vectors(self, detector: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return detector ``detector``'s ``(positive, negative)`` outward unit arm vectors.

        The vectors point from the beamsplitter vertex to the two far vertices
        and are dimensionless; their physical length is ``arm_length``.
        """
        definition = self.detectors[detector]
        beamsplitter = self.vertices[definition.beamsplitter_vertex]
        positive = self.vertices[definition.positive_vertex] - beamsplitter
        negative = self.vertices[definition.negative_vertex] - beamsplitter
        return _unit(positive), _unit(negative)

    def detector_tensor(self, detector: int) -> NDArray[np.float64]:
        """Return detector ``detector``'s dimensionless ``(3, 3)`` detector tensor."""
        positive, negative = self.outward_arm_vectors(detector)
        return 0.5 * (np.outer(positive, positive) - np.outer(negative, negative))

    def detector_tensors(self) -> NDArray[np.float64]:
        """Return the ``(3, 3, 3)`` stack of all detector tensors, indexed by detector first."""
        return np.stack([self.detector_tensor(i) for i in range(N_DETECTORS)])

    def arm_lengths(self) -> NDArray[np.float64]:
        """Return the two physical arm lengths of every detector, shape ``(3, 2)``."""
        lengths = np.empty((N_DETECTORS, 2))
        for index, definition in enumerate(self.detectors):
            beamsplitter = self.vertices[definition.beamsplitter_vertex]
            lengths[index, 0] = np.linalg.norm(self.vertices[definition.positive_vertex] - beamsplitter)
            lengths[index, 1] = np.linalg.norm(self.vertices[definition.negative_vertex] - beamsplitter)
        return lengths

    def gw_response(self, metric_perturbation: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the ideal long-wavelength strain response ``s_k = d_k : h``.

        Args:
            metric_perturbation: Symmetric ``(3, 3)`` metric perturbation.

        Returns:
            A length-3 real array of dimensionless strains.
        """
        h = np.asarray(metric_perturbation, dtype=float)
        if h.shape != (3, 3):
            raise ValueError(f"metric_perturbation must have shape (3, 3), got {h.shape}.")
        return np.array([float(np.sum(self.detector_tensor(i) * h)) for i in range(N_DETECTORS)])

    def strain_from_test_mass_displacements(
        self, displacements: NDArray[np.floating] | NDArray[np.complexfloating]
    ) -> NDArray[np.complex128]:
        """Return each detector's complex strain from its three test-mass displacements.

        ``displacements[k]`` is a ``(3, 3)`` array whose rows are the three
        vertex displacement vectors of detector ``k``, ordered
        ``(beamsplitter, positive_end, negative_end)``. The response is the
        explicit differential optical-length functional

            s_k = ( a_k . (x_pos - x_bs) - b_k . (x_neg - x_bs) ) / L,

        which contracts to ``(a_k (x) a_k - b_k (x) b_k) / 2 : h`` for a
        long-wavelength metric perturbation ``h`` (the arm displacement is then
        ``x = (1/2) h . r`` and the arm vectors have length ``L``). It is
        complex-linear, so spectral displacements pass through unchanged.

        Args:
            displacements: ``(3, 3, 3)`` array indexed ``[detector, mass, axis]``.

        Returns:
            A length-3 complex strain array.
        """
        displacement = np.asarray(displacements, dtype=complex)
        if displacement.shape != (N_DETECTORS, 3, 3):
            raise ValueError(f"displacements must have shape ({N_DETECTORS}, 3, 3), got {displacement.shape}.")
        strains = np.empty(N_DETECTORS, dtype=complex)
        for index in range(N_DETECTORS):
            positive, negative = self.outward_arm_vectors(index)
            x_bs = displacement[index, 0]
            x_pos = displacement[index, 1]
            x_neg = displacement[index, 2]
            strains[index] = (positive @ (x_pos - x_bs) - negative @ (x_neg - x_bs)) / self.arm_length
        return strains

    def null_vector(self) -> NDArray[np.float64]:
        """Return the normalized null vector ``u = (1, 1, 1) / sqrt(3)`` on the three strain channels."""
        return np.full(N_DETECTORS, 1.0 / np.sqrt(N_DETECTORS))

    def null_projector(self) -> NDArray[np.float64]:
        """Return ``P_n = u u^T``, the rank-one projector onto the null direction."""
        u = self.null_vector()
        return np.outer(u, u)

    def signal_projector(self) -> NDArray[np.float64]:
        """Return ``P_g = I - u u^T``, the projector onto the GW-sensitive subspace ``ker(u^T)``."""
        return np.eye(N_DETECTORS) - self.null_projector()

    def signal_basis(self) -> NDArray[np.float64]:
        """Return a ``(3, 2)`` orthonormal basis of ``ker(u^T)``, the GW-sensitive subspace.

        The columns are an explicit basis, so a caller can construct a
        deliberately null-invisible strain vector without ever treating the null
        as a fourth channel.
        """
        basis, _ = np.linalg.qr(self.signal_projector())
        # ``signal_projector`` has rank two; its first two QR columns span the image.
        return basis[:, :2]

    def null_projection(self, channels: NDArray[np.floating] | NDArray[np.complexfloating]) -> NDArray[np.complex128]:
        """Return ``u^T channels`` along channel axis 0 (adds no channel)."""
        return np.tensordot(self.null_vector(), np.asarray(channels, dtype=complex), axes=([0], [0]))

    def signal_projection(self, channels: NDArray[np.floating] | NDArray[np.complexfloating]) -> NDArray[np.complex128]:
        """Return ``(I - u u^T) channels`` along channel axis 0 (adds no channel)."""
        return np.tensordot(self.signal_projector(), np.asarray(channels, dtype=complex), axes=([1], [0]))


def equilateral_triangle(
    arm_length: float,
    orientation: float = 0.0,
    names: tuple[str, str, str] = ("ET1", "ET2", "ET3"),
) -> TriangleGeometry:
    """Construct the canonical closed ET triangle with explicit vertices and signs.

    Vertices lie on a circle of circumradius ``arm_length / sqrt(3)`` at angles
    ``90 + 120 k`` degrees (plus ``orientation``). Detector ``k`` sits at vertex
    ``k`` and reads outward arms to vertices ``k + 1`` and ``k + 2`` (mod 3), the
    cyclic triple-Michelson assignment. The three detector tensors sum to zero.

    Args:
        arm_length: Common arm length in metres, strictly positive.
        orientation: Global rotation of the whole triangle in the horizontal
            plane, in radians. A rigid rotation leaves every strain unchanged.
        names: Per-detector channel names.

    Returns:
        The :class:`TriangleGeometry`.
    """
    if not (arm_length > 0.0):
        raise ValueError(f"arm_length must be positive, got {arm_length}.")
    circumradius = arm_length / np.sqrt(3.0)
    base_angles = np.deg2rad(np.array([90.0, 210.0, 330.0])) + orientation
    vertices = np.column_stack([circumradius * np.cos(base_angles), circumradius * np.sin(base_angles), np.zeros(3)])
    detectors = tuple(
        DetectorDefinition(
            name=names[k],
            beamsplitter_vertex=k,
            positive_vertex=(k + 1) % N_VERTICES,
            negative_vertex=(k + 2) % N_VERTICES,
        )
        for k in range(N_DETECTORS)
    )
    return TriangleGeometry(vertices=vertices, arm_length=float(arm_length), detectors=detectors)  # type: ignore[arg-type]


def rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Return the ``(3, 3)`` rotation about the vertical axis by ``angle`` radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


#: The cyclic ``(beamsplitter, positive, negative)`` vertex triples used by
#: :func:`equilateral_triangle`; re-exported so a caller can reproduce the sign
#: convention without hard-coding it.
CANONICAL_DETECTOR_VERTICES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

__all__ = [
    "CANONICAL_DETECTOR_VERTICES",
    "N_DETECTORS",
    "N_VERTICES",
    "DetectorDefinition",
    "TriangleGeometry",
    "equilateral_triangle",
    "rotation_matrix",
]
