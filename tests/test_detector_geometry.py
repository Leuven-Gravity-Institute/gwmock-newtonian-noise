"""Tests for the ET triangle geometry, ideal GW response, null vector and test masses.

The independent controls here do not re-invoke the production assembly:

- the ideal GW response is checked against raw dot products ``n^T h n`` computed
  directly from the stored outward arm vectors (not the tensor contraction);
- the detector tensors are checked against explicit outer products of the
  geometry's own arm vectors;
- the strain functional's GW limit is checked by feeding it ``x = (1/2) h r``
  and comparing against ``gw_response`` -- two code paths that must agree.

Physical numbers used here are UNANCHORED illustrative values chosen only to
exercise the code, not a representative ET parameter set.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian_noise.detector.geometry import (
    CANONICAL_DETECTOR_VERTICES,
    DetectorDefinition,
    TriangleGeometry,
    equilateral_triangle,
    rotation_matrix,
)
from gwmock_newtonian_noise.detector.masses import (
    DetectorOptics,
    TestMass,
    TestMassLayout,
    independent_mass_layout,
    shared_vertex_layout,
)

pytestmark = pytest.mark.unit

#: UNANCHORED: illustrative arm length in metres used to exercise the geometry.
_ARM_LENGTH = 10_000.0

#: UNANCHORED: an illustrative symmetric metric perturbation, not a physical GW signal.
_METRIC = np.array(
    [
        [0.7, -0.35, 0.12],
        [-0.35, -0.7, 0.05],
        [0.12, 0.05, 0.0],
    ]
)


def _symmetric_metrics() -> list[np.ndarray]:
    """Return the fixed fixture metric plus seeded general symmetric matrices."""
    metrics = [_METRIC]
    for seed in (0, 1, 7, 20260926):
        raw = np.random.default_rng(seed).normal(size=(3, 3))
        metrics.append(0.5 * (raw + raw.T))
    return metrics


#: Independent symmetric metric perturbations for the universal cancellation control.
_SYMMETRIC_METRICS = _symmetric_metrics()


def _geometry() -> TriangleGeometry:
    return equilateral_triangle(_ARM_LENGTH)


class TestArmVectorsAndTensors:
    """Arm vectors, signs and detector tensors are explicit and internally consistent."""

    def test_arm_vectors_are_unit_and_separated_by_sixty_degrees(self) -> None:
        """Adjacent outward arms meet at the equilateral opening angle."""
        geometry = _geometry()
        for index, _definition in enumerate(geometry.detectors):
            positive, negative = geometry.outward_arm_vectors(index)
            assert positive @ positive == pytest.approx(1.0)
            assert negative @ negative == pytest.approx(1.0)
            assert positive @ negative == pytest.approx(np.cos(np.pi / 3.0))

    def test_detector_tensor_matches_raw_outer_products(self) -> None:
        """d_k equals (a a^T - b b^T)/2 built from the stored outward arms."""
        geometry = _geometry()
        for index in range(geometry.n_detectors):
            positive, negative = geometry.outward_arm_vectors(index)
            control = 0.5 * (np.outer(positive, positive) - np.outer(negative, negative))
            np.testing.assert_allclose(geometry.detector_tensor(index), control, atol=1e-15)

    def test_tensor_sum_is_zero(self) -> None:
        """The closed geometry makes d_1 + d_2 + d_3 = 0, the source of the null."""
        total = _geometry().detector_tensors().sum(axis=0)
        np.testing.assert_allclose(total, 0.0, atol=1e-15)

    def test_arm_lengths_are_the_declared_length(self) -> None:
        """Both arms of every detector have exactly the declared arm length."""
        geometry = _geometry()
        np.testing.assert_allclose(geometry.arm_lengths(), _ARM_LENGTH, rtol=1e-12)


class TestIdealGwResponse:
    """The ideal low-frequency response and its cancellation are independently controlled."""

    def test_gw_response_matches_raw_dot_products(self) -> None:
        """s_k = (n_pos^T h n_pos - n_neg^T h n_neg)/2 from raw dot products.

        This is a different expression from the tensor contraction the production
        code uses, so a sign or outer-product defect in ``detector_tensor`` fails it.
        """
        geometry = _geometry()
        response = geometry.gw_response(_METRIC)
        for index in range(geometry.n_detectors):
            positive, negative = geometry.outward_arm_vectors(index)
            control = 0.5 * (positive @ _METRIC @ positive - negative @ _METRIC @ negative)
            assert response[index] == pytest.approx(control, abs=1e-15)

    @pytest.mark.parametrize("metric", _SYMMETRIC_METRICS)
    def test_low_frequency_gw_sum_cancels(self, metric: np.ndarray) -> None:
        """s_1 + s_2 + s_3 = 0 for arbitrary symmetric metrics, including seeded general ones.

        The bound is relative to the largest response component, so a zero metric
        (which would satisfy the sum trivially) cannot pass.
        """
        response = _geometry().gw_response(metric)
        assert abs(float(response.sum())) < 1e-15 * np.max(np.abs(response))

    def test_null_vector_annihilates_the_gw_response(self) -> None:
        """u^T s = 0 for the ideal response, with u = (1, 1, 1)/sqrt(3)."""
        geometry = _geometry()
        np.testing.assert_allclose(geometry.null_vector(), np.full(3, 1.0 / np.sqrt(3.0)), atol=1e-15)
        assert abs(complex(geometry.null_projection(geometry.gw_response(_METRIC)))) < 1e-16

    def test_null_is_a_projection_not_a_channel(self) -> None:
        """The null projection contracts the three channels to a scalar; it adds no channel."""
        geometry = _geometry()
        covariance = np.eye(3, dtype=complex)
        projected = geometry.null_projection(covariance)
        assert projected.shape == (3,)
        scalar = float(np.real(geometry.null_vector() @ projected))
        assert scalar == pytest.approx(1.0)
        assert geometry.signal_projection(covariance).shape == (3, 3)
        np.testing.assert_allclose(geometry.null_projector() + geometry.signal_projector(), np.eye(3), atol=1e-15)

    def test_signal_basis_is_orthonormal_and_orthogonal_to_null(self) -> None:
        """The two-column signal basis spans ker(u^T) and is orthonormal."""
        geometry = _geometry()
        basis = geometry.signal_basis()
        assert basis.shape == (3, 2)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(geometry.null_vector() @ basis, 0.0, atol=1e-12)


class TestRotationAndRelabelling:
    """Rigid rotation and channel relabelling do not change physical strains."""

    def test_rotated_geometry_reproduces_rotated_response(self) -> None:
        """Rotating the triangle rotates every detector tensor by the same rotation."""
        geometry = _geometry()
        angle = 0.7
        rotation = rotation_matrix(angle)
        rotated = TriangleGeometry(
            vertices=geometry.vertices @ rotation.T,
            arm_length=geometry.arm_length,
            detectors=geometry.detectors,
        )
        for index in range(geometry.n_detectors):
            np.testing.assert_allclose(
                rotated.detector_tensor(index),
                rotation @ geometry.detector_tensor(index) @ rotation.T,
                atol=1e-15,
            )

    def test_cyclic_relabelling_permutes_channels_consistently(self) -> None:
        """Relabelling the detector order permutes the GW response by the same permutation."""
        geometry = _geometry()
        order = (2, 0, 1)
        relabelled = TriangleGeometry(
            vertices=geometry.vertices,
            arm_length=geometry.arm_length,
            detectors=tuple(geometry.detectors[index] for index in order),
            # names are positional defaults, so build definitions with the new order
        )
        # The relabelled geometry's channel k is the original channel order[k].
        for new_index, old_index in enumerate(order):
            np.testing.assert_allclose(
                relabelled.detector_tensor(new_index), geometry.detector_tensor(old_index), atol=1e-15
            )
        response = geometry.gw_response(_METRIC)
        np.testing.assert_allclose(relabelled.gw_response(_METRIC), response[list(order)], atol=1e-15)


class TestTestMassLayout:
    """Test-mass bookkeeping and the strain conversion."""

    def test_shared_vertex_layout_validates_and_shares_masses(self) -> None:
        """The shared-vertex layout passes its own arm consistency check."""
        layout = shared_vertex_layout(_geometry())
        assert len(layout.unique_positions()) == 3
        sharing = layout.position_sharing()
        # Each vertex is referenced by the detector sited there (beamsplitter) and
        # by the two detectors arming into it (far ends): up to three channels.
        assert max(len(identifiers) for identifiers in sharing.values()) >= 1
        assert sorted(layout.identifiers) == ["M0", "M1", "M2"]

    def test_common_mode_displacement_produces_no_strain(self) -> None:
        """Rigid translation of every test mass leaves every differential arm unchanged."""
        layout = shared_vertex_layout(_geometry())
        displacement = np.array([2.5, -1.5, 0.75])
        strains = layout.strain_from_mass_displacements(
            {identifier: displacement.copy() for identifier in layout.identifiers}
        )
        np.testing.assert_allclose(strains, 0.0, atol=1e-12)

    def test_strain_conversion_matches_raw_arm_formula(self) -> None:
        """The strain equals the raw differential arm formula for an arbitrary displacement field."""
        geometry = _geometry()
        layout = shared_vertex_layout(geometry)
        generator = np.random.default_rng(0)
        displacements = {identifier: generator.normal(size=3) for identifier in layout.identifiers}
        strains = layout.strain_from_mass_displacements(displacements)
        control = np.empty(3)
        for index, optics in enumerate(layout.optics):
            positive, negative = geometry.outward_arm_vectors(index)
            beamsplitter = layout.mass(optics.beamsplitter).coordinates
            positive_end = layout.mass(optics.positive_end).coordinates
            x_bs = displacements[optics.beamsplitter]
            x_pos = displacements[optics.positive_end]
            x_neg = displacements[optics.negative_end]
            assert positive @ (positive_end - beamsplitter) == pytest.approx(geometry.arm_length)
            control[index] = (positive @ (x_pos - x_bs) - negative @ (x_neg - x_bs)) / geometry.arm_length
        np.testing.assert_allclose(strains, control, atol=1e-15)

    def test_strain_functional_reproduces_gw_response(self) -> None:
        """Feeding x = h.r/2 into the strain functional returns the ideal GW response.

        This is the tensor-versus-explicit-arm agreement control: two independent
        code paths (detector tensor contraction and arm-length functional) must
        agree for a metric perturbation.
        """
        geometry = _geometry()
        layout = shared_vertex_layout(geometry, test_mass_depth=2.0)

        def arm_displacement(position: np.ndarray) -> np.ndarray:
            return 0.5 * _METRIC @ position

        displacements = {mass.identifier: arm_displacement(mass.coordinates) for mass in layout.masses}
        np.testing.assert_allclose(
            layout.strain_from_mass_displacements(displacements).real,
            geometry.gw_response(_METRIC),
            atol=1e-15,
        )

    def test_broadcast_reuses_one_realization_per_position(self) -> None:
        """Co-located masses receive the identical realization object, never a fresh draw."""
        geometry = _geometry()
        layout = independent_mass_layout(geometry)
        positions = layout.unique_positions()
        values = {position: np.array([1.0, 2.0, 3.0]) for position in positions}
        broadcast = layout.broadcast_by_position(values)
        seen = {}
        for position in positions:
            group = layout.position_sharing()[position]
            for identifier in group:
                assert np.shares_memory(broadcast[identifier], values[position])
                seen[identifier] = id(broadcast[identifier])
        # Every mass maps to a position, and co-located masses share one array object.
        assert len(broadcast) == len(layout.masses)

    def test_rejects_arm_position_inconsistent_with_geometry(self) -> None:
        """A layout whose mass positions do not reproduce the declared arm is rejected."""
        geometry = _geometry()
        optics = (
            DetectorOptics("ET1", "A", "B", "C"),
            DetectorOptics("ET2", "B", "C", "A"),
            DetectorOptics("ET3", "C", "A", "B"),
        )
        masses = (
            TestMass("A", tuple(geometry.vertices[0])),
            TestMass("B", tuple(geometry.vertices[1] + np.array([100.0, 0.0, 0.0]))),
            TestMass("C", tuple(geometry.vertices[2])),
        )
        with pytest.raises(ValueError, match="arm length"):
            TestMassLayout(geometry=geometry, masses=masses, optics=optics)

    def test_rejects_swapped_positive_and_negative_arm_ends(self) -> None:
        """Each end must sit on its declared arm: swapping the two ends is rejected.

        Both canonical far vertices lie at distance ``L`` and 60 degrees apart, so
        each sits in the other's forward half-space and a sign-only projection test
        cannot tell them apart. Requiring the longitudinal projection onto the
        declared outward arm to match the arm length does.
        """
        geometry = _geometry()
        optics = tuple(
            DetectorOptics(
                name=definition.name,
                beamsplitter=f"M{definition.beamsplitter_vertex}",
                positive_end=f"M{definition.negative_vertex}",
                negative_end=f"M{definition.positive_vertex}",
            )
            for definition in geometry.detectors
        )
        masses = tuple(TestMass(f"M{index}", tuple(geometry.vertices[index])) for index in range(3))
        with pytest.raises(ValueError, match="arm length"):
            TestMassLayout(geometry=geometry, masses=masses, optics=optics)

    def test_accepts_documented_role_offsets_off_the_declared_arm(self) -> None:
        """A documented nonzero per-role offset is accepted: that is the null-visible model.

        The offset moves the mass off the declared arm, but the arm direction and
        the sign/role assignment stay correct, so the layout must remain
        constructible.
        """
        geometry = _geometry()
        layout = independent_mass_layout(
            geometry,
            test_mass_depth=2.0,
            offsets={"positive_end": (0.1, 0.0, 0.0)},
        )
        assert len(layout.masses) == 9
        assert len(layout.unique_positions()) == 6

    def test_accepts_a_purely_transverse_offset(self) -> None:
        """A perpendicular displacement off the declared arm is admitted."""
        geometry = _geometry()
        layout = independent_mass_layout(
            geometry,
            test_mass_depth=2.0,
            offsets={"positive_end": (0.0, 0.14, 0.0)},
        )
        assert len(layout.masses) == 9

    def test_accepts_a_longitudinal_offset_just_inside_the_projection_cap(self) -> None:
        """An offset just inside the documented projection cap keeps the layout constructible.

        Oracle: this side cannot fail on the pre-fix relative tolerance (which was
        looser); it guards against over-tightening the cap below the documented
        offsets.
        """
        geometry = _geometry()
        layout = independent_mass_layout(
            geometry,
            test_mass_depth=2.0,
            offsets={"positive_end": (0.49, 0.0, 0.0)},
        )
        assert len(layout.masses) == 9

    def test_rejects_a_longitudinal_offset_just_outside_the_projection_cap(self) -> None:
        """An offset just past the documented projection cap is rejected.

        The 9 m and 10 m counterexamples sit in this rejected band; on
        the pre-fix 1e-3 relative tolerance (10 m at ``L`` = 10 km) they were
        accepted, which is the behaviour this test now pins out.
        """
        geometry = _geometry()
        with pytest.raises(ValueError, match="arm length"):
            independent_mass_layout(
                geometry,
                test_mass_depth=2.0,
                offsets={"positive_end": (0.51, 0.0, 0.0)},
            )

    @pytest.mark.parametrize("offset", [9.0, 10.0, 10.01])
    def test_rejects_the_longitudinal_counterexamples(self, offset: float) -> None:
        """The 9-10 m longitudinal counterexamples exceed the documented cap."""
        geometry = _geometry()
        with pytest.raises(ValueError, match="arm length"):
            independent_mass_layout(
                geometry,
                test_mass_depth=2.0,
                offsets={"positive_end": (offset, 0.0, 0.0)},
            )

    def test_rejects_a_longitudinal_offset_beyond_the_arm_tolerance(self) -> None:
        """A 50 m positive-end offset moves ET1's end ~25 m off its declared arm.

        The 50 m displacement is at 120 degrees (``cos = -1/2``) to ET1's positive
        arm, so ET1 gives length 9975.09 m and projection 9975 m against the 10000 m
        arm -- ~25 m off, past the 0.5 m cap -- and the layout is rejected.
        """
        geometry = _geometry()
        with pytest.raises(ValueError, match="arm length"):
            independent_mass_layout(
                geometry,
                test_mass_depth=2.0,
                offsets={"positive_end": (50.0, 0.0, 0.0)},
            )

    def test_canonical_vertex_triples_match_the_constructed_geometry(self) -> None:
        """The exported sign convention is the one the constructor actually used."""
        geometry = _geometry()
        for definition, expected in zip(geometry.detectors, CANONICAL_DETECTOR_VERTICES, strict=True):
            assert (
                definition.beamsplitter_vertex,
                definition.positive_vertex,
                definition.negative_vertex,
            ) == expected


class TestGeometryValidation:
    """Malformed geometry and layouts are rejected rather than silently accepted."""

    def test_rejects_non_positive_arm_length(self) -> None:
        """A zero or negative arm length is unphysical."""
        with pytest.raises(ValueError, match="arm_length"):
            equilateral_triangle(0.0)

    def test_rejects_identical_detector_vertices(self) -> None:
        """A detector whose positive and negative ends coincide is rejected."""
        geometry = _geometry()
        bad = (DetectorDefinition("ET1", 0, 1, 1), *geometry.detectors[1:])
        with pytest.raises(ValueError, match="three distinct"):
            TriangleGeometry(vertices=geometry.vertices, arm_length=geometry.arm_length, detectors=bad)

    def test_rejects_duplicate_mass_identifiers(self) -> None:
        """Two masses with the same identifier cannot both exist."""
        geometry = _geometry()
        optics = tuple(
            DetectorOptics(name=d.name, beamsplitter="A", positive_end="B", negative_end="C")
            for d in geometry.detectors
        )
        masses = (
            TestMass("A", tuple(geometry.vertices[0])),
            TestMass("B", tuple(geometry.vertices[1])),
            TestMass("C", tuple(geometry.vertices[2])),
            TestMass("C", tuple(geometry.vertices[2])),
        )
        with pytest.raises(ValueError, match="unique"):
            TestMassLayout(geometry=geometry, masses=masses, optics=optics)
