"""The generic family consumer, the Rayleigh adapter, mixtures and the no-site-default rule."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _family_fixtures import assert_covariance_close, body, love, model, point, rayleigh

from gwmock_newtonian.families import (
    BodyWaveFamily,
    CavityModel,
    FamilyJointModel,
    FamilyMixture,
    LoveWaveFamily,
    PointSourceFamily,
    RayleighFamily,
    family_joint_covariance,
    family_joint_mode_responses,
    phase_velocity_spread,
)
from gwmock_newtonian.families.love import LayerOverHalfSpace
from gwmock_newtonian.families.point import PointSource
from gwmock_newtonian.witness.response import joint_covariance, joint_mode_responses

FREQUENCIES = np.array([0.0, 0.5, 2.0, 5.0])


class TestRayleighAdapter:
    """The generic consumer reproduces the existing Rayleigh joint model through the adapter."""

    def test_joint_covariance_matches_the_existing_operator(self, joint_detector_config, joint_array) -> None:
        """Entry for entry, including the timing-offset and mixed-quantity witnesses of the fixture."""
        family = RayleighFamily(field=joint_detector_config.field, n_depth=joint_detector_config.n_depth)
        generic = FamilyJointModel(family, joint_detector_config.layout, joint_array, cavity=None)
        expected = joint_covariance(FREQUENCIES, joint_detector_config, joint_array).matrix
        actual = family_joint_covariance(FREQUENCIES, generic).matrix
        assert_covariance_close(actual, expected, 1e-12)

    def test_per_mode_responses_match_mode_for_mode(self, joint_detector_config, joint_array) -> None:
        """Not only the covariance: each azimuth mode's response vector is the same."""
        family = RayleighFamily(field=joint_detector_config.field, n_depth=joint_detector_config.n_depth)
        generic = FamilyJointModel(family, joint_detector_config.layout, joint_array, cavity=None)
        responses, power = family_joint_mode_responses(FREQUENCIES, generic)
        expected = joint_mode_responses(FREQUENCIES, joint_detector_config, joint_array)
        np.testing.assert_allclose(responses, expected, rtol=1e-12, atol=1e-12 * np.max(np.abs(expected[..., 3:])))
        np.testing.assert_allclose(
            responses[..., :3], expected[..., :3], rtol=1e-12, atol=1e-12 * np.max(np.abs(expected[..., :3]))
        )
        assert power[0].sum() == 0.0

    def test_both_newtonian_terms_cannot_be_disabled(self) -> None:
        """A Rayleigh family with neither bulk nor areal term has no Newtonian coupling to speak of."""
        with pytest.raises(ValueError, match="include_volume"):
            dataclasses.replace(rayleigh(), include_volume=False, include_areal=False)


class TestMixture:
    """Independent families add."""

    def test_mixture_covariance_is_the_sum_of_its_components(self) -> None:
        """Covariances of independent fields add (noise counted once, so it is switched off here)."""
        components = (rayleigh(), body(True), point())
        mixed = family_joint_covariance(FREQUENCIES, model(FamilyMixture(components), self_noise=False)).matrix
        summed = sum(family_joint_covariance(FREQUENCIES, model(c, self_noise=False)).matrix for c in components)
        assert_covariance_close(mixed, summed, 1e-12)

    def test_components_must_share_the_background_density(self) -> None:
        """A layered Love medium and a half-space of different density are not one medium."""
        mixture = FamilyMixture((love(), body(True)))
        with pytest.raises(ValueError, match="disagree"):
            mixture.density_at(np.array([[0.0, 0.0, 50.0]]))
        # Below the layer the densities agree, so the same mixture is valid there.
        mixture.density_at(np.array([[0.0, 0.0, 400.0]]))

    def test_empty_mixture_is_rejected(self) -> None:
        """A mixture needs a component."""
        with pytest.raises(ValueError, match="at least one"):
            FamilyMixture(())


class TestPhaseVelocitySpread:
    """The lateral-inhomogeneity proxy."""

    def test_spread_preserves_surface_vertical_power_and_decorrelates_distant_sensors(self) -> None:
        """Steering is normalized to the surface vertical amplitude, so that power is unchanged; coherence drops."""
        base = rayleigh()
        spread = FamilyMixture(phase_velocity_spread(base, [0.8, 0.9, 1.0], [1.0, 2.0, 1.0]))
        positions = np.array([[0.0, 0.0, 0.0], [1500.0, 0.0, 0.0]])
        frequency = 3.0

        def covariance(family) -> np.ndarray:
            u = family.displacement(positions, frequency)[:, :, 2]
            return np.einsum("mi,mj,m->ij", u, u.conj(), family.mode_power(frequency))

        original, spread_covariance = covariance(base), covariance(spread)
        assert spread_covariance[0, 0].real == pytest.approx(original[0, 0].real, rel=1e-12)
        assert not np.isclose(spread_covariance[0, 1], original[0, 1], rtol=1e-2)

    def test_rejects_malformed_spreads(self) -> None:
        """Mismatched, empty, non-positive or all-zero inputs are refused."""
        for factors, weights in [([], []), ([1.0], [1.0, 2.0]), ([0.0], [1.0]), ([1.0], [0.0])]:
            with pytest.raises(ValueError, match="velocity"):
                phase_velocity_spread(rayleigh(), factors, weights)


class TestJointModelValidation:
    """The generic consumer's guards."""

    def test_cavity_deeper_than_a_test_mass_is_rejected(self) -> None:
        """The cavity must lie in the medium below every test mass's depth."""
        joint = model(body(False), cavity=400.0)
        with pytest.raises(ValueError, match="deeper"):
            family_joint_covariance(np.array([2.0]), joint)

    def test_overlapping_channel_names_are_rejected(self) -> None:
        """Strain and witness names must be disjoint."""
        joint = model(body(False))
        clash = dataclasses.replace(joint.array.sensors[0], name=joint.layout.geometry.names[0])
        array = dataclasses.replace(joint.array, sensors=(clash, *joint.array.sensors[1:]))
        with pytest.raises(ValueError, match="disjoint"):
            family_joint_covariance(np.array([2.0]), dataclasses.replace(joint, array=array))

    def test_mismatched_precomputed_responses_are_rejected(self) -> None:
        """Responses for another grid cannot be paired with this one."""
        joint = model(body(False))
        responses = family_joint_mode_responses(np.array([1.0, 2.0]), joint)
        with pytest.raises(ValueError, match="shape"):
            family_joint_covariance(np.array([2.0]), joint, responses=responses)


class TestNoSiteDefaults:
    """Every site parameter is configurable, listed as UNANCHORED, and has no default."""

    @pytest.mark.parametrize("family_type", [BodyWaveFamily, LoveWaveFamily, PointSourceFamily, RayleighFamily])
    def test_unanchored_parameters_are_real_fields_without_defaults(self, family_type) -> None:
        """A listed parameter must exist and must be required: no guessed site value can hide in a default."""
        fields = {field.name: field for field in dataclasses.fields(family_type)}
        assert family_type.UNANCHORED_PARAMETERS
        for name in family_type.UNANCHORED_PARAMETERS:
            assert name in fields, name
            assert fields[name].default is dataclasses.MISSING, name
            assert fields[name].default_factory is dataclasses.MISSING, name

    @pytest.mark.parametrize("site_type", [LayerOverHalfSpace, PointSource])
    def test_site_value_types_have_no_defaults_at_all(self, site_type) -> None:
        """The layered medium and point sources are pure site description: every field is required."""
        for field in dataclasses.fields(site_type):
            assert field.default is dataclasses.MISSING, field.name

    def test_cavity_and_body_geometry_must_be_stated(self) -> None:
        """Whether there is a cavity, and whether there is a free surface, have no default."""
        cavity = {field.name: field for field in dataclasses.fields(FamilyJointModel)}["cavity"]
        assert cavity.default is dataclasses.MISSING
        radius = {field.name: field for field in dataclasses.fields(CavityModel)}["radius"]
        assert radius.default is dataclasses.MISSING
        free_surface = {field.name: field for field in dataclasses.fields(BodyWaveFamily)}["free_surface"]
        assert free_surface.default is dataclasses.MISSING

    def test_docstrings_mark_every_unanchored_parameter(self) -> None:
        """Each family's documentation says UNANCHORED, so the status is visible where it is configured."""
        import gwmock_newtonian.families.body as body_module
        import gwmock_newtonian.families.love as love_module
        import gwmock_newtonian.families.point as point_module
        import gwmock_newtonian.families.response as response_module

        for module in (body_module, love_module, point_module, response_module):
            assert "UNANCHORED" in (module.__doc__ or ""), module.__name__
