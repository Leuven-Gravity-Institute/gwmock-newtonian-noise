"""Body-wave family: free-surface reflection anchors, composition, anisotropy and the diffuse proxy."""

from __future__ import annotations

import numpy as np
import pytest
from _family_fixtures import MEDIUM, body, flat, model

from gwmock_newtonian.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian.families import (
    BodyWaveFamily,
    SphereGrid,
    equipartition_shear_to_p_ratio,
    equipartitioned_body_waves,
    family_joint_covariance,
    family_strain_responses,
    free_surface_components,
    uniform_sphere_density,
    von_mises_fisher_density,
)
from gwmock_newtonian.families.gravity import component_displacement
from gwmock_newtonian.seismic.medium import rayleigh_velocity_ratio

ALPHA, BETA = MEDIUM.p_wave_speed, MEDIUM.s_wave_speed
#: P critical angle of SV incidence, ``asin(beta / alpha)`` = 0.584 rad for the fixture medium.
SV_CRITICAL = float(np.arcsin(BETA / ALPHA))


def _direction(incidence: float, azimuth: float = 0.6) -> np.ndarray:
    return np.array([np.sin(incidence) * np.cos(azimuth), np.sin(incidence) * np.sin(azimuth), -np.cos(incidence)])


def _amplitudes(wave_type: str, incidence: float, frequency: float = 2.0) -> tuple[complex, complex, complex]:
    """Return the reflected (P, SV, SH) amplitudes."""
    _, p, sv, sh = free_surface_components(MEDIUM, wave_type, _direction(incidence), frequency)
    return p.amplitude, sv.amplitude, sh.amplitude


class TestFreeSurfaceReflection:
    """The free-surface solution against convention-free and textbook anchors."""

    @pytest.mark.parametrize(
        ("wave_type", "incidence"), [("P", 0.2), ("P", 1.2), ("SV", 0.3), ("SV", 1.0), ("SH", 0.7)]
    )
    def test_surface_is_traction_free_by_numerical_differentiation(self, wave_type: str, incidence: float) -> None:
        """``sigma_xz = sigma_yz = sigma_zz = 0`` at ``z = 0`` from finite differences of the total displacement and Hooke's law.

        Independent of the algebraic traction the solver uses: a polarization or
        sign slip in either the solve or the components shows up here.
        """
        frequency = 2.0
        components = free_surface_components(MEDIUM, wave_type, _direction(incidence), frequency)
        origin = np.array([13.0, -7.0, 0.0])
        step = 0.05

        def gradient() -> np.ndarray:
            grad = np.zeros((3, 3), dtype=complex)  # grad[j, i] = d u_i / d x_j
            for j in range(3):
                offset = np.zeros(3)
                offset[j] = step
                plus = component_displacement(components, (origin + offset)[None, :])[0]
                minus = component_displacement(components, (origin - offset)[None, :])[0]
                grad[j] = (plus - minus) / (2.0 * step)
            return grad

        grad = gradient()
        divergence = np.trace(grad)
        traction = MEDIUM.lame_mu * (grad[:, 2] + grad[2, :])
        traction[2] += MEDIUM.lame_lambda * divergence
        incident_scale = MEDIUM.lame_mu * 2.0 * np.pi * frequency / BETA
        assert np.max(np.abs(traction)) < 1e-6 * incident_scale

    @pytest.mark.parametrize("incidence", [0.05, 0.4, 0.9, 1.4])
    def test_p_incidence_matches_the_closed_form_pp_coefficient(self, incidence: float) -> None:
        """The textbook free-surface PP coefficient (Aki & Richards 2002, Chap. 5), same polarization convention."""
        p = np.sin(incidence) / ALPHA
        cos_i, cos_j = np.cos(incidence), np.sqrt(1.0 - (p * BETA) ** 2)
        a = (1.0 / BETA**2 - 2.0 * p**2) ** 2
        b = 4.0 * p**2 * (cos_i / ALPHA) * (cos_j / BETA)
        reflected_p, _, reflected_sh = _amplitudes("P", incidence)
        assert reflected_p == pytest.approx((-a + b) / (a + b), abs=1e-10)
        assert abs(reflected_sh) < 1e-12

    @pytest.mark.parametrize("incidence", [0.1, 0.5, 1.0, 1.45])
    def test_p_incidence_conserves_vertical_energy_flux(self, incidence: float) -> None:
        """``|PP|^2 + (beta cos j / alpha cos i) |PS|^2 = 1``: convention-free."""
        reflected_p, reflected_sv, _ = _amplitudes("P", incidence)
        cos_j = np.sqrt(1.0 - (np.sin(incidence) * BETA / ALPHA) ** 2)
        balance = abs(reflected_p) ** 2 + BETA * cos_j / (ALPHA * np.cos(incidence)) * abs(reflected_sv) ** 2
        assert balance == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("incidence", [0.1, 0.4, SV_CRITICAL - 0.05])
    def test_precritical_sv_incidence_conserves_vertical_energy_flux(self, incidence: float) -> None:
        """``|SS|^2 + (alpha cos i_p / beta cos j) |SP|^2 = 1`` below the critical angle."""
        reflected_p, reflected_sv, _ = _amplitudes("SV", incidence)
        cos_p = np.sqrt(1.0 - (np.sin(incidence) * ALPHA / BETA) ** 2)
        balance = abs(reflected_sv) ** 2 + ALPHA * cos_p / (BETA * np.cos(incidence)) * abs(reflected_p) ** 2
        assert balance == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("incidence", [SV_CRITICAL + 0.05, 1.0, 1.4])
    def test_postcritical_sv_is_totally_reflected_with_an_evanescent_p(self, incidence: float) -> None:
        """Past the critical angle the reflected P carries no flux: ``|SS| = 1`` and ``Im K_z > 0``."""
        components = free_surface_components(MEDIUM, "SV", _direction(incidence), 2.0)
        assert abs(components[2].amplitude) == pytest.approx(1.0, abs=1e-10)
        assert components[1].wavevector[2].real == 0.0
        assert components[1].wavevector[2].imag > 0.0

    @pytest.mark.parametrize("incidence", [0.2, 0.8, 1.3])
    def test_sh_is_reflected_with_unit_coefficient_and_doubles_at_the_surface(self, incidence: float) -> None:
        """SH decouples: reflection +1, surface displacement twice the incident, purely horizontal."""
        components = free_surface_components(MEDIUM, "SH", _direction(incidence), 2.0)
        reflected_p, reflected_sv, reflected_sh = (c.amplitude for c in components[1:])
        assert reflected_sh == pytest.approx(1.0, abs=1e-12)
        assert abs(reflected_p) < 1e-12
        assert abs(reflected_sv) < 1e-12
        surface = component_displacement(components, np.zeros((1, 3)))[0]
        assert np.linalg.norm(surface) == pytest.approx(2.0, abs=1e-12)
        assert abs(surface[2]) < 1e-15

    def test_near_normal_p_incidence_doubles_the_vertical_surface_displacement(self) -> None:
        """The free-surface amplification of a vertical P wave is 2 (PP -> -1 at normal incidence)."""
        components = free_surface_components(MEDIUM, "P", _direction(1e-4), 2.0)
        surface = component_displacement(components, np.zeros((1, 3)))[0]
        assert abs(surface[2]) == pytest.approx(2.0, abs=1e-6)

    def test_pp_denominator_vanishes_at_the_rayleigh_slowness(self) -> None:
        """The free-surface reflection pole is the package's Rayleigh root: an independent dispersion check."""
        c_rayleigh = np.sqrt(rayleigh_velocity_ratio(BETA / ALPHA)) * BETA
        p = 1.0 / c_rayleigh
        cos_i_over_alpha = np.sqrt(complex(1.0 / ALPHA**2 - p**2))
        cos_j_over_beta = np.sqrt(complex(1.0 / BETA**2 - p**2))
        denominator = (1.0 / BETA**2 - 2.0 * p**2) ** 2 + 4.0 * p**2 * cos_i_over_alpha * cos_j_over_beta
        assert abs(denominator) < 1e-10 * (1.0 / BETA**2) ** 2

    def test_rejects_downgoing_or_vertical_incidence_and_unknown_types(self) -> None:
        """An incident wave must propagate up toward the surface and have a horizontal direction."""
        with pytest.raises(ValueError, match="upward"):
            free_surface_components(MEDIUM, "P", np.array([0.3, 0.0, 0.9]), 2.0)
        with pytest.raises(ValueError, match="vertical"):
            free_surface_components(MEDIUM, "SV", np.array([0.0, 0.0, -1.0]), 2.0)
        with pytest.raises(ValueError, match="wave_type"):
            free_surface_components(MEDIUM, "Love", np.array([0.3, 0.0, -0.9]), 2.0)


class TestComposition:
    """The P/S composition is explicit and each type couples as its physics dictates."""

    def test_each_type_has_its_own_power(self) -> None:
        """The three PSDs are reported separately and the mode powers are type-major."""
        family = body(False, p=1.0, sv=0.5, sh=0.25)
        assert family.type_power(2.0) == {"P": 1.0, "SV": 0.5, "SH": 0.25}
        power = family.mode_power(2.0).reshape(3, -1).sum(axis=1)
        np.testing.assert_allclose(power, [1.0, 0.5, 0.25], rtol=1e-12)

    def test_shear_only_infinite_medium_couples_only_through_the_cavity(self) -> None:
        """S waves source no gravity in an infinite medium; the cavity wall restores the coupling."""
        family = body(False)
        shear = slice(family.grid.size, None)  # the SV and SH modes
        no_cavity = family_strain_responses(model(family, cavity=None), 2.0)[shear]
        with_cavity = family_strain_responses(model(family, cavity=10.0), 2.0)[shear]
        # K . e of a shear component is zero up to rounding of the cross product (~1e-16 relative).
        assert np.max(np.abs(no_cavity)) < 1e-12 * np.max(np.abs(with_cavity))

    def test_p_coupling_with_a_cavity_is_two_thirds_of_the_infinite_medium_value(self) -> None:
        """``(8 pi/3) / (4 pi) = 2/3``: the cavity coefficient pair, through the full strain chain."""
        family = body(False)
        p_modes = slice(0, family.grid.size)
        no_cavity = family_strain_responses(model(family, cavity=None), 2.0)[p_modes]
        with_cavity = family_strain_responses(model(family, cavity=10.0), 2.0)[p_modes]
        np.testing.assert_allclose(with_cavity, (2.0 / 3.0) * no_cavity, rtol=1e-12, atol=0.0)

    def test_sh_has_no_vertical_displacement_in_the_infinite_medium(self) -> None:
        """SH polarization is horizontal by construction."""
        sh_only = body(False)
        positions = np.array([[0.0, 0.0, 100.0], [50.0, 20.0, 300.0]])
        displacement = sh_only.displacement(positions, 2.0)
        n = sh_only.grid.size
        assert np.max(np.abs(displacement[2 * n :, :, 2])) < 1e-15

    def test_composition_changes_the_joint_covariance(self) -> None:
        """P-dominated and S-dominated fields are distinguishable by the joint model."""
        p_heavy = family_joint_covariance(np.array([2.0]), model(body(True, p=1.0, sv=0.1, sh=0.1))).matrix
        s_heavy = family_joint_covariance(np.array([2.0]), model(body(True, p=0.1, sv=1.0, sh=1.0))).matrix
        assert not np.allclose(p_heavy, s_heavy, rtol=1e-3)

    def test_half_space_positions_above_the_surface_are_rejected(self) -> None:
        """With a free surface the medium is ``z >= 0``."""
        with pytest.raises(ValueError, match="z >= 0"):
            body(True).displacement(np.array([[0.0, 0.0, -1.0]]), 2.0)


class TestDirectionality:
    """The direction grid and anisotropic (von Mises--Fisher) densities."""

    def test_grid_cells_are_unit_and_upgoing_when_requested(self) -> None:
        """Directions are unit vectors; the half-space grid covers the upgoing hemisphere only."""
        for upgoing in (False, True):
            directions = SphereGrid(5, 7, upgoing_only=upgoing).directions()
            np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-14)
            if upgoing:
                assert np.all(directions[:, 2] < 0.0)
            else:
                assert np.sum(directions[:, 2] < 0.0) == np.sum(directions[:, 2] > 0.0)

    def test_equal_area_grid_integrates_low_order_moments_exactly(self) -> None:
        """Isotropic weights give ``<n> = 0`` and the midpoint-rule second moments exactly.

        The midpoint rule in ``cos(theta)`` with ``n`` cells gives
        ``<n_z^2> = 1/3 - (2/n)^2 / 12``; the azimuth rule is exact for ``cos^2``.
        """
        n_polar = 12
        grid = SphereGrid(n_polar, 16)
        weights = grid.weights(uniform_sphere_density)
        directions = grid.directions()
        np.testing.assert_allclose(weights @ directions, 0.0, atol=1e-15)
        second = np.einsum("m,mi,mj->ij", weights, directions, directions)
        zz = 1.0 / 3.0 - (2.0 / n_polar) ** 2 / 12.0
        np.testing.assert_allclose(second, np.diag([(1.0 - zz) / 2.0, (1.0 - zz) / 2.0, zz]), atol=1e-14)

    def test_von_mises_fisher_zero_concentration_is_isotropic(self) -> None:
        """``kappa = 0`` reproduces the uniform weights."""
        grid = SphereGrid(4, 8)
        np.testing.assert_allclose(
            grid.weights(von_mises_fisher_density([0.0, 1.0, -1.0], 0.0)), grid.weights(uniform_sphere_density)
        )

    def test_concentrated_field_is_beamed_and_changes_the_covariance(self) -> None:
        """A large ``kappa`` puts the weight near the mean direction and alters the joint covariance."""
        mean = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
        beamed = body(True, density=von_mises_fisher_density(mean, 30.0))
        weights = beamed.grid.weights(beamed.directional_density)
        assert weights @ (beamed.grid.directions() @ mean) > 0.85
        isotropic = family_joint_covariance(np.array([2.0]), model(body(True))).matrix
        anisotropic = family_joint_covariance(np.array([2.0]), model(beamed)).matrix
        assert not np.allclose(isotropic, anisotropic, rtol=1e-3)

    def test_rejects_degenerate_densities_and_grids(self) -> None:
        """Negative, all-zero, wrong-shaped densities and too-coarse grids are refused."""
        grid = SphereGrid(2, 2)
        with pytest.raises(ValueError, match="non-negative"):
            grid.weights(lambda d: -np.ones(d.shape[0]))
        with pytest.raises(ValueError, match="positive somewhere"):
            grid.weights(lambda d: np.zeros(d.shape[0]))
        with pytest.raises(ValueError, match="shape"):
            grid.weights(lambda d: np.ones(3))
        with pytest.raises(ValueError, match="at least"):
            SphereGrid(1, 4)
        with pytest.raises(ValueError, match="kappa"):
            von_mises_fisher_density([0.0, 0.0, 1.0], -1.0)
        with pytest.raises(ValueError, match="mean_direction"):
            von_mises_fisher_density([0.0, 0.0, 0.0], 1.0)


class TestDiffuseProxy:
    """The equipartitioned diffuse body-wave field (Weaver 1982)."""

    def test_energy_ratio_is_two_alpha_over_beta_cubed(self) -> None:
        """``E_S / E_P = 2 (alpha / beta)^3``."""
        assert equipartition_shear_to_p_ratio(MEDIUM) == pytest.approx(2.0 * (ALPHA / BETA) ** 3, rel=1e-15)

    def test_composition_follows_equipartition_and_conserves_total_power(self) -> None:
        """``P : SV : SH = 1 : R/2 : R/2`` and the three sum to the total PSD."""
        family = equipartitioned_body_waves(MEDIUM, flat(2.0))
        powers = family.type_power(3.0)
        ratio = equipartition_shear_to_p_ratio(MEDIUM)
        assert sum(powers.values()) == pytest.approx(2.0, rel=1e-14)
        assert (powers["SV"] + powers["SH"]) / powers["P"] == pytest.approx(ratio, rel=1e-14)
        assert powers["SV"] == powers["SH"]
        assert family.free_surface is False


class TestValidation:
    """PSD and frequency validation."""

    def test_negative_psd_is_rejected(self) -> None:
        """A negative PSD value is not a spectrum."""
        family = BodyWaveFamily(
            medium=MEDIUM,
            p_psd=flat(-1.0),
            sv_psd=flat(1.0),
            sh_psd=flat(1.0),
            directional_density=uniform_sphere_density,
            free_surface=False,
        )
        with pytest.raises(ValueError, match="p_psd"):
            family.mode_power(1.0)

    def test_gravitational_constant_is_the_package_constant(self) -> None:
        """The P coupling uses the package's CODATA G: ``|a| = 4 pi G rho0 |u|`` for a single P mode."""
        family = body(False, p=1.0, sv=0.0, sh=0.0)
        position = np.array([[0.0, 0.0, 200.0]])
        a = family.acceleration(position, 2.0)[0, 0]
        u = family.displacement(position, 2.0)[0, 0]
        np.testing.assert_allclose(a, 4.0 * np.pi * GRAVITATIONAL_CONSTANT * MEDIUM.density * u, rtol=1e-13)
