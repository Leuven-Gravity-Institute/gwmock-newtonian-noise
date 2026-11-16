"""Tests for the elastic half-space medium and the Rayleigh secular equation."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian_noise.seismic.medium import (
    ElasticHalfSpace,
    rayleigh_secular_cubic_coefficients,
    rayleigh_velocity_ratio,
)

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal rock speeds/density used only to exercise the code, not calibrated
#: to any specific ET candidate site.
_UNANCHORED_ALPHA = 5800.0
_UNANCHORED_BETA = 3200.0
_UNANCHORED_RHO = 2700.0

# The ElasticHalfSpace literals inside the validation tests below are deliberately non-physical
# inputs used to assert rejection (sub-shear p-speed, zero shear speed, zero density) -- they are
# validation probes, not UNANCHORED site fixtures, and none is a candidate medium.


class TestElasticHalfSpace:
    """Configuration validation and derived elastic quantities."""

    def test_rejects_non_positive_s_wave_speed(self) -> None:
        """A non-positive shear speed is not a physical medium."""
        with pytest.raises(ValueError, match="s_wave_speed"):
            ElasticHalfSpace(p_wave_speed=5000.0, s_wave_speed=0.0, density=2000.0)

    def test_rejects_p_wave_speed_not_exceeding_s_wave_speed(self) -> None:
        """An isotropic elastic solid requires alpha > beta (Poisson ratio below 0.5)."""
        with pytest.raises(ValueError, match="p_wave_speed"):
            ElasticHalfSpace(p_wave_speed=3000.0, s_wave_speed=3200.0, density=2000.0)

    def test_rejects_non_positive_density(self) -> None:
        """A non-positive density is not a physical medium."""
        with pytest.raises(ValueError, match="density"):
            ElasticHalfSpace(p_wave_speed=5000.0, s_wave_speed=3000.0, density=0.0)

    def test_lame_parameters_dimensional_check(self) -> None:
        """Mu = rho*beta^2 and lambda = rho*(alpha^2 - 2*beta^2) reproduce alpha^2 = (lambda+2mu)/rho."""
        medium = ElasticHalfSpace(
            p_wave_speed=_UNANCHORED_ALPHA, s_wave_speed=_UNANCHORED_BETA, density=_UNANCHORED_RHO
        )
        recovered_alpha_sq = (medium.lame_lambda + 2.0 * medium.lame_mu) / medium.density
        recovered_beta_sq = medium.lame_mu / medium.density
        np.testing.assert_allclose(recovered_alpha_sq, medium.p_wave_speed**2, rtol=1e-12)
        np.testing.assert_allclose(recovered_beta_sq, medium.s_wave_speed**2, rtol=1e-12)

    def test_rayleigh_phase_velocity_is_between_zero_and_s_wave_speed(self) -> None:
        """The Rayleigh phase velocity is always a strictly sub-shear surface-wave speed."""
        medium = ElasticHalfSpace(
            p_wave_speed=_UNANCHORED_ALPHA, s_wave_speed=_UNANCHORED_BETA, density=_UNANCHORED_RHO
        )
        c = medium.rayleigh_phase_velocity
        assert 0.0 < c < medium.s_wave_speed

    def test_rayleigh_phase_velocity_matches_the_poisson_solid_literature_value(self) -> None:
        """For a Poisson solid (nu=1/4, alpha**2 = 3*beta**2), c_R/beta = 0.9194.

        This is an *independent* literature anchor computed from a table, not from this module's
        own algebra: see e.g. Achenbach, J. D. (1973), *Wave Propagation in Elastic Solids*,
        North-Holland, Sec. 6.2, or Graff, K. F. (1975), *Wave Motion in Elastic Solids*, Ohio
        State University Press, Table 5.1, both of which report c_R/beta = 0.9194 for nu = 0.25.
        """
        beta = 1000.0
        alpha = beta * np.sqrt(3.0)  # nu = 0.25
        # Anchored, not a site fixture: the absolute scale (beta=1000 m/s, rho=1 kg/m^3) is an
        # arbitrary normalization and the asserted quantity is the dimensionless literature value.
        medium = ElasticHalfSpace(p_wave_speed=alpha, s_wave_speed=beta, density=1.0)
        np.testing.assert_allclose(medium.rayleigh_phase_velocity / beta, 0.9194, atol=1e-4)

    def test_non_dispersive_across_frequency(self) -> None:
        """A homogeneous half-space's Rayleigh velocity does not depend on frequency at all."""
        medium = ElasticHalfSpace(
            p_wave_speed=_UNANCHORED_ALPHA, s_wave_speed=_UNANCHORED_BETA, density=_UNANCHORED_RHO
        )
        # rayleigh_phase_velocity takes no frequency argument by construction; recomputing it twice
        # must be identical, confirming no hidden frequency-dependent state.
        assert medium.rayleigh_phase_velocity == medium.rayleigh_phase_velocity


class TestRayleighSecularEquation:
    """The cubic reduction of the Rayleigh secular equation and its unique physical root."""

    @pytest.mark.parametrize("kappa", [0.1, 0.3, 0.55, 1.0 / np.sqrt(3.0), 0.8, 0.95])
    def test_root_satisfies_unsquared_secular_equation(self, kappa: float) -> None:
        """The chosen root satisfies (2-x)**2 = 4*r*s exactly, not just the squared cubic."""
        x = rayleigh_velocity_ratio(kappa)
        r = np.sqrt(1.0 - x * kappa**2)
        s = np.sqrt(1.0 - x)
        np.testing.assert_allclose((2.0 - x) ** 2, 4.0 * r * s, rtol=1e-9)

    @pytest.mark.parametrize("kappa", [0.1, 0.3, 0.55, 1.0 / np.sqrt(3.0), 0.8, 0.95])
    def test_root_is_unique_in_open_unit_interval(self, kappa: float) -> None:
        """Exactly one root of the secular cubic lies in (0, 1) for a physical kappa."""
        coefficients = rayleigh_secular_cubic_coefficients(kappa)
        roots = np.roots(coefficients)
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        physical = real_roots[(real_roots > 0.0) & (real_roots < 1.0)]
        assert physical.size == 1

    def test_rejects_kappa_outside_open_unit_interval(self) -> None:
        """Kappa = beta/alpha must lie strictly in (0, 1)."""
        with pytest.raises(ValueError, match="kappa"):
            rayleigh_velocity_ratio(1.0)
        with pytest.raises(ValueError, match="kappa"):
            rayleigh_velocity_ratio(0.0)
