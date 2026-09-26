"""Tests for the Rayleigh-mode depth eigenfunctions.

The single-mode/depth-dependence limit's independent analytic control lives
here (:class:`TestIndependentElasticityControl`): it re-derives the traction
at the free surface and the elastodynamic (Navier-Cauchy) equation of motion
directly from raw dilatational/shear potentials and elementary calculus, and
checks the closed-form shapes in :mod:`gwmock_newtonian.seismic.depth`
against that derivation -- not by recomputing the same closed form with the
same algebra.

The same raw-potential field is the independent control for the *production*
displacement depth shapes and the signed bulk-density steering:
:meth:`TestIndependentElasticityControl.test_production_shapes_match_potential_displacements`
compares ``horizontal_shape`` / ``vertical_shape`` (as they enter the
displacement, ``1j * k * horizontal_shape`` and ``-k * vertical_shape``)
against the potential displacements at several depths, and
:meth:`TestIndependentElasticityControl.test_production_density_steering_matches_bulk_mass_conservation`
compares :func:`~gwmock_newtonian.seismic.field.steering_vectors`' density
channel against ``-rho0 * div(u)`` for a unit surface vertical displacement.
Those comparisons are what make the control able to fail on a production
depth-shape or density-sign defect; the local potentials alone cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_newtonian.seismic.depth import depth_rates, dilatation_shape, horizontal_shape, vertical_shape
from gwmock_newtonian.seismic.field import RayleighWaveFieldConfig, steering_vectors
from gwmock_newtonian.seismic.medium import ElasticHalfSpace
from gwmock_newtonian.seismic.spectrum import uniform_directional_density

pytestmark = pytest.mark.unit

#: UNANCHORED: generic crustal rock speeds/density used only to exercise the code, not calibrated
#: to any specific ET candidate site.
_MEDIUM = ElasticHalfSpace(p_wave_speed=5800.0, s_wave_speed=3200.0, density=2700.0)

#: UNANCHORED: an illustrative single operating frequency, not a chosen ET observing band.
_FREQUENCY_HZ = 1.3


def _rates():
    omega = 2.0 * np.pi * _FREQUENCY_HZ
    c = _MEDIUM.rayleigh_phase_velocity
    return depth_rates(_MEDIUM, omega, c), omega, c


def _steering_config() -> RayleighWaveFieldConfig:
    """A minimal config for :func:`steering_vectors`; its PSD/direction are irrelevant to depth shapes."""
    return RayleighWaveFieldConfig(
        medium=_MEDIUM,
        surface_displacement_psd=np.ones_like,
        directional_density=uniform_directional_density,
    )


class TestDepthRates:
    """Validation and dimensional sanity of :func:`depth_rates`."""

    def test_rejects_non_positive_angular_frequency(self) -> None:
        """A non-positive angular frequency is unphysical."""
        with pytest.raises(ValueError, match="angular_frequency"):
            depth_rates(_MEDIUM, 0.0, 1000.0)

    def test_rejects_phase_velocity_outside_evanescent_range(self) -> None:
        """A phase velocity at or above the shear speed is not a surface-guided mode."""
        with pytest.raises(ValueError, match="phase_velocity"):
            depth_rates(_MEDIUM, 10.0, _MEDIUM.s_wave_speed)

    def test_decay_rates_dimensionally_consistent_with_wavenumber(self) -> None:
        """q_alpha and q_beta are k times a dimensionless O(1) factor, both real and positive."""
        rates, _, _ = _rates()
        assert rates.q_alpha == pytest.approx(rates.k * rates.r)
        assert rates.q_beta == pytest.approx(rates.k * rates.s)
        assert 0.0 < rates.r < 1.0
        assert 0.0 < rates.s < 1.0


class TestDepthShapesDecayToZero:
    """Every depth eigenfunction is evanescent: it decays to zero far from the free surface."""

    @pytest.mark.parametrize("shape_fn", [horizontal_shape, vertical_shape, dilatation_shape])
    def test_decays_at_large_depth(self, shape_fn) -> None:
        """A depth far past both decay lengths leaves a negligible shape value."""
        rates, _, _ = _rates()
        large_z = 50.0 / (rates.k * min(rates.r, rates.s))
        assert abs(shape_fn(large_z, rates)) < 1e-6 * abs(shape_fn(0.0, rates) or 1.0)


class TestIndependentElasticityControl:
    """Independent control: the traction-free boundary condition and Navier's equation.

    Re-derives the displacement field from raw potentials phi = A*exp(-q_alpha*z)*P,
    psi = B*exp(-q_beta*z)*P (P = exp(i*(k*x - omega*t))) with exact elementary-calculus
    derivatives -- not finite differences, not a re-typing of
    :mod:`gwmock_newtonian.seismic.depth`'s combined closed forms -- and checks that
    the resulting stress field is traction-free at z=0 and that the displacement
    satisfies the elastodynamic Cauchy equation of motion everywhere. The boundary-
    condition coefficient B is the one this package's derivation requires (see the
    :mod:`gwmock_newtonian.seismic.depth` module docstring).

    The same raw-potential field is then the independent control for the production
    outputs: the closed-form ``horizontal_shape`` / ``vertical_shape`` are compared to
    the potential displacement components, and ``steering_vectors``' density channel to
    the bulk mass-conservation ``-rho0 * div(u)``, at several depths. Those comparisons
    are what let the control fail on a production depth-shape or density-sign defect;
    evaluating the local potentials against Hooke's law and Newton's second law alone
    does not constrain the separately implemented closed forms.
    """

    @staticmethod
    def _fields(x0: float, z0: float, t0: float):
        rates, omega, _ = _rates()
        k = rates.k
        q_a = k * rates.r
        q_b = k * rates.s
        s = rates.s
        lam = _MEDIUM.lame_lambda
        mu = _MEDIUM.lame_mu
        rho = _MEDIUM.density
        alpha = _MEDIUM.p_wave_speed
        beta = _MEDIUM.s_wave_speed

        a_amp = 1.0
        b_amp = -1j * (1.0 + s**2) / (2.0 * s)

        p = np.exp(1j * (k * x0 - omega * t0))
        ea = np.exp(-q_a * z0)
        eb = np.exp(-q_b * z0)

        ux = (1j * k * a_amp * ea + q_b * b_amp * eb) * p
        uz = (-q_a * a_amp * ea + 1j * k * b_amp * eb) * p

        dux_dx = 1j * k * ux
        duz_dx = 1j * k * uz
        dux_dz = (1j * k * a_amp * (-q_a) * ea + q_b * b_amp * (-q_b) * eb) * p
        duz_dz = (-q_a * a_amp * (-q_a) * ea + 1j * k * b_amp * (-q_b) * eb) * p

        sigma_zz = lam * (dux_dx + duz_dz) + 2.0 * mu * duz_dz
        sigma_xz = mu * (dux_dz + duz_dx)

        d2ux_dt2 = -(omega**2) * ux
        d2uz_dt2 = -(omega**2) * uz

        phi = a_amp * ea * p
        psi = b_amp * eb * p
        div_u = -((omega / alpha) ** 2) * phi
        curl_u = (omega / beta) ** 2 * psi

        ddiv_dx = 1j * k * div_u
        ddiv_dz = -q_a * div_u
        dcurl_dx = 1j * k * curl_u
        dcurl_dz = -q_b * curl_u

        navier_x = rho * d2ux_dt2 - ((lam + 2.0 * mu) * ddiv_dx - mu * (-dcurl_dz))
        navier_z = rho * d2uz_dt2 - ((lam + 2.0 * mu) * ddiv_dz - mu * dcurl_dx)

        return {
            "ux": ux,
            "uz": uz,
            "sigma_zz": sigma_zz,
            "sigma_xz": sigma_xz,
            "navier_x": navier_x,
            "navier_z": navier_z,
            "div_u": div_u,
        }

    def test_free_surface_is_traction_free(self) -> None:
        """sigma_zz and sigma_xz vanish at z=0 for every horizontal position and time."""
        lam = _MEDIUM.lame_lambda
        mu = _MEDIUM.lame_mu
        scale = max(abs(lam), abs(mu))
        for x0, t0 in [(0.0, 0.0), (0.3, 0.1), (-1.7, 0.42)]:
            fields = self._fields(x0, 0.0, t0)
            assert abs(fields["sigma_zz"]) / scale < 1e-9
            assert abs(fields["sigma_xz"]) / scale < 1e-9

    @pytest.mark.parametrize("z0", [0.0, 0.5, 3.0, 20.0])
    def test_navier_equation_holds_at_depth(self, z0: float) -> None:
        """Rho * d2u/dt2 = (lambda+2mu)*grad(div u) - mu*curl(curl u) holds at every depth."""
        _, omega, _ = _rates()
        scale = _MEDIUM.density * omega**2
        fields = self._fields(0.4, z0, 0.2)
        assert abs(fields["navier_x"]) / scale < 1e-9
        assert abs(fields["navier_z"]) / scale < 1e-9

    @pytest.mark.parametrize("z0", [0.0, 0.7, 5.0])
    def test_dilatation_shape_matches_laplacian_of_p_potential(self, z0: float) -> None:
        """div(u) = laplacian(phi) exactly, an identity independent of the boundary condition fixing r, s, B."""
        rates, _, _ = _rates()
        fields = self._fields(0.0, z0, 0.0)
        production = dilatation_shape(z0, rates)
        np.testing.assert_allclose(fields["div_u"], production, rtol=1e-10)

    @pytest.mark.parametrize("z0", [0.0, 0.2, 1.0, 5.0, 20.0])
    def test_production_shapes_match_potential_displacements(self, z0: float) -> None:
        """The production depth shapes equal the raw-potential displacement components at every depth.

        The production docstrings state ``u_along(z) = 1j * k * horizontal_shape(z)`` and
        ``u_z(z) = -k * vertical_shape(z)`` per unit potential amplitude; evaluated at
        ``(x, t) = (0, 0)`` the raw-potential phase is unity and these are directly comparable.
        Without this comparison a defect in either production shape leaves the control green.
        """
        rates, _, _ = _rates()
        k = rates.k
        fields = self._fields(0.0, z0, 0.0)
        np.testing.assert_allclose(1j * k * horizontal_shape(z0, rates), fields["ux"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(-k * vertical_shape(z0, rates), fields["uz"], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("z0", [0.0, 0.2, 1.0, 5.0, 20.0])
    def test_production_density_steering_matches_bulk_mass_conservation(self, z0: float) -> None:
        """The density steering equals ``-rho0 * div(u)`` for a unit surface vertical displacement.

        ``steering_vectors`` normalizes every channel to a unit *surface* vertical displacement,
        so the raw potentials are rescaled by ``1 / u_z(0)`` before evaluating the bulk
        mass-conservation term ``delta_rho = -rho0 * div(u)``. This is the signed density control:
        negating the density channel in ``field.py`` makes it fail.
        """
        surface = self._fields(0.0, 0.0, 0.0)
        unit_scale = 1.0 / surface["uz"]
        fields = self._fields(0.0, z0, 0.0)
        expected = -_MEDIUM.density * fields["div_u"] * unit_scale

        steering = steering_vectors([(0.0, 0.0, z0)], _FREQUENCY_HZ, _steering_config())
        production = steering["density"][0, 0]

        np.testing.assert_allclose(production, expected, rtol=1e-10)
