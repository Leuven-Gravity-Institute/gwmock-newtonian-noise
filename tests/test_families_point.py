"""Point-source family: normalization, near/far field and Gauss's law."""

from __future__ import annotations

import numpy as np
import pytest
from _family_fixtures import MEDIUM, flat, point

from gwmock_newtonian_noise.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian_noise.families import PointSource, PointSourceFamily, point_sources

SOURCE = np.array([400.0, 250.0, 450.0])
FREQUENCY = 3.0
K = 2.0 * np.pi * FREQUENCY / MEDIUM.p_wave_speed


def _sphere(radius: float, n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return unit vectors and solid-angle weights of a Gauss-Legendre x uniform sphere rule."""
    mu, w_mu = np.polynomial.legendre.leggauss(n)
    phi = (np.arange(n) + 0.5) * 2.0 * np.pi / n
    sin_t = np.sqrt(1.0 - mu**2)
    unit = np.stack(
        [sin_t[:, None] * np.cos(phi)[None, :], sin_t[:, None] * np.sin(phi)[None, :], np.repeat(mu[:, None], n, 1)],
        axis=-1,
    ).reshape(-1, 3)
    return unit, np.repeat(w_mu, n) * (2.0 * np.pi / n) * radius**2


class TestField:
    """Kinematics of the spherical P wave."""

    def test_reference_distance_amplitude_is_the_mode_amplitude(self) -> None:
        """The radial displacement at ``R_ref`` is exactly the unit reference amplitude, in every direction."""
        family = point()
        directions = np.array([[1.0, 0.0, 0.0], [0.0, -0.6, 0.8], [-0.36, 0.48, -0.8]])
        positions = SOURCE[None, :] + 100.0 * directions
        u = family.displacement(positions, FREQUENCY)[0]
        np.testing.assert_allclose(np.einsum("pi,pi->p", u, directions), 1.0, rtol=1e-13)
        np.testing.assert_allclose(np.cross(u, directions), 0.0, atol=1e-14)

    def test_far_field_decays_as_one_over_distance_and_near_field_as_one_over_distance_squared(self) -> None:
        """``|u| = k / R`` (far, up to ``1/(k R)^2``) and ``|ik/R - 1/R^2|`` (near), over ``|g(R_ref)|``."""
        family = point()
        direction = np.array([0.0, 0.0, -1.0])
        far = family.displacement((SOURCE + 2.0e5 * direction)[None, :], FREQUENCY)[0, 0]
        near = family.displacement((SOURCE + 2.0 * direction)[None, :], FREQUENCY)[0, 0]
        reference = abs(1j * K / 100.0 - 1.0 / 100.0**2)
        assert np.linalg.norm(far) == pytest.approx(K / 2.0e5 / reference, rel=1e-5)  # kR = 650
        assert np.linalg.norm(near) == pytest.approx(abs(1j * K / 2.0 - 1.0 / 4.0) / reference, rel=1e-12)

    def test_positions_too_close_to_a_source_are_rejected(self) -> None:
        """The near field diverges; evaluation inside ``min_distance`` is refused."""
        with pytest.raises(ValueError, match="min_distance"):
            point().displacement((SOURCE + np.array([0.0, 0.0, 0.5]))[None, :], FREQUENCY)

    def test_rejects_malformed_sources(self) -> None:
        """Empty families, bad positions and non-positive reference distances are refused."""
        with pytest.raises(ValueError, match="at least one"):
            PointSourceFamily(medium=MEDIUM, sources=())
        with pytest.raises(ValueError, match="reference_distance"):
            PointSource((0.0, 0.0, 0.0), flat(1.0), 0.0)
        with pytest.raises(ValueError, match="position"):
            PointSource((0.0, 0.0), flat(1.0), 1.0)
        family = point_sources(MEDIUM, [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)], flat(2.0), 50.0)
        np.testing.assert_array_equal(family.mode_power(1.0), [2.0, 2.0])


class TestGaussLaw:
    """The surface integral of ``a . dA`` equals ``-4 pi G M_enclosed`` with the enclosed mass integrated independently."""

    @pytest.mark.parametrize("radius", [5.0, 300.0, 1700.0])
    def test_flux_of_the_acceleration_matches_the_enclosed_mass(self, radius: float) -> None:
        """The enclosed mass is the smooth ``rho0 k^2 psi`` plus the source-point volume injection.

        With ``u = grad(psi)``, ``psi = S e^{ikR}/R``, ``delta_rho = -rho0 lap(psi) =
        rho0 k^2 psi + 4 pi rho0 S delta``. The smooth part is integrated by radial
        quadrature here, independently of the family's gradient formula; the flux
        of the family's acceleration through the sphere must equal ``-4 pi G M``.
        This checks the ``4 pi G rho0`` coupling, its sign and the near-field term.
        """
        family = point()
        unit, weights = _sphere(radius)
        a = family.acceleration(SOURCE[None, :] + radius * unit, FREQUENCY)[0]
        flux = np.sum(weights * np.einsum("pi,pi->p", a, unit))

        strength = 1.0 / (np.exp(1j * K * 100.0) * (1j * K / 100.0 - 1.0 / 100.0**2))  # S for unit amplitude
        nodes, w = np.polynomial.legendre.leggauss(400)
        r = 0.5 * radius * (nodes + 1.0)
        smooth = np.sum(
            0.5 * radius * w * MEDIUM.density * K**2 * strength * np.exp(1j * K * r) / r * 4.0 * np.pi * r**2
        )
        enclosed = smooth + 4.0 * np.pi * MEDIUM.density * strength
        np.testing.assert_allclose(flux, -4.0 * np.pi * GRAVITATIONAL_CONSTANT * enclosed, rtol=1e-9)
