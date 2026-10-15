"""Independent checks of the plane-wave Newtonian kernels and the cavity term.

The production kernels are closed forms. Each is checked here against a route
that does not share its algebra: a brute-force real-space Green's-function sum
for the full-space relation and the cavity term, the package's existing
numerical depth quadrature for the half-space closed form, and the full-space
relation for the deep limit of the half space.
"""

from __future__ import annotations

import numpy as np
import pytest
from _family_fixtures import MEDIUM

from gwmock_newtonian.detector.newtonian import GRAVITATIONAL_CONSTANT, RayleighMode, plane_wave_acceleration
from gwmock_newtonian.families import CavityModel, free_surface_components
from gwmock_newtonian.families.gravity import PlaneWaveComponent, full_space_acceleration, half_space_acceleration
from gwmock_newtonian.families.response import CAVITY_WALL_COEFFICIENT

G = GRAVITATIONAL_CONSTANT
RHO = MEDIUM.density


def _plane_component(kind: str, wavenumber: float, direction: np.ndarray) -> PlaneWaveComponent:
    direction = direction / np.linalg.norm(direction)
    transverse = np.cross([0.0, 0.0, 1.0], direction)
    transverse /= np.linalg.norm(transverse)
    polarization = direction if kind == "P" else transverse
    return PlaneWaveComponent((wavenumber * direction).astype(complex), polarization.astype(complex), 0.7 - 0.2j)


def _real_space_acceleration(
    component: PlaneWaveComponent, centre: np.ndarray, inner: float, outer: float
) -> np.ndarray:
    """Return ``G sum delta_rho (r - r0) / |r - r0|^3`` over the shell ``inner < |r - r0| < outer``.

    Brute-force tensor quadrature: Gauss-Legendre in radius and ``cos(theta)``,
    uniform in azimuth. Shares nothing with the closed forms except the source
    density ``delta_rho = -i rho0 (K . e) c exp(i K . r)``.
    """
    n_radius, n_cos, n_phi = 400, 64, 64
    x, wx = np.polynomial.legendre.leggauss(n_radius)
    radius = inner + 0.5 * (outer - inner) * (x + 1.0)
    w_radius = 0.5 * (outer - inner) * wx
    mu, w_mu = np.polynomial.legendre.leggauss(n_cos)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    sin_t = np.sqrt(1.0 - mu**2)
    unit = np.stack(
        [
            sin_t[:, None] * np.cos(phi)[None, :],
            sin_t[:, None] * np.sin(phi)[None, :],
            np.repeat(mu[:, None], n_phi, 1),
        ],
        axis=-1,
    ).reshape(-1, 3)
    w_angle = np.repeat(w_mu, n_phi) * (2.0 * np.pi / n_phi)
    source = -1j * RHO * component.divergence_factor * component.amplitude
    projection = unit @ component.wavevector  # (n_dir,)
    total = np.zeros(3, dtype=complex)
    # d^3r = r^2 dr dOmega and (r - r0)/|r - r0|^3 = unit / r^2: the r^2 cancels.
    for r, w in zip(radius, w_radius, strict=True):
        total += w * (np.exp(1j * r * projection) * w_angle) @ unit
    return G * source * np.exp(1j * (centre @ component.wavevector)) * total


def _wall_acceleration(component: PlaneWaveComponent, centre: np.ndarray, radius: float) -> np.ndarray:
    """Return the field at the centre of the areal density ``-rho0 u . r_hat`` on a sphere, by quadrature."""
    mu, w_mu = np.polynomial.legendre.leggauss(64)
    n_phi = 64
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    sin_t = np.sqrt(1.0 - mu**2)
    unit = np.stack(
        [
            sin_t[:, None] * np.cos(phi)[None, :],
            sin_t[:, None] * np.sin(phi)[None, :],
            np.repeat(mu[:, None], n_phi, 1),
        ],
        axis=-1,
    ).reshape(-1, 3)
    w_angle = np.repeat(w_mu, n_phi) * (2.0 * np.pi / n_phi)
    points = centre[None, :] + radius * unit
    displacement = component.amplitude * np.exp(1j * (points @ component.wavevector))[:, None] * component.polarization
    sigma = -RHO * np.einsum("ij,ij->i", displacement, unit)
    # Field of a surface mass element at the centre: G sigma dA unit / radius^2 with dA = radius^2 dOmega.
    return G * (sigma * w_angle) @ unit


class TestFullSpace:
    """The infinite-medium relation ``a = 4 pi G rho0 u_P``."""

    @pytest.mark.parametrize("direction", [np.array([0.3, -0.5, 0.8]), np.array([-1.0, 0.2, -0.1])])
    def test_p_wave_matches_a_real_space_green_sum(self, direction: np.ndarray) -> None:
        """Sign and the 4 pi coefficient, against a brute-force Green's-function sum.

        The sphere is truncated at ``K R = 8 pi``: the radial integral of the
        angular average is ``1 - j0(K R)``, which is exactly 1 at a zero of ``sin``,
        so the truncation itself introduces no error.
        """
        wavenumber = 2.0 * np.pi / 500.0
        component = _plane_component("P", wavenumber, direction)
        centre = np.array([120.0, -40.0, 300.0])
        expected = full_space_acceleration([component], centre[None, :], RHO)[0]
        quadrature = _real_space_acceleration(component, centre, 0.0, 8.0 * np.pi / wavenumber)
        np.testing.assert_allclose(quadrature, expected, rtol=1e-9, atol=0.0)
        # Maintained anchor: the vector relative error is at the round-off floor of the
        # sum, measured 3.3e-14 and 3.5e-14 for the two directions. The bound keeps the
        # quoted "3e-14" order of magnitude true; a per-component check would be looser,
        # because a small component carries a larger relative error (up to 1.1e-13).
        relative_error = np.linalg.norm(quadrature - expected) / np.linalg.norm(expected)
        assert relative_error < 1e-13, relative_error
        displacement = component.amplitude * component.polarization * np.exp(1j * centre @ component.wavevector)
        np.testing.assert_allclose(expected, 4.0 * np.pi * G * RHO * displacement, rtol=1e-12, atol=0.0)

    def test_shear_wave_sources_no_gravity(self) -> None:
        """A transverse component has ``K . e = 0``: no bulk density, no acceleration."""
        component = _plane_component("S", 0.01, np.array([0.2, 0.4, -0.9]))
        assert full_space_acceleration([component], np.array([[0.0, 0.0, 100.0]]), RHO).tolist() == [[0j, 0j, 0j]]


class TestHalfSpace:
    """The closed-form half-space kernel against the package's numerical depth quadrature."""

    @pytest.mark.parametrize(
        ("wave_type", "incidence"),
        [("P", 0.4), ("P", 1.1), ("SV", 0.3), ("SV", 0.9)],  # SV at 0.9 rad is post-critical: evanescent P
    )
    def test_closed_form_matches_numerical_depth_quadrature(self, wave_type: str, incidence: float) -> None:
        """Each reflected/incident component, bulk plus areal, agrees with ``plane_wave_acceleration``."""
        frequency = 4.0
        direction = np.array([np.sin(incidence) * np.cos(0.4), np.sin(incidence) * np.sin(0.4), -np.cos(incidence)])
        components = free_surface_components(MEDIUM, wave_type, direction, frequency)
        positions = np.array([[10.0, -20.0, 150.0], [0.0, 0.0, 30.0], [300.0, 50.0, 0.0]])
        closed = half_space_acceleration(components, positions, RHO)
        numerical = np.zeros_like(closed)
        for component in components:
            horizontal = np.real(component.wavevector[:2])
            k = float(np.hypot(*horizontal))
            mode = RayleighMode(
                frequency, k, float(np.arctan2(horizontal[1], horizontal[0])), 2 * np.pi * frequency / k
            )
            amplitude = -1j * RHO * component.divergence_factor * component.amplitude
            q = complex(component.wavevector[2])
            numerical += plane_wave_acceleration(
                positions,
                mode,
                lambda z, a=amplitude, q=q: a * np.exp(1j * q * z),
                RHO,
                surface_vertical_displacement=component.amplitude * component.polarization[2],
                n_depth=800,
                depth_max=float(positions[:, 2].max()) + 30.0 / k,
            )
        np.testing.assert_allclose(closed, numerical, rtol=1e-8, atol=1e-10 * np.max(np.abs(closed)))

    def test_deep_limit_is_the_full_space_relation(self) -> None:
        """Many horizontal wavelengths below the surface, the surface terms decay away."""
        direction = np.array([np.sin(0.5), 0.0, -np.cos(0.5)])
        components = free_surface_components(MEDIUM, "P", direction, 40.0)
        deep = np.array([[0.0, 0.0, 4000.0]])
        np.testing.assert_allclose(
            half_space_acceleration(components, deep, RHO), full_space_acceleration(components, deep, RHO), rtol=1e-9
        )
        shallow = np.array([[0.0, 0.0, 5.0]])
        assert not np.allclose(
            half_space_acceleration(components, shallow, RHO), full_space_acceleration(components, shallow, RHO)
        )

    def test_rejects_positions_above_the_surface_and_vertical_components(self) -> None:
        """The kernel is defined in the medium and for non-vertical components only."""
        component = _plane_component("P", 0.01, np.array([0.3, 0.0, -0.9]))
        with pytest.raises(ValueError, match="z >= 0"):
            half_space_acceleration([component], np.array([[0.0, 0.0, -1.0]]), RHO)
        vertical = PlaneWaveComponent(np.array([0.0, 0.0, 0.01], dtype=complex), np.array([0, 0, 1], dtype=complex))
        with pytest.raises(ValueError, match="non-vertical"):
            half_space_acceleration([vertical], np.array([[0.0, 0.0, 1.0]]), RHO)


class TestCavity:
    """The leading-order cavity-wall term against an exact real-space evaluation."""

    @pytest.mark.parametrize("kind", ["P", "S"])
    def test_cavity_total_matches_a_real_space_shell_plus_wall_sum(self, kind: str) -> None:
        """Bulk outside the cavity plus the wall layer, summed directly, approaches the model as ``K a -> 0``.

        The model is the full-space relation plus ``-(4 pi/3) G rho0 u``; the
        neglected bulk inside the cavity is ``O((K a)^2)``. The real-space sum
        excludes the ball and adds the wall density ``-rho0 u . r_hat`` exactly.
        This anchors the ``(8 pi/3, -4 pi/3)`` P/S pair, including its sign.
        """
        wavenumber = 2.0 * np.pi / 500.0
        radius = 0.02 / wavenumber
        component = _plane_component(kind, wavenumber, np.array([0.4, 0.3, -0.85]))
        centre = np.array([0.0, 50.0, 300.0])
        displacement = component_displacement_at(component, centre)
        model = (
            full_space_acceleration([component], centre[None, :], RHO)[0]
            + CavityModel(radius).acceleration(displacement[None, None, :], np.array([RHO]), centre[None, :])[0, 0]
        )
        exact = _real_space_acceleration(component, centre, radius, 8.0 * np.pi / wavenumber) + _wall_acceleration(
            component, centre, radius
        )
        # The neglected in-cavity bulk is (K a)^2 / 10 relative: 4e-5 here, measured.
        np.testing.assert_allclose(exact, model, rtol=0.0, atol=1e-4 * np.linalg.norm(model))
        coefficient = 8.0 * np.pi / 3.0 if kind == "P" else -4.0 * np.pi / 3.0
        np.testing.assert_allclose(model, coefficient * G * RHO * displacement, rtol=1e-12, atol=0.0)

    def test_wall_coefficient_is_minus_four_pi_over_three(self) -> None:
        """The constant the response layer applies."""
        assert pytest.approx(-4.0 * np.pi / 3.0, rel=1e-15) == CAVITY_WALL_COEFFICIENT

    def test_cavity_must_lie_inside_the_medium(self) -> None:
        """A cavity reaching the surface plane is rejected, as is a non-positive radius."""
        with pytest.raises(ValueError, match="deeper"):
            CavityModel(20.0).acceleration(np.zeros((1, 1, 3), complex), np.array([RHO]), np.array([[0.0, 0.0, 20.0]]))
        with pytest.raises(ValueError, match="radius"):
            CavityModel(0.0)


def component_displacement_at(component: PlaneWaveComponent, point: np.ndarray) -> np.ndarray:
    """Return one component's displacement at one point."""
    return component.amplitude * component.polarization * np.exp(1j * (point @ component.wavevector))
