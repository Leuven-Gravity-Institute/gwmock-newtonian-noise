"""Love-wave family: dispersion, eigenfunction physics and Newtonian coupling."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _family_fixtures import LAYER, love, model

from gwmock_newtonian.families import family_joint_covariance, family_strain_responses, love_phase_velocity
from gwmock_newtonian.families.love import LayerOverHalfSpace, love_dispersion_residual

BETA1, BETA2 = LAYER.layer_s_wave_speed, LAYER.half_space_s_wave_speed
H = LAYER.layer_thickness


class TestDispersion:
    """The fundamental-mode root of Love's relation."""

    @pytest.mark.parametrize("frequency", [0.05, 0.5, 2.0, 8.0, 30.0])
    def test_root_satisfies_the_relation_and_lies_between_the_shear_speeds(self, frequency: float) -> None:
        """The residual vanishes and ``beta1 < c < beta2`` with ``k H s1 < pi / 2`` (fundamental mode)."""
        c = love_phase_velocity(LAYER, frequency)
        assert BETA1 < c < BETA2
        scale = LAYER.half_space_mu
        assert abs(love_dispersion_residual(LAYER, frequency, c)) < 1e-9 * scale
        k = 2.0 * np.pi * frequency / c
        assert k * H * np.sqrt(c**2 / BETA1**2 - 1.0) < 0.5 * np.pi

    def test_limits_approach_the_half_space_and_layer_shear_speeds(self) -> None:
        """Long waves see the half-space (``c -> beta2``); short waves the layer (``c -> beta1``)."""
        assert love_phase_velocity(LAYER, 1e-3) == pytest.approx(BETA2, rel=1e-4)
        assert love_phase_velocity(LAYER, 200.0) == pytest.approx(BETA1, rel=1e-3)

    def test_phase_velocity_decreases_with_frequency(self) -> None:
        """Normal dispersion of the fundamental mode."""
        velocities = [love_phase_velocity(LAYER, f) for f in np.geomspace(0.05, 30.0, 12)]
        assert np.all(np.diff(velocities) < 0.0)

    def test_layer_must_be_slower_than_the_half_space(self) -> None:
        """A fast-over-slow structure guides no Love wave."""
        with pytest.raises(ValueError, match="slower layer"):
            dataclasses.replace(LAYER, layer_s_wave_speed=BETA2 + 1.0)
        with pytest.raises(ValueError, match="layer_thickness"):
            dataclasses.replace(LAYER, layer_thickness=0.0)


class TestEigenfunction:
    """Boundary conditions and the equation of motion, by numerical differentiation."""

    FREQUENCY = 2.0

    def _shape(self, z: np.ndarray) -> np.ndarray:
        return love().depth_shape(np.asarray(z, dtype=float), self.FREQUENCY)

    def test_free_surface_is_traction_free(self) -> None:
        """``mu dw/dz = 0`` at ``z = 0`` (one-sided difference)."""
        step = 1e-3
        derivative = (-3.0 * self._shape([0.0])[0] + 4.0 * self._shape([step])[0] - self._shape([2 * step])[0]) / (
            2 * step
        )
        k = 2.0 * np.pi * self.FREQUENCY / love_phase_velocity(LAYER, self.FREQUENCY)
        assert abs(derivative) < 1e-6 * k

    def test_displacement_and_traction_are_continuous_at_the_interface(self) -> None:
        """``w`` and ``mu dw/dz`` match across ``z = H`` from each side."""
        step = 1e-3
        above = self._shape([H - 2 * step, H - step, H - 1e-9])
        below = self._shape([H, H + step, H + 2 * step])
        assert above[2] == pytest.approx(below[0], rel=1e-8)
        slope_above = (3 * above[2] - 4 * above[1] + above[0]) / (2 * step)
        slope_below = (-3 * below[0] + 4 * below[1] - below[2]) / (2 * step)
        assert LAYER.layer_mu * slope_above == pytest.approx(LAYER.half_space_mu * slope_below, rel=1e-5)

    @pytest.mark.parametrize(("depth", "mu", "rho"), [(80.0, None, None), (450.0, None, None)])
    def test_equation_of_motion_holds_in_each_region(self, depth: float, mu, rho) -> None:
        """``mu (w'' - k^2 w) = -rho omega^2 w`` for SH motion in each homogeneous region."""
        in_layer = depth < H
        mu = LAYER.layer_mu if in_layer else LAYER.half_space_mu
        rho = LAYER.layer_density if in_layer else LAYER.half_space_density
        omega = 2.0 * np.pi * self.FREQUENCY
        k = omega / love_phase_velocity(LAYER, self.FREQUENCY)
        step = 0.5
        w = self._shape([depth - step, depth, depth + step])
        second = (w[0] - 2 * w[1] + w[2]) / step**2
        assert mu * (second - k**2 * w[1]) == pytest.approx(-rho * omega**2 * w[1], rel=1e-5)

    def test_surface_normalization_and_decay(self) -> None:
        """``w(0) = 1`` and the half-space tail decays."""
        shape = self._shape([0.0, H + 1000.0, H + 5000.0])
        assert shape[0] == 1.0
        assert abs(shape[2]) < abs(shape[1])


class TestNewtonianCoupling:
    """Love waves are seen by witnesses and reach the strain only through the cavity."""

    def test_displacement_is_horizontal_transverse_and_divergence_free(self) -> None:
        """No vertical motion, no motion along propagation, and ``div(u) = 0`` numerically."""
        family = love()
        frequency = 2.0
        point = np.array([30.0, -20.0, 120.0])
        u = family.displacement(point[None, :], frequency)[:, 0, :]
        assert np.max(np.abs(u[:, 2])) == 0.0
        azimuths = family.azimuth_grid()
        along = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)], axis=-1)
        assert np.max(np.abs(np.einsum("mi,mi->m", u, along))) < 1e-15
        step = 0.01
        divergence = np.zeros(family.n_modes, dtype=complex)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            plus = family.displacement((point + offset)[None, :], frequency)[:, 0, axis]
            minus = family.displacement((point - offset)[None, :], frequency)[:, 0, axis]
            divergence += (plus - minus) / (2 * step)
        k = 2.0 * np.pi * frequency / love_phase_velocity(LAYER, frequency)
        assert np.max(np.abs(divergence)) < 1e-8 * k

    def test_field_intrinsic_gravity_is_zero_and_the_cavity_restores_coupling(self) -> None:
        """No bulk/areal source, so the strain is zero without a cavity and non-zero with one."""
        family = love()
        assert not np.any(family.acceleration(np.array([[0.0, 0.0, 300.0]]), 2.0))
        assert not np.any(family_strain_responses(model(family, cavity=None), 2.0))
        assert np.max(np.abs(family_strain_responses(model(family, cavity=10.0), 2.0))) > 0.0

    def test_witnesses_see_love_waves_on_horizontal_axes_only(self) -> None:
        """The vertical witnesses' auto-spectra are zero; the horizontal ones are not."""
        covariance = family_joint_covariance(np.array([2.0]), model(love(), self_noise=False))
        auto = dict(zip(covariance.witness_channel_order, np.diagonal(covariance.witness_block()[0]).real, strict=True))
        assert auto["S0z"] == 0.0
        assert auto["C3z"] == 0.0
        assert auto["B1x"] > 0.0
        assert auto["C2y"] > 0.0

    def test_density_follows_the_layering(self) -> None:
        """The cavity term uses the local density: layer above ``H``, half-space below."""
        densities = love().density_at(np.array([[0.0, 0.0, 10.0], [0.0, 0.0, H], [0.0, 0.0, H + 10.0]]))
        np.testing.assert_array_equal(
            densities, [LAYER.layer_density, LAYER.half_space_density, LAYER.half_space_density]
        )

    def test_rejects_positions_above_the_surface_and_coarse_grids(self) -> None:
        """The medium is ``z >= 0``; at least four azimuth cells."""
        with pytest.raises(ValueError, match="non-negative"):
            love().displacement(np.array([[0.0, 0.0, -5.0]]), 2.0)
        with pytest.raises(ValueError, match="z >= 0"):
            love().acceleration(np.array([[0.0, 0.0, -5.0]]), 2.0)
        with pytest.raises(ValueError, match="n_azimuth"):
            love(n_azimuth=2)
        assert isinstance(LAYER, LayerOverHalfSpace)
