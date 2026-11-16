"""Controls every field family must pass: zero-source, dimensional, symmetry, covariance and realization.

The same checks run on every family -- the Rayleigh adapter, both body-wave
geometries, the diffuse body-wave proxy, Love waves, point sources and a
mixture of all of them -- so a family cannot be added without them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _family_fixtures import (
    LAYER,
    MEDIUM,
    SAMPLING_FREQUENCY,
    SEGMENT_DURATION,
    assert_covariance_close,
    flat,
    model,
    realization_grid,
)

from gwmock_newtonian_noise.detector.newtonian import GRAVITATIONAL_CONSTANT
from gwmock_newtonian_noise.families import (
    BodyWaveFamily,
    FamilyMixture,
    FieldFamily,
    LoveWaveFamily,
    PointSource,
    PointSourceFamily,
    RayleighFamily,
    equipartitioned_body_waves,
    family_joint_covariance,
    family_joint_mode_responses,
    synthesize_family_joint,
    uniform_sphere_density,
)
from gwmock_newtonian_noise.families.love import LayerOverHalfSpace
from gwmock_newtonian_noise.seismic.field import RayleighWaveFieldConfig
from gwmock_newtonian_noise.seismic.medium import ElasticHalfSpace
from gwmock_newtonian_noise.seismic.spectrum import uniform_directional_density

KINDS = ("rayleigh", "body_full", "body_half", "diffuse", "love", "point", "mixture")

#: Frequency used by the single-frequency controls.
F0 = 3.0

#: Azimuth cells of the gridded families; rotations by multiples of 2 pi / N map the grid onto itself.
N_AZIMUTH = 8

#: Evaluation positions (metres, z down): surface, borehole and deep points, none near a point source.
POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [150.0, 80.0, 120.0], [-60.0, 40.0, 290.0], [70.0, -90.0, 310.0], [900.0, -400.0, 260.0]]
)


def _medium(speed_scale: float, density_scale: float) -> ElasticHalfSpace:
    return ElasticHalfSpace(
        p_wave_speed=MEDIUM.p_wave_speed * speed_scale,
        s_wave_speed=MEDIUM.s_wave_speed * speed_scale,
        density=MEDIUM.density * density_scale,
    )


def build(kind: str, *, speed_scale: float = 1.0, density_scale: float = 1.0, psd: float = 1.0) -> FieldFamily:
    """Return a family of ``kind``; ``speed_scale`` also scales every length that is part of the family."""
    medium = _medium(speed_scale, density_scale)
    if kind == "rayleigh":
        field = RayleighWaveFieldConfig(
            medium=medium,
            surface_displacement_psd=flat(psd),
            directional_density=uniform_directional_density,
            n_azimuth=N_AZIMUTH,
        )
        return RayleighFamily(field=field, n_depth=64)
    if kind in {"body_full", "body_half"}:
        return BodyWaveFamily(
            medium=medium,
            p_psd=flat(psd),
            sv_psd=flat(0.5 * psd),
            sh_psd=flat(0.7 * psd),
            directional_density=uniform_sphere_density,
            free_surface=kind == "body_half",
            n_polar=3,
            n_azimuth=N_AZIMUTH,
        )
    if kind == "diffuse":
        return equipartitioned_body_waves(medium, flat(psd), n_polar=3, n_azimuth=N_AZIMUTH)
    if kind == "love":
        layer = LayerOverHalfSpace(
            layer_thickness=LAYER.layer_thickness * speed_scale,
            layer_s_wave_speed=LAYER.layer_s_wave_speed * speed_scale,
            layer_density=LAYER.layer_density * density_scale,
            half_space_s_wave_speed=LAYER.half_space_s_wave_speed * speed_scale,
            half_space_density=LAYER.half_space_density * density_scale,
        )
        return LoveWaveFamily(
            medium=layer,
            surface_displacement_psd=flat(psd),
            directional_density=uniform_directional_density,
            n_azimuth=N_AZIMUTH,
        )
    if kind == "point":
        return PointSourceFamily(
            medium=medium,
            sources=(
                PointSource(
                    (400.0 * speed_scale, 250.0 * speed_scale, 450.0 * speed_scale), flat(psd), 100.0 * speed_scale
                ),
                PointSource(
                    (-300.0 * speed_scale, -500.0 * speed_scale, 150.0 * speed_scale),
                    flat(0.5 * psd),
                    100.0 * speed_scale,
                ),
            ),
        )
    if kind == "mixture":
        # Love is left out: its layered density differs from the half-space medium, which a mixture rejects.
        return FamilyMixture(
            tuple(
                build(k, speed_scale=speed_scale, density_scale=density_scale, psd=psd)
                for k in ("rayleigh", "body_half", "point")
            )
        )
    raise ValueError(kind)


def _response_covariance(family: FieldFamily, positions: np.ndarray, frequency: float = F0) -> np.ndarray:
    """Return the covariance of stacked (displacement, acceleration) at ``positions`` -- field only."""
    displacement = family.displacement(positions, frequency).reshape(family.n_modes, -1)
    acceleration = family.acceleration(positions, frequency).reshape(family.n_modes, -1)
    stacked = np.concatenate([displacement, acceleration / GRAVITATIONAL_CONSTANT], axis=1)
    return np.einsum("mi,mj,m->ij", stacked, stacked.conj(), family.mode_power(frequency))


@pytest.mark.parametrize("kind", KINDS)
class TestContract:
    """Shapes and the protocol."""

    def test_family_satisfies_the_protocol_with_a_fixed_mode_count(self, kind: str) -> None:
        """The family is a FieldFamily whose mode count does not depend on frequency."""
        family = build(kind)
        assert isinstance(family, FieldFamily)
        for frequency in (0.5, F0, 7.0):
            assert family.mode_power(frequency).shape == (family.n_modes,)
            assert family.displacement(POSITIONS, frequency).shape == (family.n_modes, len(POSITIONS), 3)
            assert family.acceleration(POSITIONS, frequency).shape == (family.n_modes, len(POSITIONS), 3)

    def test_non_positive_frequency_is_rejected(self, kind: str) -> None:
        """A zero frequency has no propagating mode and is refused."""
        family = build(kind)
        with pytest.raises(ValueError, match="frequency"):
            family.displacement(POSITIONS, 0.0)


@pytest.mark.parametrize("kind", KINDS)
class TestZeroSource:
    """A family with no power produces exactly nothing."""

    def test_zero_power_gives_a_zero_covariance_and_a_zero_realization(self, kind: str) -> None:
        """Zero source power gives an exactly zero covariance and realization."""
        joint = model(build(kind, psd=0.0), self_noise=False)
        frequencies = realization_grid()
        responses, power = family_joint_mode_responses(frequencies, joint)
        assert not np.any(power)
        covariance = family_joint_covariance(frequencies, joint, responses=(responses, power))
        assert not np.any(covariance.matrix)
        series = synthesize_family_joint(responses, power, joint, SEGMENT_DURATION, SAMPLING_FREQUENCY, seed=1)
        assert all(not np.any(values) for values in series.values())

    def test_a_powered_family_is_not_zero(self, kind: str) -> None:
        """Guard for the control above: the same configuration with power is non-trivial."""
        joint = model(build(kind), self_noise=False)
        covariance = family_joint_covariance(np.array([F0]), joint)
        # Not every witness need see every family (a vertical sensor sees no Love wave), but some must.
        assert np.trace(covariance.witness_block()[0]).real > 0.0
        assert np.all(np.diagonal(covariance.strain_block()[0]).real > 0.0)


@pytest.mark.parametrize("kind", KINDS)
class TestDimensional:
    """Units and scaling: linearity in power and density, and length-scale invariance of the kernels."""

    def test_covariance_is_linear_in_the_source_power(self, kind: str) -> None:
        """Scaling the source PSD scales the field covariance by the same factor."""
        frequencies = np.array([1.0, F0])
        base = family_joint_covariance(frequencies, model(build(kind), self_noise=False)).matrix
        scaled = family_joint_covariance(frequencies, model(build(kind, psd=3.5), self_noise=False)).matrix
        assert_covariance_close(scaled, 3.5 * base, 1e-10)

    def test_acceleration_is_proportional_to_density_and_displacement_is_not(self, kind: str) -> None:
        """Gravity scales with the background density; the kinematics do not."""
        base, heavy = build(kind), build(kind, density_scale=2.0)
        np.testing.assert_allclose(
            heavy.displacement(POSITIONS, F0), base.displacement(POSITIONS, F0), rtol=1e-10, atol=0.0
        )
        np.testing.assert_allclose(
            heavy.acceleration(POSITIONS, F0), 2.0 * base.acceleration(POSITIONS, F0), rtol=1e-9, atol=1e-30
        )
        np.testing.assert_allclose(heavy.density_at(POSITIONS), 2.0 * base.density_at(POSITIONS), rtol=1e-12)

    def test_kernels_are_invariant_under_a_common_length_scale(self, kind: str) -> None:
        """Scaling every speed and length by s at fixed frequency scales every wavelength by s.

        The Newtonian kernel ``G delta_rho / r^2 d^3 r`` with ``delta_rho ~ rho0 k u``
        is scale-free, so displacement and acceleration per unit amplitude at the
        scaled positions must be unchanged. A stray length in any kernel breaks this.
        """
        scale = 2.5
        base, scaled = build(kind), build(kind, speed_scale=scale)
        np.testing.assert_allclose(
            scaled.displacement(scale * POSITIONS, F0), base.displacement(POSITIONS, F0), rtol=1e-7, atol=1e-14
        )
        reference = base.acceleration(POSITIONS, F0)
        atol = 1e-7 * float(np.max(np.abs(reference))) if np.any(reference) else 0.0
        np.testing.assert_allclose(scaled.acceleration(scale * POSITIONS, F0), reference, rtol=1e-6, atol=atol)

    def test_acceleration_per_g_rho_is_order_unity_per_unit_displacement(self, kind: str) -> None:
        """``|a| / (G rho0 |u|)`` is a pure number of order ``4 pi``: a units slip shows up as powers of ten."""
        family = build(kind)
        acceleration = np.abs(family.acceleration(POSITIONS[1:], F0)).max()
        displacement = np.abs(family.displacement(POSITIONS[1:], F0)).max()
        ratio = acceleration / (GRAVITATIONAL_CONSTANT * MEDIUM.density * displacement)
        if kind == "love":
            assert ratio == 0.0
        else:
            assert 1e-2 < ratio < 1e2, ratio


@pytest.mark.parametrize("kind", KINDS)
class TestSymmetry:
    """Rotating the whole configuration about the vertical axis rotates the covariance."""

    @staticmethod
    def _rotation(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _rotated(self, kind: str, angle: float) -> FieldFamily:
        """Return the family with its own geometry rotated (the gridded families are already invariant)."""
        rotation = self._rotation(angle)

        def rotate(family: FieldFamily) -> FieldFamily:
            if isinstance(family, FamilyMixture):
                return FamilyMixture(tuple(rotate(component) for component in family.components))
            if not isinstance(family, PointSourceFamily):
                return family  # isotropic in azimuth on a grid that a cell rotation maps onto itself
            return dataclasses.replace(
                family,
                sources=tuple(
                    dataclasses.replace(source, position=tuple(rotation @ np.asarray(source.position)))
                    for source in family.sources
                ),
            )

        return rotate(build(kind))

    def test_covariance_rotates_with_the_configuration(self, kind: str) -> None:
        """C(R p) = R C(p) R^T for a rotation that maps the direction grid onto itself."""
        angle = 3 * 2.0 * np.pi / N_AZIMUTH
        rotation = self._rotation(angle)
        base = _response_covariance(build(kind), POSITIONS)
        rotated = _response_covariance(self._rotated(kind, angle), POSITIONS @ rotation.T)
        block = np.kron(np.eye(2 * len(POSITIONS)), rotation)
        expected = block @ base @ block.T
        scale = np.max(np.abs(base))
        np.testing.assert_allclose(rotated, expected, rtol=0.0, atol=1e-9 * scale)

    def test_the_symmetry_control_can_fail(self, kind: str) -> None:
        """Guard: the same check against an unrotated covariance fails (the configuration is not trivial)."""
        angle = 3 * 2.0 * np.pi / N_AZIMUTH
        rotation = self._rotation(angle)
        base = _response_covariance(build(kind), POSITIONS)
        rotated = _response_covariance(self._rotated(kind, angle), POSITIONS @ rotation.T)
        assert np.max(np.abs(rotated - base)) > 1e-3 * np.max(np.abs(base))


@pytest.mark.parametrize("kind", KINDS)
class TestCovarianceAndRealization:
    """The joint covariance is a valid covariance, and realizations converge to it."""

    def test_every_slice_is_hermitian_positive_semidefinite(self, kind: str) -> None:
        """Each joint covariance slice is Hermitian with no negative eigenvalue."""
        covariance = family_joint_covariance(np.array([0.7, F0, 6.0]), model(build(kind)))
        for matrix in covariance.matrix:
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12 * np.max(np.abs(matrix)))
            eigenvalues = np.linalg.eigvalsh(matrix)
            assert eigenvalues.min() > -1e-10 * eigenvalues.max()

    def test_realizations_are_seed_reproducible(self, kind: str) -> None:
        """The same seed reproduces a realization bit for bit; another seed differs."""
        joint = model(build(kind))
        responses, power = family_joint_mode_responses(realization_grid(), joint)
        first = synthesize_family_joint(responses, power, joint, SEGMENT_DURATION, SAMPLING_FREQUENCY, seed=7)
        again = synthesize_family_joint(responses, power, joint, SEGMENT_DURATION, SAMPLING_FREQUENCY, seed=7)
        other = synthesize_family_joint(responses, power, joint, SEGMENT_DURATION, SAMPLING_FREQUENCY, seed=8)
        for name in first:
            np.testing.assert_array_equal(first[name], again[name])
        assert any(not np.array_equal(first[name], other[name]) for name in first)

    def test_realized_cross_spectra_converge_to_the_covariance(self, kind: str) -> None:
        """Monte-Carlo periodograms, ``(2/(fs N)) E[conj(X_a) X_b]``, match ``C`` within sampling error.

        Every entry is compared in units of ``sqrt(C_aa C_bb) / sqrt(n)``, the
        standard error of a Gaussian cross-periodogram average, so a missing
        conjugate, a wrong power normalization or a mode drawn per channel instead
        of shared fails by many standard errors.
        """
        joint = model(build(kind))
        frequencies = realization_grid()
        responses, power = family_joint_mode_responses(frequencies, joint)
        analytic = family_joint_covariance(frequencies, joint, responses=(responses, power)).matrix
        n_realizations = 300
        n_samples = round(SEGMENT_DURATION * SAMPLING_FREQUENCY)
        names = (*joint.layout.geometry.names, *joint.array.channel_names)
        accumulator = np.zeros_like(analytic)
        seeds = np.random.SeedSequence(2026).spawn(n_realizations)
        for seed in seeds:
            series = synthesize_family_joint(responses, power, joint, SEGMENT_DURATION, SAMPLING_FREQUENCY, seed)
            spectra = np.fft.rfft(np.stack([series[name] for name in names]), axis=1)
            accumulator += np.einsum("af,bf->fab", spectra.conj(), spectra)
        empirical = accumulator * 2.0 / (SAMPLING_FREQUENCY * n_samples * n_realizations)
        interior = slice(1, frequencies.size - 1)
        diagonal = np.sqrt(np.einsum("fii->fi", analytic[interior]).real)
        scale = diagonal[:, :, None] * diagonal[:, None, :]
        z = np.abs(empirical[interior] - analytic[interior]) * np.sqrt(n_realizations) / scale
        assert np.max(z) < 6.0, np.max(z)
        assert np.mean(z) < 1.2, np.mean(z)
