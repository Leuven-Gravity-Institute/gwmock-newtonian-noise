# gwmock-newtonian-noise

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-newtonian-noise/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-newtonian-noise/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`gwmock-newtonian-noise` simulates Newtonian noise in triangular gravitational-wave
detectors such as the Einstein Telescope (ET): a Rayleigh-wave seismic field, the
Newtonian-noise strain response of the three ET triangle detectors, seismic
witness sensors and arrays observing the same field, and the joint
strain+witness covariance and realizations built from them.

Two backends implement the optional
[`JointStrainWitnessSimulator` protocol](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/joint_protocol.py)
from the public
[`gwmock-noise`](https://github.com/Leuven-Gravity-Institute/gwmock-noise)
package, and both register themselves as `gwmock_noise.joint_backends` entry
points, so `gwmock-noise` discovers them without importing this package
directly:

- `newtonian_rayleigh` (`gwmock_newtonian_noise.joint.NewtonianJointSimulator`) — the
  physical backend: three ET strains plus N witnesses, with a Hermitian
  positive-semidefinite one-sided joint covariance and seeded realizations of a
  declared segmented process (see the module docstring for the process and the
  cross-spectrum sign convention).
- `newtonian_scaffold` (`gwmock_newtonian_noise.simulator.NewtonianScaffoldSimulator`)
  — a physics-free placeholder (temporally white, fixed equicorrelation) kept
  for protocol and adapter conformance tests.

A compatibility adapter (`StrainOnlyAdapter`) lets existing `gwmock-noise`
consumers receive strain-only output from either backend.

## Broader Newtonian-noise families

`gwmock_newtonian_noise.families` extends the physics beyond the minimal Rayleigh
field, for out-of-family tests. Every family is a set of independent latent
modes with a displacement and a Newtonian-acceleration response, so one generic
consumer builds the strain, witness and joint covariances and realizations for
all of them, and independent families combine by mixing:

- `RayleighFamily` — the existing Rayleigh field through the family contract (it
  reproduces the existing joint covariance), plus `phase_velocity_spread`, a
  lateral-inhomogeneity proxy;
- `BodyWaveFamily` — P, SV and SH plane body waves with separate, required PSDs
  and a three-dimensional directional density, either in the infinite-medium
  limit or in a half-space with free-surface reflections;
  `equipartitioned_body_waves` is the diffuse (scattering-limit) field;
- `LoveWaveFamily` — fundamental Love waves of a layer over a half-space; they
  source no field-intrinsic gravity in this geometry but are seen by horizontal
  witnesses and reach the strain through the cavity term;
- `PointSourceFamily` — localized isotropic P-wave sources with their near
  field;
- `CavityModel` — the cavity-wall term of a test mass in an underground hall.

Sensors and test masses are placed in three dimensions (`z` is depth). No family
fixes a site: each lists its site parameters in `UNANCHORED_PARAMETERS`, none of
which has a default, and every value used in the tests is illustrative. Layered
Rayleigh modes, topography, attenuation, cavity scattering and free-surface
effects on point sources are not modelled.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`
(`uv run prek install` to enable them locally). Documentation is built with
[Zensical](https://zensical.org) from `docs/`
(`uv run --group docs zensical build`).
