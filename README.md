# gwmock-newtonian

[![Python CI](https://github.com/isaac-cf-wong/gwmock-newtonian/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/gwmock-newtonian/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`gwmock-newtonian` simulates Newtonian noise in triangular gravitational-wave
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

- `newtonian_rayleigh` (`gwmock_newtonian.joint.NewtonianJointSimulator`) — the
  physical backend: three ET strains plus N witnesses, with a Hermitian
  positive-semidefinite one-sided joint covariance and seeded realizations of a
  declared segmented process (see the module docstring for the process and the
  cross-spectrum sign convention).
- `newtonian_scaffold` (`gwmock_newtonian.simulator.NewtonianScaffoldSimulator`)
  — a physics-free placeholder (temporally white, fixed equicorrelation) kept
  for protocol and adapter conformance tests.

A compatibility adapter (`StrainOnlyAdapter`) lets existing `gwmock-noise`
consumers receive strain-only output from either backend.

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
