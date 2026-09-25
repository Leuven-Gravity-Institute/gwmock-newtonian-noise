# gwmock-newtonian

[![Python CI](https://github.com/isaac-cf-wong/gwmock-newtonian/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/gwmock-newtonian/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`gwmock-newtonian` simulates Newtonian noise in triangular gravitational-wave
detectors such as the Einstein Telescope (ET). Its eventual scope is the
Newtonian-noise seismic field, detector/witness response, and joint
strain+witness covariance/realization generation — **none of that physics is implemented yet**; see below.

In its current scaffold form (protocol conformance only) this
package implements **no Newtonian/seismic physics**. What exists is structural
scaffolding: a validated config schema, RNG policy, channel/provenance types,
and a covariance/realization/stream API conforming to the optional
[`JointStrainWitnessSimulator` protocol](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/joint_protocol.py)
from the public
[`gwmock-noise`](https://github.com/Leuven-Gravity-Institute/gwmock-noise)
package, backed by a physics-free placeholder simulator (temporally white, fixed
equicorrelation) so the protocol, RNG policy, and adapter can be
conformance-tested before any physical model exists. The package registers
itself as a `gwmock_noise.joint_backends` entry point, so `gwmock-noise` can
discover this backend without ever importing it directly. A
compatibility adapter lets existing `gwmock-noise` consumers keep receiving
strain-only output from this joint-capable (but not yet physical) simulator. The
physical model is not implemented yet.

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
