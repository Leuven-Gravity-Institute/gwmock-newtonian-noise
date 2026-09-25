---
title: API Reference
description: Public API of gwmock-newtonian.
icon: material/api
---

`gwmock-newtonian` implements the optional
[`JointStrainWitnessSimulator`](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/joint_protocol.py)
protocol from the public `gwmock-noise` package.

> **Note:** In the package scaffold (protocol conformance only) no
> Newtonian/seismic physics is modelled -- the config schema, RNG policy, and
> covariance/realization/stream APIs below are structural scaffolding only.

## Main Sections

- **[Configuration](config)** – validated configuration schema.
- **[RNG policy](rng)** – deterministic RNG policy.
- **[Provenance](provenance)** – machine-readable realization provenance.
- **[Covariance](covariance)** – analytic covariance API.
- **[Simulator](simulator)** – realization and stream API.
- **[Compatibility adapter](adapter)** – strain-only view for legacy
  `gwmock-noise` consumers.
- **[Utility](utils)** – logging utilities.
