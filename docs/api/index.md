---
title: API Reference
description: Public API of gwmock-newtonian.
icon: material/api
---

`gwmock-newtonian` implements the optional
[`JointStrainWitnessSimulator`](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/joint_protocol.py)
protocol from the public `gwmock-noise` package.

Two backends implement it: the physical `newtonian_rayleigh` backend
([Joint simulator](joint)) and the physics-free `newtonian_scaffold` placeholder
([Simulator](simulator)), kept for conformance tests.

## Main Sections

- **[Configuration](config)** – validated configuration schema.
- **[RNG policy](rng)** – deterministic RNG policy.
- **[Provenance](provenance)** – machine-readable realization provenance.
- **[Covariance](covariance)** – analytic covariance API.
- **[Joint simulator](joint)** – the physical joint strain+witness backend.
- **[Simulator](simulator)** – the physics-free scaffold backend.
- **[Compatibility adapter](adapter)** – strain-only view for legacy
  `gwmock-noise` consumers.
- **[Utility](utils)** – logging utilities.
