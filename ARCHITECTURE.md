# ALADYN — Architecture & Design Decisions

> **Automated Library for Advanced DYNamics** — 3D rigid multibody simulation
> in Python, evolution of [PMD](../PMD) (planar) into the spatial domain.

This document fixes the foundational design choices. Every contributor (and
future-self) must read it before adding code. Deviations require an explicit
update here.

---

## 1. Scope

In-scope **now** (v0.x):
- 3D rigid bodies with absolute coordinates.
- Lower-pair joints (Revolute, Prismatic, Spherical, Cylindrical, Universal,
  Planar) and primitive constraints (sph–sph distance, dot-1, dot-2, …).
- Applied forces and torques, gravity, TSDA / RSDA elements, user-defined
  forces.
- DAE-index-3 dynamics with Baumgarte stabilization (and projection as
  fallback).
- Initial assembly via Newton–Raphson on position constraints.
- Kinematic, static and dynamic analyses.
- Examples, regression tests, Sphinx docs.

In-scope as **architectural placeholders** (modules reserved, implemented later):
- Contact / collision (`contact/`).
- Flexible bodies via Floating Frame of Reference (`model/flex_body.py`).
- Sensors (`sensors/`), actuators (`actuators/`), control loops (`control/`).
- Pre/post-processor GUI (`gui/`, mirroring PMD).
- Exporters (URDF, MBDyn, FMI/FMU) under `io/`.

Out-of-scope: real-time, GPU, distributed solvers.

---

## 2. Reference textbook

**Primary:** A. A. Shabana — *Computational Dynamics* (3rd ed.) and *Dynamics
of Multibody Systems* (4th ed.).

Notation, symbol names and equation numbers cited in docstrings refer to
Shabana unless stated otherwise. Secondary references (Nikravesh, Haug,
Featherstone) may be cited when their formulation is clearer for a specific
topic.

---

## 3. Orientation parametrization

**Primary:** Euler parameters (unit quaternions), `p = [e0, e1, e2, e3]^T`,
with the constraint `p^T p = 1`.

Rationale:
- No singularities (gimbal lock) — required for general spatial motion.
- Linear in the rotation matrix `A(p)`.
- Standard in modern multibody codes (ADAMS, MBDyn, Chrono).
- Shabana ch. 2 / ch. 6 develops the entire formulation in Euler parameters.

Conventions:
- Angular velocity in the **body frame** is `ω' = 2 G(p) ṗ` (Shabana eq. 2.107).
- Angular velocity in the **global frame** is `ω = 2 E(p) ṗ`.
- `G` and `E` are the 3×4 transformation matrices defined in Shabana ch. 2.
- Euler/Cardan angles are supported only as **input/output utilities** in
  `math/rotations.py`, never as state variables.

---

## 4. Equations of motion

**Formulation:** absolute coordinates + Lagrange multipliers (DAE).

For each body i, generalized coordinates are
`q_i = [R_i^T, p_i^T]^T  ∈ ℝ^7`, so the system state size is `n = 7·nB`.

The augmented system (Shabana eq. 6.142–6.146):

```
| M         Φ_q^T | | q̈ |   | Q_e + Q_v |
|                 | |   | = |           |
| Φ_q        0    | | λ  |   |     γ     |
```

with the Euler-parameter normalization constraint
`Φ_p,i ≡ p_i^T p_i − 1 = 0` appended to `Φ`.

**Stabilization:** Baumgarte (default α = β = 5) implemented in
`dynamics/stabilization.py`. Coordinate projection is provided as a fallback
for ill-conditioned cases.

**Index reduction:** none for now (we integrate the index-1 form obtained by
twice differentiating the constraints, with Baumgarte). A future
`coordinates.py` may add coordinate partitioning to obtain a pure ODE.

---

## 5. Package layout

`src/aladyn/` — proper *src-layout*. Installable via `pip install -e .`. No
`sys.path` manipulation in `__init__.py` (PMD's pattern is intentionally
abandoned).

| package         | responsibility                                              |
|-----------------|-------------------------------------------------------------|
| `core/`         | `Base` counter, `UnitSystem`, logging, generic utilities    |
| `math/`         | vectors, rotation matrices, quaternions, SE(3) transforms   |
| `model/`        | `Ground`, `Body`, `Marker`, `shapes`, (future) flex bodies  |
| `constraints/`  | joints, primitives, drivers, `Function`                     |
| `forces/`       | gravity, springs, applied forces/torques, user forces       |
| `dynamics/`     | EoM assembly (M, Φ, Φ_q, γ, Q), coordinate handling, stab.  |
| `solver/`       | top-level model facade, kinematics / statics / dynamics     |
| `builder/`      | initial assembly via NR                                     |
| `io/`           | serialization, exporters                                    |
| `contact/`      | (placeholder) collision detection + force models            |
| `sensors/`      | (placeholder) measurement objects                           |
| `actuators/`    | (placeholder) controllable forces/torques                   |
| `control/`      | (placeholder) control laws & loops                          |

Cross-package import rule (enforced by review):

```
math  →  (no internal deps)
core  →  (no internal deps)
model → core, math
constraints, forces → core, math, model
dynamics → core, math, model, constraints, forces
solver → everything except gui / io
builder → core, math, model, constraints
io → model, constraints, forces, solver
gui → solver, io                (one-way; never imported by core code)
```

No circular imports. `solver.model.SpatialMultibodyModel` is the only public
facade that wires everything together.

---

## 6. Testing strategy

- `tests/unit/` — one file per source module; pure-math tests get hammered
  with `hypothesis`-style randomized inputs (when introduced).
- `tests/integration/` — assembly, kinematic chains, dynamic benchmarks at
  the subsystem level.
- `tests/regression/` — golden trajectories under `refs/`. Regenerated only
  via an explicit `_regen_refs.py` script (same pattern as PMD).
- Markers: `unit`, `integration`, `regression`, `slow`.

A quaternion-norm drift check and a constraint-violation check
(`||Φ||∞ < tol`) are part of every dynamic regression test.

---

## 7. Coding conventions

- Python ≥ 3.10, type hints on all public APIs.
- NumPy arrays for runtime data; `numpy.typing.NDArray` in signatures.
- Docstrings: NumPy style, with explicit reference to Shabana equations,
  e.g. `# Shabana eq. (6.142)`.
- Logging via `logging.getLogger(__name__)`; no `print`.
- Public symbols re-exported from each subpackage `__init__.py`.
- No optional dependencies imported at module top-level (lazy imports for
  `gui`, `io` exporters, optional integrators).

---

## 8. Roadmap (high level)

1. `math/` — rotations, quaternions, transforms (+ exhaustive unit tests).
2. `core/` + `model/` — `Base`, `Ground`, `RigidBody`, `Marker`.
3. `constraints/lower_pair.py` — Revolute, Spherical (minimal viable joint
   set).
4. `forces/` — gravity, applied force/torque.
5. `dynamics/eom.py` + `solver/dynamics.py` — first DAE integration of a 3D
   pendulum.
6. Remaining joints, primitive constraints, springs.
7. `builder/assembly.py` — initial NR assembly.
8. Examples + regression refs.
9. `io/serialization.py` — JSON model save/load.
10. GUI (mirror PMD).
11. Contact, flexible bodies, sensors/actuators/control.
