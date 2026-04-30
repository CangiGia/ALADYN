# ALADYN

**Automated Library for Advanced DYNamics** — a Python library for
3D rigid multibody simulation, spatial evolution of [PMD](https://github.com/CangiGia/PMD).

> Status: scaffolding / pre-alpha. See [ARCHITECTURE.md](ARCHITECTURE.md)
> for the design decisions that govern the codebase.

## Highlights (target)

- 3D rigid bodies parametrized with **Euler parameters (unit quaternions)** —
  no singularities.
- **Absolute coordinates + DAE** formulation with Baumgarte stabilization,
  following Shabana, *Computational Dynamics*.
- Full lower-pair joint family (R, P, S, C, U, Planar) plus primitive
  constraints.
- Modular layout reserving space for contacts, flexible bodies (FFR),
  sensors/actuators/control and a Qt GUI.

## Install (development)

```powershell
pip install -e .[dev]
```

## Layout

```
src/aladyn/
    core/  math/  model/  constraints/  forces/
    dynamics/  solver/  builder/  io/
    contact/  sensors/  actuators/  control/    (placeholders)
tests/{unit,integration,regression}
examples/  validation/  docs/  gui/
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the rationale behind every
choice and the import-graph rules.

## License

See [LICENSE](LICENSE).

