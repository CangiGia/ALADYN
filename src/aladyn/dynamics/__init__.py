"""Equations-of-motion assembly and stabilization.

This package is the *physics* layer. It produces the matrices/vectors the
solver consumes. Numerical integration belongs to ``solver/``, not here.

Modules
-------
eom            : assembles M, Φ, Φ_q, γ, Q for the augmented DAE system.
coordinates    : Euler-parameter handling, ``q``/``q̇`` packing/unpacking,
                 angular velocity ↔ ``ṗ`` conversions for the whole system.
stabilization  : Baumgarte stabilization and coordinate projection.
"""
