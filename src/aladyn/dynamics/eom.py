"""Augmented DAE assembly.

Builds, for the current state ``(q, q̇, t)``:

    | M       Φ_q^T | | q̈ |   | Q_e + Q_v |
    |               | |   | = |           |
    | Φ_q      0    | | λ  |   |     γ     |

where the constraint vector ``Φ`` includes both joint constraints and the
per-body Euler-parameter normalization ``p^T p − 1 = 0``.

Reference: Shabana, *Computational Dynamics*, eq. (6.142)–(6.146).

To be implemented.
"""
