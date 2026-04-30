"""Abstract base class for all constraints (``Joint``).

Concrete constraints implement:
- ``n_eq``               — number of scalar equations contributed.
- ``Phi(q) -> NDArray``  — constraint residual.
- ``Phi_q(q) -> NDArray``— Jacobian wrt generalized coordinates.
- ``gamma(q, qd) -> NDArray`` — RHS of the acceleration-level equation.

To be implemented.
"""
