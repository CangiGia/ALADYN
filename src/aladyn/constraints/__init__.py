"""Constraints: lower-pair joints, primitive constraints, drivers.

Every joint is an ABC subclass exposing ``Phi(q)``, ``Phi_q(q)`` and
``gamma(q, q̇)`` (Shabana eqs. 6.66–6.74). The Euler-parameter
normalization constraint ``p^T p − 1 = 0`` is appended automatically by the
dynamics layer for every body, so joints do not handle it.
"""
