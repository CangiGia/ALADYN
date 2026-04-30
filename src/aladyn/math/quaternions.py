"""Euler-parameter (unit-quaternion) algebra.

This is the **primary** orientation representation of ALADYN. All bodies
store their orientation as a 4-vector ``p = [e0, e1, e2, e3]`` subject to
``p^T p = 1``.

Provides:
- rotation matrix ``A(p)`` (Shabana eq. 2.96);
- transformation matrices ``G(p)`` (body frame, eq. 2.107) and ``E(p)``
  (global frame, eq. 2.103);
- relations ``ω' = 2 G(p) ṗ``, ``ω = 2 E(p) ṗ``;
- quaternion product, conjugate, normalization, SLERP;
- conversions to/from rotation matrix and Euler angles.

To be implemented.
"""
