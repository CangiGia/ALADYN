"""3D mathematical primitives.

This package is the foundation of the entire library. It MUST stay free of
any ALADYN-internal dependency (only NumPy/SciPy allowed) and MUST be the
most heavily unit-tested package — bugs here propagate everywhere.

Modules
-------
vectors      : skew/tilde matrices, cross-product helpers.
rotations    : rotation matrices, Euler/Cardan angle conversions.
quaternions  : Euler-parameter algebra, ``A(p)``, ``G(p)``, ``E(p)``,
               relations between ``ṗ`` and ``ω``.
transforms   : SE(3) homogeneous transforms and twist algebra.
"""
