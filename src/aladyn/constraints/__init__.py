"""Constraints: kinematic joints connecting bodies.

Every joint is a :class:`~aladyn.constraints.base.Constraint` subclass
exposing :meth:`phi`, :meth:`phi_q` and :meth:`gamma` (Shabana eqs. 6.66
-- 6.74). The Euler-parameter normalization constraint
:math:`\\mathbf p^\\mathsf{T}\\mathbf p - 1 = 0` is appended automatically
by the dynamics layer for every body, so joints do not handle it.

Each joint lives in its own module named after it: see
:mod:`~aladyn.constraints.spherical`, :mod:`~aladyn.constraints.revolute`,
etc.
"""

from .base import Constraint, JacobianBlock
from .cylindrical import CylindricalJoint
from .planar import PlanarJoint
from .prismatic import PrismaticJoint
from .revolute import RevoluteJoint
from .spherical import SphericalJoint
from .universal import UniversalJoint

__all__ = [
    "Constraint",
    "CylindricalJoint",
    "JacobianBlock",
    "PlanarJoint",
    "PrismaticJoint",
    "RevoluteJoint",
    "SphericalJoint",
    "UniversalJoint",
]
