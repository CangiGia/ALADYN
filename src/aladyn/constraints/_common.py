"""Shared low-level helpers used by all joint implementations.

Not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..math import quaternions as _q
from ..math.vectors import skew

if TYPE_CHECKING:
    from ..model.body import RigidBody
    from ..model.ground import Ground

__all__ = ["centripetal", "dA_dp", "has_dofs"]


def has_dofs(body: RigidBody | Ground) -> bool:
    """Return ``True`` iff ``body`` exposes free DoFs (i.e. is not a ground).

    Ground sets ``mass = +inf``; rigid bodies have finite mass.
    """
    return bool(np.isfinite(getattr(body, "mass", np.inf)))


def dA_dp(u: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\partial(A\,\mathbf s')/\partial \mathbf p`, shape ``(3, 4)``.

    Given a body-frame vector :math:`\mathbf s'` rotated into the global
    frame as :math:`\mathbf u = A(\mathbf p)\,\mathbf s'`, the derivative
    with respect to the body's Euler parameters is

    .. math::

        \frac{\partial \mathbf u}{\partial \mathbf p}
        = -\,\widetilde{\mathbf u}\;\bigl(2\,E(\mathbf p)\bigr).

    Parameters
    ----------
    u
        Global-frame vector :math:`A(\mathbf p)\,\mathbf s'`, shape ``(3,)``.
    p
        Unit quaternion :math:`\mathbf p`, shape ``(4,)``.
    """
    return -skew(u) @ (2.0 * _q.E(p))


def centripetal(omega: ArrayLike, u: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\boldsymbol\omega \times (\boldsymbol\omega \times \mathbf u)`."""
    w = np.asarray(omega, dtype=np.float64)
    v = np.asarray(u, dtype=np.float64)
    return np.cross(w, np.cross(w, v))
