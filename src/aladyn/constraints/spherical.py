r"""Spherical joint (ball-and-socket).

The spherical joint coincides the origins of two markers, eliminating
their relative translation while leaving all three relative rotations
free. It contributes three scalar equations.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import centripetal, dA_dp, has_dofs
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["SphericalJoint"]


class SphericalJoint(Constraint):
    r"""Ball-and-socket joint coinciding two marker origins.

    The constraint enforces

    .. math::

        \boldsymbol\Phi(q) = \mathbf r_i^P - \mathbf r_j^P
        = (\mathbf R_i + A_i\,\mathbf s'_i)
        - (\mathbf R_j + A_j\,\mathbf s'_j) = \mathbf 0,

    i.e. the global-frame coincidence of the origins of two markers. It
    contributes three scalar equations and constrains three translational
    degrees of freedom while leaving all relative rotations free.

    Parameters
    ----------
    marker_i, marker_j
        The two markers whose origins must coincide. Their parents may be
        any combination of :class:`~aladyn.model.body.RigidBody` and
        :class:`~aladyn.model.ground.Ground`.
    name
        Optional human-readable name.
    """

    n_eq: int = 3

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j

    # ── State queries on the two parents ──────────────────────────────

    @property
    def marker_i(self) -> Marker:
        """First marker (defining the ``+`` side of the residual)."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Second marker (defining the ``-`` side of the residual)."""
        return self._mj

    # ── Constraint contract ───────────────────────────────────────────

    def phi(self) -> NDArray[np.float64]:
        """Residual ``r_i - r_j`` evaluated at the current body states."""
        return self._mi.position_global - self._mj.position_global

    def phi_q(self) -> list[JacobianBlock]:
        r"""Per-body Jacobian blocks.

        For the marker on body ``b``, with :math:`\mathbf u_b = A_b\,\mathbf s'_b`,
        the contribution to :math:`\partial\Phi/\partial q_b` is

        .. math::

            \frac{\partial\Phi}{\partial R_b} = \pm\,I_3, \qquad
            \frac{\partial\Phi}{\partial p_b} = \mp\,\widetilde{\mathbf u_b}\;
            \bigl(2\,E(\mathbf p_b)\bigr),

        with ``+`` for body ``i`` and ``-`` for body ``j``.
        """
        blocks: list[JacobianBlock] = []

        for sign, marker in ((+1.0, self._mi), (-1.0, self._mj)):
            parent = marker.parent
            # Skip the ground: its 7 coordinates are fixed.
            if not has_dofs(parent):
                continue
            u = parent.rotation_matrix @ marker.position_local  # A_b · s'_b
            J_R = sign * np.eye(3, dtype=np.float64)
            J_p = sign * dA_dp(u, parent.quaternion)
            blocks.append((parent, J_R, J_p))

        return blocks

    def gamma(self) -> NDArray[np.float64]:
        r"""Acceleration-level RHS.

        Differentiating :math:`\Phi` twice in time yields

        .. math::

            \ddot{\mathbf r}_b^P = \ddot{\mathbf R}_b + \dot{\boldsymbol\omega}_b
            \times \mathbf u_b + \boldsymbol\omega_b \times (\boldsymbol\omega_b
            \times \mathbf u_b).

        Collecting the parts independent from ``q̈`` gives

        .. math::

            \boldsymbol\gamma = -\bigl[\boldsymbol\omega_i \times
            (\boldsymbol\omega_i \times \mathbf u_i)
            - \boldsymbol\omega_j \times
            (\boldsymbol\omega_j \times \mathbf u_j)\bigr].
        """
        ui = self._mi.parent.rotation_matrix @ self._mi.position_local
        uj = self._mj.parent.rotation_matrix @ self._mj.position_local
        wi = self._mi.parent.omega_global
        wj = self._mj.parent.omega_global
        return -(centripetal(wi, ui) - centripetal(wj, uj))
