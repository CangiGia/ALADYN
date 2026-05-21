r"""Revolute joint (hinge / pin).

The revolute joint allows a single rotational degree of freedom about a
common axis. It is built as a :class:`~aladyn.constraints.spherical.SphericalJoint`
(coincidence of the two marker origins, 3 equations) plus two ``dot-1``
constraints enforcing that the joint axis on body :math:`i` (the z-axis of
``marker_i``) stays parallel to the joint axis on body :math:`j` (the z-axis
of ``marker_j``). Total: **5 equations**, **1 DoF**.

Joint axis convention
---------------------
The revolute axis is the **z-axis** (third column) of each marker's local
orientation. The two body-fixed vectors orthogonal to the joint axis on body
``i`` — namely the x- and y- axes of ``marker_i`` — are dotted with the
j-side axis. Aligning both dot products to zero forces the two z-axes to
be parallel.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..math import quaternions as _q
from ._common import centripetal, dA_dp, has_dofs
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["RevoluteJoint"]


class RevoluteJoint(Constraint):
    r"""Hinge joint enforcing coaxial rotation between two markers.

    Constraint equations (5 total):

    1. **Coincidence (3)** — ``r_i^P - r_j^P = 0``.
    2. **Axis alignment (2)** — ``x_i · z_j = 0`` and ``y_i · z_j = 0``,
       where ``x_i, y_i`` are the x- and y-axes of ``marker_i`` (in the
       global frame) and ``z_j`` is the z-axis of ``marker_j``.

    The joint axis (z-axis of either marker, identical when the constraint
    is satisfied) is the single remaining rotational DoF.

    Parameters
    ----------
    marker_i, marker_j
        The two markers defining the hinge; their z-axes are the hinge
        axis. Their parents may be :class:`~aladyn.model.body.RigidBody`
        or :class:`~aladyn.model.ground.Ground`.
    name
        Optional human-readable name.
    """

    n_eq: int = 5

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j

    @property
    def marker_i(self) -> Marker:
        """Marker on body ``i`` (defines the ``+`` side of the residual)."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Marker on body ``j`` (defines the ``-`` side of the residual)."""
        return self._mj

    # ── Constraint contract ───────────────────────────────────────────

    def phi(self) -> NDArray[np.float64]:
        """Stack of coincidence (3) and two axis-alignment dot products (2)."""
        # Coincidence.
        d = self._mi.position_global - self._mj.position_global
        # Axis alignment: x_i · z_j and y_i · z_j.
        Ai = self._mi.orientation_global
        zj = self._mj.orientation_global[:, 2]
        dot_xz = float(Ai[:, 0] @ zj)
        dot_yz = float(Ai[:, 1] @ zj)
        return np.array([d[0], d[1], d[2], dot_xz, dot_yz], dtype=np.float64)

    def phi_q(self) -> list[JacobianBlock]:
        r"""Per-body Jacobian blocks, stacked across the 5 equations.

        For the coincidence rows (0-2), see
        :meth:`~aladyn.constraints.spherical.SphericalJoint.phi_q`.

        For a dot-1 row :math:`\Phi = \mathbf u_a^\mathsf{T} \mathbf u_b`
        (with ``a`` on body ``i`` and ``b`` on body ``j``):

        .. math::

            \frac{\partial\Phi}{\partial R_*} = \mathbf 0, \qquad
            \frac{\partial\Phi}{\partial \mathbf p_i}
            = (\mathbf u_a \times \mathbf u_b)^\mathsf{T}\;2\,E(\mathbf p_i),
            \qquad
            \frac{\partial\Phi}{\partial \mathbf p_j}
            = -(\mathbf u_a \times \mathbf u_b)^\mathsf{T}\;2\,E(\mathbf p_j).
        """
        bi, bj = self._mi.parent, self._mj.parent
        ui = bi.rotation_matrix @ self._mi.position_local
        uj_orig = bj.rotation_matrix @ self._mj.position_local

        Ai_g = self._mi.orientation_global  # global cols = global axes of marker_i
        Aj_g = self._mj.orientation_global
        x_i, y_i = Ai_g[:, 0], Ai_g[:, 1]
        z_j = Aj_g[:, 2]

        # ── Body i contribution ───────────────────────────────────────
        blocks: list[JacobianBlock] = []
        if has_dofs(bi):
            J_R_i = np.zeros((5, 3))
            J_p_i = np.zeros((5, 4))
            # Coincidence rows.
            J_R_i[:3, :] = np.eye(3)
            J_p_i[:3, :] = dA_dp(ui, bi.quaternion)
            # Dot-1 rows: ∂(u_a · u_b)/∂p_a = (u_a × u_b)^T · 2 E(p_a).
            twoEi = 2.0 * _q.E(bi.quaternion)
            J_p_i[3, :] = np.cross(x_i, z_j) @ twoEi
            J_p_i[4, :] = np.cross(y_i, z_j) @ twoEi
            blocks.append((bi, J_R_i, J_p_i))

        # ── Body j contribution ───────────────────────────────────────
        if has_dofs(bj):
            J_R_j = np.zeros((5, 3))
            J_p_j = np.zeros((5, 4))
            J_R_j[:3, :] = -np.eye(3)
            J_p_j[:3, :] = -dA_dp(uj_orig, bj.quaternion)
            twoEj = 2.0 * _q.E(bj.quaternion)
            # Dot-1 rows: ∂(u_a · u_b)/∂p_b = -(u_a × u_b)^T · 2 E(p_b).
            J_p_j[3, :] = -np.cross(x_i, z_j) @ twoEj
            J_p_j[4, :] = -np.cross(y_i, z_j) @ twoEj
            blocks.append((bj, J_R_j, J_p_j))

        return blocks

    def gamma(self) -> NDArray[np.float64]:
        r"""Acceleration-level RHS, stacked across the 5 equations.

        For coincidence rows, see
        :meth:`~aladyn.constraints.spherical.SphericalJoint.gamma`.

        For a dot-1 row :math:`\Phi = \mathbf u_a^\mathsf{T} \mathbf u_b`,

        .. math::

            \boldsymbol\gamma = -\Bigl[\,\bigl(\boldsymbol\omega_a \times
            (\boldsymbol\omega_a \times \mathbf u_a)\bigr) \cdot \mathbf u_b
            + 2\,(\boldsymbol\omega_a \times \mathbf u_a) \cdot
            (\boldsymbol\omega_b \times \mathbf u_b)
            + \mathbf u_a \cdot \bigl(\boldsymbol\omega_b \times
            (\boldsymbol\omega_b \times \mathbf u_b)\bigr)\Bigr].
        """
        bi, bj = self._mi.parent, self._mj.parent
        wi, wj = bi.omega_global, bj.omega_global

        # Coincidence rows.
        ui_pt = bi.rotation_matrix @ self._mi.position_local
        uj_pt = bj.rotation_matrix @ self._mj.position_local
        g_coin = -(centripetal(wi, ui_pt) - centripetal(wj, uj_pt))

        # Dot-1 rows.
        x_i = self._mi.orientation_global[:, 0]
        y_i = self._mi.orientation_global[:, 1]
        z_j = self._mj.orientation_global[:, 2]
        g_dot_xz = _gamma_dot1(wi, x_i, wj, z_j)
        g_dot_yz = _gamma_dot1(wi, y_i, wj, z_j)

        return np.array([g_coin[0], g_coin[1], g_coin[2], g_dot_xz, g_dot_yz], dtype=np.float64)


# ─── Module-local helpers ─────────────────────────────────────────────


def _gamma_dot1(
    w_a: NDArray[np.float64],
    u_a: NDArray[np.float64],
    w_b: NDArray[np.float64],
    u_b: NDArray[np.float64],
) -> float:
    r"""γ contribution of the dot-1 constraint :math:`\mathbf u_a \cdot \mathbf u_b`."""
    cwa_ua = np.cross(w_a, u_a)
    cwb_ub = np.cross(w_b, u_b)
    term1 = centripetal(w_a, u_a) @ u_b
    term2 = 2.0 * (cwa_ua @ cwb_ub)
    term3 = u_a @ centripetal(w_b, u_b)
    return -float(term1 + term2 + term3)
