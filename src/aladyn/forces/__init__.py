"""External / internal forces and torques.

All force objects expose a ``contribute(q, qd, t) -> (Q_R, Q_p)`` returning
their contribution to the generalized force vector for the bodies they act
on. The ``dynamics/eom.py`` module sums them into the global ``Q_e``.
"""
