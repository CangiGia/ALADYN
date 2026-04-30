"""Model entities: ``Ground``, ``RigidBody``, ``Marker``, shapes.

A *body* in ALADYN owns 7 generalized coordinates ``q = [R^T, p^T]^T`` with
``R ∈ ℝ^3`` and ``p`` a unit quaternion. ``Marker`` is a body-fixed frame
(point + orientation) used to attach joints and forces.
"""
