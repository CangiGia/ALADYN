"""Top-level solver façade and analysis drivers.

Modules
-------
model       : ``SpatialMultibodyModel`` — public façade wiring everything.
kinematics  : position / velocity / acceleration analyses.
statics     : equilibrium analysis.
dynamics    : DAE time integration.
integrators : adapters around scipy and (later) custom integrators.
"""

from .kinematics import AssemblyResult, assemble_position, solve_velocity

__all__ = ["AssemblyResult", "assemble_position", "solve_velocity"]
