from __future__ import annotations
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from ..math.quaternion import hamilton_product


@jax.tree_util.register_dataclass
@dataclass
class RigidBodyParams:
    """All fields are jax data fields (traceable/vmappable)"""

    mass_kg: float
    I: jnp.ndarray
    force_N: jnp.ndarray
    torque_Nm: jnp.ndarray


@jax.jit
def rigid_body_derivative(t: float, state: jnp.ndarray, params: RigidBodyParams):
    state = jnp.asarray(state)
    v = state[3:6]
    q = state[6:10]
    w = state[10:13]

    # Position derivative is velocity
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = jnp.asarray(params.force_N) / params.mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    dqdt = 0.5 * hamilton_product(q, w)

    # Angular derivative based on (Schaub 4.34-35)
    I = params.I
    I_inv = jnp.linalg.inv(I)
    torque = jnp.asarray(params.torque_Nm)

    # τ = parameters.torque_body
    dwdt = I_inv @ (torque - jnp.cross(w, I @ w))

    return jnp.concatenate((drdt, dvdt, dqdt, dwdt))
