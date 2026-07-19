"""
TODO:
- Normal J1 gravity
- J2, J3 even
- gravity gradient
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

from ..world import MU_EARTH, R_EARTH, J2_EARTH


@partial(jax.jit, static_argnames=("use_j2",))
def grav_accel(
    r: jnp.ndarray, mu: float = MU_EARTH, *, use_j2=True, r_cb: float = R_EARTH
):
    """Gravity acceleration \\
    Vallado 4e Equation 1-14 p. 23 np.array([axy*x, axy*y, az*z]) \\


    Could not for the life of me find the J2 formula in Vallado, but found in [Poliastro docs](https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html) and in [this paper](https://ntrs.nasa.gov/api/citations/20040031507/downloads/20040031507.pdf)


    Args:
        r (jnp.ndarray): Position from center of center body [m/s]
        mu (float, optional): Gravitational parameter [m3/s2] Defaults to MU_EARTH.
        use_j2 (bool, optional): Whether to use J2. Defaults to True. (static arg)
        r_cb (float, optional): (FOR J2) Radius of central body [m]. Defaults to R_EARTH.

    Returns:
        jnp.ndarray: Gravitational acceleration [m/s2]
    """
    r = jnp.asarray(r)
    r_norm = norm(r)

    # Prevents singularities (also inf * 0 == nan)
    singular = jnp.any(jnp.isinf(r)) | (jnp.abs(r_norm) < 1e-6)
    safe_r_norm = jnp.where(singular, 1.0, r_norm)

    # J1
    g = -mu / (safe_r_norm**3) * r

    # J2
    if use_j2:
        x, y, z = r
        coeff = 3 / 2 * J2_EARTH * mu * r_cb**2 / (2 * safe_r_norm**5)
        axy = coeff * ((5 * z * z) / (safe_r_norm * safe_r_norm) - 1)
        az = coeff * ((5 * z * z) / (safe_r_norm * safe_r_norm) - 3)
        g = g + jnp.array([axy * x, axy * y, az * z])

    return jnp.where(singular, jnp.zeros(3), g)


# TODO: test?
