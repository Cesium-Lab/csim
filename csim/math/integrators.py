import jax.numpy as jnp
from typing import Callable

# @njit
# def euler_func(
#     t: float,
#     dt: float,
#     x_prev: jnp.ndarray,
#     x_dot: Callable[[float, jnp.ndarray], jnp.ndarray],
# ):
#     """Euler's Method using a function for the derivative"""
#     x_n = dt * x_dot(t, x_prev) + x_prev

#     return x_n


def rk4_func(
    t: float,
    dt: float,
    x_prev: float,
    x_dot: Callable[[float, jnp.ndarray], jnp.ndarray],
    params=None,
):
    """Not itself jit-compiled: `x_dot` is a generic Python callable (arbitrary
    ODEs in tests, not just rigid-body dynamics), and jax can't jit through an
    arbitrary function argument without marking it static. Jit the individual
    `x_dot` implementations (see rigid_body.py, gravity.py) instead -- this
    stays a thin, cheap composition of whatever those already-compiled calls
    return."""
    k1 = x_dot(t, x_prev, params)
    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2, params)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2, params)
    k4 = x_dot(t + dt, x_prev + dt * k3, params)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_n
