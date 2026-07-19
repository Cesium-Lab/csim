import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable

from ..entities import Spacecraft
from ..math import rk4_func, unit
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
from ..physics import grav_accel

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something


class Simulator:
    """
    Generic fixed-step simulator for any state size and dynamics
    All dynamics-specific details (integrator, transformation, renormalization, etc.) are in
    `step_fn`

    `step_fn` is free to be a jax-jit-compiled function (see rigid_body_step_fn) -- state
    history is still kept in a plain numpy array here, since the sequential Python loop
    driving it isn't itself a jax computation.
    """

    def __init__(
        self,
        state0: np.ndarray,
        t0: float,
        dt: float,
        n_steps: int,
        step_fn: Callable[[float, np.ndarray], np.ndarray],
        stop_fn: Callable[[np.ndarray], bool] = None,
    ):
        """
        stop_fn: optional. Checked against the new state after each step; if it
            returns True, simulate() stops early and trims self.X and self.t
        """
        self.t_curr = t0
        self.dt = dt
        self.n_steps = n_steps
        self.step_fn = step_fn
        self.stop_fn = stop_fn

        n = len(state0)
        self.X = np.zeros((n_steps + 1, n))
        self.t = t0 + dt * np.arange(n_steps + 1)
        self.X[0] = state0

    def simulate(self):
        final_step = self.n_steps
        for step in range(1, self.n_steps + 1):
            self.X[step] = self.step_fn(self.t_curr, self.X[step - 1])
            self.t_curr += self.dt

            if self.stop_fn is not None and self.stop_fn(self.X[step]):
                final_step = step
                break

        self.X = self.X[: final_step + 1]
        self.t = self.t[: final_step + 1]


# Example step functions


def rigid_body_step_fn(
    dt: float, spacecraft: Spacecraft
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Builds the `step_fn(t, state) -> next_state` for Simulator, matching the
    rigid-body-under-gravity-only behavior this file used to hardcode directly.

    `step_fn` itself is jit-compiled, so a whole RK4 step (4 derivative evals +
    gravity + renormalization) is one fused XLA dispatch instead of several
    small ones -- this is also what lets `vmap_rigid_body_step_fn` batch it
    over many spacecraft/trajectories at once."""

    @jax.jit
    def step_fn(t, state):
        accel = grav_accel(state[:3])
        params = RigidBodyParams(
            spacecraft.mass, spacecraft.I, accel * spacecraft.mass, jnp.zeros(3)
        )
        next_state = rk4_func(t, dt, state, rigid_body_derivative, params)
        next_state = next_state.at[6:10].set(unit(next_state[6:10]))
        return next_state

    return step_fn


def vmap_rigid_body_step_fn(
    dt: float, spacecraft: Spacecraft
) -> Callable[[float, jnp.ndarray], jnp.ndarray]:
    """Same dynamics as rigid_body_step_fn, but batched: `state` is (batch, 13)
    instead of (13,), and every row is propagated in parallel on top of the
    same `step_fn` jaxpr (one compile, reused for any batch size). This is the
    scalable path -- run thousands of spacecraft/dispersed initial conditions
    through the same fixed-step propagation for the cost of a handful of
    XLA dispatches instead of a Python loop per trajectory."""

    step_fn = rigid_body_step_fn(dt, spacecraft)
    return jax.jit(jax.vmap(step_fn, in_axes=(None, 0)))


def simulate_batch(
    states0: np.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    spacecraft: Spacecraft,
):
    """Propagate a batch of initial states (batch, 13) forward `n_steps` under
    rigid-body-under-gravity dynamics, all in parallel via vmap.

    Returns:
        X: (n_steps + 1, batch, 13) state history
        t: (n_steps + 1,) time history
    """
    batched_step = vmap_rigid_body_step_fn(dt, spacecraft)

    states0 = np.asarray(states0)
    batch = states0.shape[0]
    X = np.zeros((n_steps + 1, batch, states0.shape[1]))
    t = t0 + dt * np.arange(n_steps + 1)
    X[0] = states0

    t_curr = t0
    for step in range(1, n_steps + 1):
        X[step] = batched_step(t_curr, X[step - 1])
        t_curr += dt

    return X, t
