import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable
from tqdm import tqdm

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

    Can be driven a whole-trajectory-at-once via `simulate()`, or one step at a time via
    `step()` -- the latter is what a closed-loop controller or a multi-body driver (calling
    `step()` on several Simulator instances in turn each tick) should use.
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
            returns True, the simulator stops advancing and (once `finalize()` is
            called, or `simulate()` completes) trims self.X and self.t
        """
        self.t0 = t0
        self.t_curr = t0
        self.dt = dt
        self.n_steps = n_steps
        self.step_fn = step_fn
        self.stop_fn = stop_fn

        n = len(state0)
        self.X = np.zeros((n_steps + 1, n))
        self.t = t0 + dt * np.arange(n_steps + 1)
        self.X[0] = state0

        self.step_idx = 0
        self.done = False

    @property
    def state(self) -> np.ndarray:
        """Current state, i.e. the state at `self.t_curr`."""
        return self.X[self.step_idx]

    def step(self) -> np.ndarray:
        """Advance the simulation by a single dt and return the new state and whether it is done"""
        if self.done:
            return self.state, self.done

        next_state = self.step_fn(self.t_curr, self.state)
        self.step_idx += 1
        self.X[self.step_idx] = next_state
        self.t_curr += self.dt

        if self.step_idx >= self.n_steps or (
            self.stop_fn is not None and self.stop_fn(next_state)
        ):
            self.done = True

        return next_state, self.done

    def finalize(self):
        """Trim self.X and self.t down to the steps actually taken so far.

        Called automatically at the end of `simulate()`; call it yourself after
        driving the simulator manually with `step()` if you want the trimmed arrays.
        """
        self.X = self.X[: self.step_idx + 1]
        self.t = self.t[: self.step_idx + 1]

    def simulate(self):
        with tqdm(total=self.n_steps, desc="Simulating") as pbar:
            while not self.done:
                self.step()
                pbar.update(1)

        self.finalize()


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
    for step in tqdm(range(1, n_steps + 1), desc="Simulating batch"):
        X[step] = batched_step(t_curr, X[step - 1])
        t_curr += dt

    return X, t
