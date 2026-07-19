import sys

import numpy as np
import pytest

sys.path.append(".")

from csim.simulation import Simulator, rigid_body_step_fn
from csim.entities import Spacecraft
from csim.world import MU_EARTH, R_EARTH


def make_linear_step_fn(dt):
    """Trivial step_fn for a 1D constant-velocity particle: state = [x, v].
    Used to test Simulator's own bookkeeping (stepping, done, stop_fn,
    finalize) without depending on the rigid-body/jax dynamics."""

    def step_fn(t, state):
        x, v = state
        return np.array([x + v * dt, v])

    return step_fn


#########################################################################################################
#               Init
#########################################################################################################


def test_state_property_returns_state0_before_stepping():
    sim = Simulator(np.array([1.0, 2.0]), 0.0, 1.0, 5, make_linear_step_fn(1.0))

    assert sim.state == pytest.approx([1.0, 2.0])
    assert sim.step_idx == 0
    assert not sim.done


#########################################################################################################
#               Stepping
#########################################################################################################


def test_step_advances_state_by_one_dt():
    dt = 0.5
    sim = Simulator(np.array([0.0, 2.0]), 0.0, dt, 5, make_linear_step_fn(dt))

    next_state, done = sim.step()

    assert next_state == pytest.approx([1.0, 2.0])
    assert sim.state == pytest.approx(next_state)
    assert sim.t_curr == pytest.approx(dt)
    assert sim.step_idx == 1
    assert not done
    assert not sim.done


#########################################################################################################
#               Stepping and Simulate
#########################################################################################################


def test_manual_stepping_matches_simulate():
    dt = 0.1
    n_steps = 10
    state0 = np.array([0.0, 3.0])

    # Full propagation
    sim_auto = Simulator(state0, 0.0, dt, n_steps, make_linear_step_fn(dt))
    sim_auto.simulate()

    # Manual stepping
    sim_manual = Simulator(state0, 0.0, dt, n_steps, make_linear_step_fn(dt))
    for _ in range(n_steps):
        sim_manual.step()
    sim_manual.finalize()

    assert sim_manual.X == pytest.approx(sim_auto.X)
    assert sim_manual.t == pytest.approx(sim_auto.t)
    assert sim_manual.X.shape == (n_steps + 1, 2)


def test_done_flag_set_after_n_steps_and_step_becomes_no_op():
    dt = 1.0
    n_steps = 3
    sim = Simulator(np.array([0.0, 1.0]), 0.0, dt, n_steps, make_linear_step_fn(dt))

    for _ in range(n_steps):
        assert not sim.done
        sim.step()

    assert sim.done
    state_at_done = sim.state.copy()

    result, done = sim.step()
    assert result == pytest.approx(state_at_done)
    assert done
    assert sim.done

    # further calls to step() should be no-ops once the simulator is done
    assert sim.state == pytest.approx(state_at_done)
    assert sim.step_idx == n_steps
    assert done
    assert sim.done


def test_stop_fn_stops_simulation_early_and_finalize_trims_arrays():
    dt = 1.0
    n_steps = 100

    def stop_fn(state):
        return state[0] >= 3.0

    sim = Simulator(
        np.array([0.0, 1.0]), 0.0, dt, n_steps, make_linear_step_fn(dt), stop_fn=stop_fn
    )
    sim.simulate()

    # x crosses the threshold exactly at t=3 (3 steps in)
    assert sim.done
    assert sim.X.shape[0] == 4  # state0 + 3 steps
    assert sim.X[-1, 0] == pytest.approx(3.0)
    assert sim.t[-1] == pytest.approx(3.0)


def test_stop_fn_also_respected_when_stepping_manually():
    dt = 1.0
    n_steps = 100

    def stop_fn(state):
        return state[0] >= 3.0

    sim = Simulator(
        np.array([0.0, 1.0]), 0.0, dt, n_steps, make_linear_step_fn(dt), stop_fn=stop_fn
    )

    steps_taken = 0
    while not sim.done:
        sim.step()
        steps_taken += 1

    assert steps_taken == 3
    assert sim.step_idx == 3

    sim.finalize()
    assert sim.X.shape == (4, 2)


#########################################################################################################
#               Finalize
#########################################################################################################


def test_finalize_trims_arrays_to_steps_actually_taken():
    dt = 1.0
    n_steps = 10
    sim = Simulator(np.array([0.0, 1.0]), 0.0, dt, n_steps, make_linear_step_fn(dt))

    for _ in range(4):
        sim.step()

    # before finalize, the pre-allocated arrays are still full-length
    assert sim.X.shape == (n_steps + 1, 2)
    assert sim.t.shape == (n_steps + 1,)

    sim.finalize()

    assert sim.X.shape == (5, 2)
    assert sim.t.shape == (5,)
    assert sim.X == pytest.approx(
        np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    )
    assert sim.t == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_finalize_with_no_steps_taken_keeps_only_initial_state():
    sim = Simulator(np.array([2.0, -1.0]), 0.0, 1.0, 10, make_linear_step_fn(1.0))

    sim.finalize()

    assert sim.X.shape == (1, 2)
    assert sim.t.shape == (1,)
    assert sim.X[0] == pytest.approx([2.0, -1.0])
    assert sim.t[0] == pytest.approx(0.0)


def test_finalize_after_full_run_is_a_no_op():
    dt = 1.0
    n_steps = 5
    sim = Simulator(np.array([0.0, 1.0]), 0.0, dt, n_steps, make_linear_step_fn(dt))
    sim.simulate()

    X_before = sim.X.copy()
    t_before = sim.t.copy()

    # calling finalize() again after it's already been trimmed shouldn't change anything
    sim.finalize()

    assert sim.X == pytest.approx(X_before)
    assert sim.t == pytest.approx(t_before)
    assert sim.X.shape == (n_steps + 1, 2)


#########################################################################################################
#               Multiple Bodies
#########################################################################################################


def test_interleaved_stepping_supports_multiple_bodies():
    """Two independent Simulators stepped in lockstep -- the way a multi-body
    driver would call .step() on each body in turn every tick -- must produce
    the same trajectories as running each one through on its own."""
    dt = 0.2
    n_steps = 15

    state0_a = np.array([0.0, 1.0])
    state0_b = np.array([5.0, -2.0])

    sim_a = Simulator(state0_a, 0.0, dt, n_steps, make_linear_step_fn(dt))
    sim_b = Simulator(state0_b, 0.0, dt, n_steps, make_linear_step_fn(dt))

    for _ in range(n_steps):
        sim_a.step()
        sim_b.step()
    sim_a.finalize()
    sim_b.finalize()

    ref_a = Simulator(state0_a, 0.0, dt, n_steps, make_linear_step_fn(dt))
    ref_a.simulate()
    ref_b = Simulator(state0_b, 0.0, dt, n_steps, make_linear_step_fn(dt))
    ref_b.simulate()

    assert sim_a.X == pytest.approx(ref_a.X)
    assert sim_b.X == pytest.approx(ref_b.X)


def test_manual_stepping_matches_simulate_with_rigid_body_dynamics():
    """Same equivalence check as test_manual_stepping_matches_simulate, but
    against the real jax-jitted rigid_body_step_fn/gravity dynamics instead of
    the trivial linear one, to make sure the refactor didn't change behavior
    for the actual physics path."""
    dt = 1.0
    n_steps = 50
    r = R_EARTH + 500e3
    v = np.sqrt(MU_EARTH / r)
    state0 = np.array([r, 0, 0, 0, v, 0, 1, 0, 0, 0, 0, 0, 0])
    sc = Spacecraft(100, np.diag([2.0, 2.0, 2.0]))

    sim_auto = Simulator(state0, 0.0, dt, n_steps, rigid_body_step_fn(dt, sc))
    sim_auto.simulate()

    sim_manual = Simulator(state0, 0.0, dt, n_steps, rigid_body_step_fn(dt, sc))
    for _ in range(n_steps):
        sim_manual.step()
    sim_manual.finalize()

    assert sim_manual.X == pytest.approx(sim_auto.X)
    assert sim_manual.t == pytest.approx(sim_auto.t)
