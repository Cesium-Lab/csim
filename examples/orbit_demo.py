"""Runnable demo, not a pytest suite: propagates an eccentric orbit with csim's
rigid-body dynamics and renders it against Earth using the new sphere-rendering
central_bodies module.

    python examples/orbit_demo.py
"""

import sys
import time

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np

from csim.constants import DEG_TO_RAD
from csim.entities import Spacecraft
from csim.math import coes_to_rv
from csim.physics.energy import calc_total_energy
from csim.simulation import Simulator, rigid_body_step_fn
from csim.visualization import plot_orbit
from csim.world import MU_EARTH, R_EARTH

if __name__ == "__main__":
    # Molniya-like orbit: eccentric, high inclination
    a, e, i, raan, aop, ta = (
        26600e3,
        0.74,
        63.4 * DEG_TO_RAD,
        0.0,
        270 * DEG_TO_RAD,
        0.0,
    )
    r_vec, v_vec = coes_to_rv(a=a, e=e, i=i, raan=raan, aop=aop, ta=ta, mu=MU_EARTH)
    state0 = np.hstack((r_vec, v_vec, [1, 0, 0, 0], [0, 0, 0]))

    dt = 0.1
    t0 = 0.0
    T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    n_periods = 10
    n_steps = int(n_periods * T / dt)

    print("Simulating a Molniya-like orbit around Earth:")
    print(
        f"  a={a / 1e3:.0f} km, e={e}, i={i / DEG_TO_RAD:.1f} deg, raan={raan / DEG_TO_RAD:.1f} deg, aop={aop / DEG_TO_RAD:.1f} deg, ta={ta / DEG_TO_RAD:.1f} deg"
    )
    print(
        f"  orbital period: {T / 60:.1f} min -> propagating {n_periods} periods ({n_steps} steps at dt={dt}s)"
    )

    sc = Spacecraft(mass=100, I=np.eye(3))
    sim = Simulator(state0, t0, dt, n_steps, rigid_body_step_fn(dt, sc))

    start = time.perf_counter()
    sim.simulate()
    elapsed = time.perf_counter() - start

    print(f"done in {elapsed:.2f}s wall-clock ({len(sim.X) / elapsed:.0f} steps/s)")
    print(f"propagated {sim.t[-1] / 60:.1f} min of sim time, {len(sim.X)} steps")
    print(
        f"perigee alt: {np.min(np.linalg.norm(sim.X[:, :3], axis=1)) - R_EARTH:.0f} m"
    )
    print(
        f"apogee alt:  {np.max(np.linalg.norm(sim.X[:, :3], axis=1)) - R_EARTH:.0f} m"
    )

    # Energy check: two-body + rigid-body dynamics conserve total energy, so this should
    # stay flat. Drift indicates integration error (too-large dt) or a dynamics bug.
    energy = np.array(
        [
            calc_total_energy(
                sc.mass, sc.I, state[0:3], state[3:6], state[10:13], MU_EARTH
            )
            for state in sim.X
        ]
    )
    drift_pct = (energy - energy[0]) / abs(energy[0]) * 100
    print(
        f"energy drift: {drift_pct.min():.2e}% to {drift_pct.max():.2e}% of initial energy"
    )

    plt.figure(figsize=(10, 4))
    plt.plot(sim.t / 60, energy)
    plt.xlabel("Time (min)")
    plt.ylabel("Total energy (J)")
    plt.title("Energy check (should be flat)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plot_orbit(sim.X, t=sim.t, downsample_rate=10)

    input("Press Enter to exit")
