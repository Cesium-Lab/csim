"""Runnable demo, not a pytest suite: propagates three spacecraft in different orbits
and plots them together using plot_orbit's multi-trajectory support (`r=[sim1.X, sim2.X, ...]`).

All three use the same dt/n_steps so their state arrays line up row-for-row -- required
since the slider walks every trajectory together off one shared time index.

    python examples/multi_spacecraft_demo.py
"""

import sys

sys.path.append(".")

import numpy as np

from csim.constants import DEG_TO_RAD
from csim.entities import Spacecraft
from csim.math import coes_to_rv
from csim.simulation import Simulator, rigid_body_step_fn
from csim.visualization import plot_orbit
from csim.world import MU_EARTH, R_EARTH

# coes_to_rv doesn't implement the exact-circular (e=0) or exact-equatorial (i=0) special
# cases (see csim/math/coes.py), so e and i are nudged just off zero below to route around
# that rather than hit NotImplementedError.

# (name, perigee_alt_m, e, i_deg, raan_deg, aop_deg, ta_deg)
ORBITS = [
    ("LEO near-equatorial", 500e3, 0.001, 0.5, 0.0, 0.0, 0.0),
    ("Polar", 1200e3, 0.001, 90.0, 30.0, 0.0, 0.0),
    ("Inclined elliptical", 800e3, 0.5, 45.0, 60.0, 90.0, 0.0),
]

if __name__ == "__main__":
    dt = 1.0
    t0 = 0.0
    # Long enough to cover a full period of the slowest (highest-apogee) orbit
    slowest_a = max((R_EARTH + r_p) / (1 - e) for _, r_p, e, *_ in ORBITS)
    n_steps = int(2 * np.pi * np.sqrt(slowest_a**3 / MU_EARTH) / dt)

    trajectories = []
    print(f"Propagating {len(ORBITS)} spacecraft, dt={dt}s, {n_steps} steps each:")
    for name, r_p, e, i_deg, raan_deg, aop_deg, ta_deg in ORBITS:
        a = (R_EARTH + r_p) / (1 - e)
        r_vec, v_vec = coes_to_rv(
            a=a, e=e, i=i_deg * DEG_TO_RAD, raan=raan_deg * DEG_TO_RAD,
            aop=aop_deg * DEG_TO_RAD, ta=ta_deg * DEG_TO_RAD, mu=MU_EARTH,
        )
        state0 = np.hstack((r_vec, v_vec, [1, 0, 0, 0], [0, 0, 0]))

        sc = Spacecraft(mass=100, I=np.eye(3))
        sim = Simulator(state0, t0, dt, n_steps, rigid_body_step_fn(dt, sc))
        sim.simulate()

        alt = np.linalg.norm(sim.X[:, :3], axis=1) - R_EARTH
        print(f"  {name}: a={a / 1e3:.0f} km, e={e}, i={i_deg:.0f} deg -- alt {alt.min() / 1e3:.0f} to {alt.max() / 1e3:.0f} km")

        trajectories.append(sim.X)

    t = np.arange(n_steps + 1) * dt
    plot_orbit(trajectories, t=t, downsample_rate=10)
