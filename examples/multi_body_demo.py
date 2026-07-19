"""Runnable demo, not a pytest suite: propagates a high-apogee Earth orbit and
plots it against both Earth and the Moon using plot_orbit's multi-body support
(`body=[...]`).

Note: csim's dynamics are Earth-only (two-body + J2, see csim/physics/gravity.py)
-- there's no lunar gravity here, so this is NOT a real translunar trajectory,
just a big ellipse reaching out to the Moon's distance. It exists to demo the
multi-body plotting, not to model a lunar transfer.

    python examples/multi_body_demo.py
"""

import sys

sys.path.append(".")

import numpy as np

from csim.central_bodies import EARTH, MOON, CentralBody
from csim.entities import Spacecraft
from csim.simulation import Simulator, rigid_body_step_fn
from csim.visualization import plot_orbit
from csim.world import MU_EARTH, R_EARTH

EARTH_MOON_DISTANCE = 384400e3  # m, mean

if __name__ == "__main__":
    # Ellipse from a 300 km-altitude perigee out to the Moon's mean distance
    r_p = R_EARTH + 300e3
    r_a = EARTH_MOON_DISTANCE
    a = (r_p + r_a) / 2
    v_p = np.sqrt(MU_EARTH * (2 / r_p - 1 / a))

    state0 = np.array([r_p, 0, 0, 0, v_p, 0, 1, 0, 0, 0, 0, 0, 0])

    dt = 1
    t0 = 0.0
    T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    n_periods = 2
    n_steps = int(T / dt)  # one full orbit

    print(
        "Propagating a 300km-perigee ellipse out to lunar distance (Earth-only dynamics):"
    )
    print(
        f"  perigee alt: {r_p - R_EARTH:.0f} m, apogee: {r_a / 1e3:.0f} km, period: {T / 3600:.1f} hr"
    )

    sc = Spacecraft(mass=100, I=np.eye(3))
    sim = Simulator(state0, t0, dt, n_steps, rigid_body_step_fn(dt, sc))
    sim.simulate()

    moon_here = CentralBody(
        name="Moon",
        radius=MOON.radius,
        color=MOON.color,
        center=[-EARTH_MOON_DISTANCE, 0, 0],
    )

    plot_orbit(sim.X, t=sim.t, body=[EARTH, moon_here])

    input("Press Enter to exit")
