"""Concrete sensor models, built on the Sensor abstraction in core.py.
State convention throughout: [r(3), v(3), q(4), w(3)] (13,), same as
rigid_body.py -- r,v inertial, q body-to-inertial, w body frame."""

import jax.numpy as jnp
from jax.numpy.linalg import norm

from ..math.quaternion import unit
from .core import Sensor, SensorEnvironment


def accelerometer_sensor(noise_std: float) -> Sensor:
    """Body-frame specific force -- the non-gravitational proper acceleration
    an accelerometer actually reads (it doesn't sense gravity itself)."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del state
        return env.specific_force_body / env.mass

    return Sensor(
        name="accelerometer",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(3) * noise_std**2,
        meas_dim=3,
    )


def gyroscope_sensor(noise_std: float) -> Sensor:
    """Body angular velocity."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return state[10:13]

    return Sensor(
        name="gyroscope",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(3) * noise_std**2,
        meas_dim=3,
    )


def star_tracker_sensor(noise_std: float) -> Sensor:
    """Attitude quaternion (body-to-inertial)."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return unit(state[6:10])

    return Sensor(
        name="star_tracker",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(4) * noise_std**2,
        meas_dim=4,
    )


def range_sensor(noise_std: float) -> Sensor:
    """Distance to a single tracked target (env.target_position). RPO/relative
    nav sensor -- e.g. a laser rangefinder or radar locked onto the target."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        r = state[0:3]
        return jnp.array([norm(r - env.target_position)])

    return Sensor(
        name="range",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(1) * noise_std**2,
        meas_dim=1,
    )


def doppler_sensor(noise_std: float) -> Sensor:
    """Range-rate to a single tracked target (env.target_position/velocity),
    i.e. relative velocity projected onto the line-of-sight direction."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        r, v = state[0:3], state[3:6]
        rel_pos = r - env.target_position
        rel_vel = v - env.target_velocity
        return jnp.array([jnp.dot(rel_vel, rel_pos) / norm(rel_pos)])

    return Sensor(
        name="doppler",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(1) * noise_std**2,
        meas_dim=1,
    )


def gps_sensor(pos_noise_std: float, vel_noise_std: float) -> Sensor:
    """Direct absolute position + velocity (inertial frame), as a real GPS
    receiver reports (position from pseudorange, velocity from carrier-phase
    Doppler -- hence the separate noise stds)."""

    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return state[0:6]

    noise_cov = jnp.diag(
        jnp.concatenate([jnp.full(3, pos_noise_std**2), jnp.full(3, vel_noise_std**2)])
    )

    return Sensor(
        name="gps",
        measurement_fn=meas_fn,
        noise_cov=noise_cov,
        meas_dim=6,
    )
