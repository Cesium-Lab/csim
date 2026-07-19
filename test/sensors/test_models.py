import sys

sys.path.append(".")

import jax.numpy as jnp
import numpy as np

from csim.sensors import (
    SensorEnvironment,
    accelerometer_sensor,
    doppler_sensor,
    gps_sensor,
    gyroscope_sensor,
    range_sensor,
    star_tracker_sensor,
)

zero_state = jnp.zeros(13)


def test_accelerometer():
    sensor = accelerometer_sensor(noise_std=0.1)
    env = SensorEnvironment(mass=3.0, specific_force_body=jnp.array([3.0, 6.0, 9.0]))

    assert np.allclose(sensor.measure(zero_state, env), [1.0, 2.0, 3.0])
    assert np.allclose(sensor.get_noise_cov(), np.eye(3) * 0.01)


def test_gyroscope():
    sensor = gyroscope_sensor(noise_std=0.01)
    state = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.0, 2.0, 3.0])

    assert np.array_equal(sensor.measure(state, SensorEnvironment()), [1.0, 2.0, 3.0])


def test_star_tracker_unitizes():
    sensor = star_tracker_sensor(noise_std=0.001)
    state = jnp.array([0, 0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0])

    assert np.array_equal(sensor.measure(state, SensorEnvironment()), [1, 0, 0, 0])


def test_range():
    sensor = range_sensor(noise_std=1.0)
    state = jnp.array([10.0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    env = SensorEnvironment(target_position=jnp.array([4.0, 0, 0]))

    assert np.allclose(sensor.measure(state, env), [6.0])

    J = sensor.jacobian(state, env)
    expected = np.zeros((1, 13))
    expected[0, 0] = 1.0  # d(range)/dr = unit(r - target) = [1,0,0] here
    assert np.allclose(J, expected)


def test_doppler_closing_target():
    sensor = doppler_sensor(noise_std=0.1)
    # Directly approaching a stationary target at 1 m/s along the LOS
    state = jnp.array([10.0, 0, 0, -1.0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    env = SensorEnvironment(
        target_position=jnp.zeros(3), target_velocity=jnp.zeros(3)
    )

    assert np.allclose(sensor.measure(state, env), [-1.0])


def test_gps():
    sensor = gps_sensor(pos_noise_std=2.0, vel_noise_std=0.05)
    state = jnp.array([1.0, 2, 3, 4, 5, 6, 1, 0, 0, 0, 0, 0, 0])

    assert np.array_equal(
        sensor.measure(state, SensorEnvironment()), [1, 2, 3, 4, 5, 6]
    )

    cov = sensor.get_noise_cov()
    assert np.allclose(np.diag(cov), [4.0, 4.0, 4.0, 0.0025, 0.0025, 0.0025])
