import sys

sys.path.append(".")

import jax.numpy as jnp
import numpy as np
import pytest

from csim.sensors import Sensor, SensorEnvironment, SensorSuite


def double_position_sensor() -> Sensor:
    """Trivial synthetic sensor for exercising the Sensor/SensorSuite API in
    isolation from any real physical model: measures 2*r."""

    def meas_fn(state, env: SensorEnvironment):
        del env
        return 2.0 * state[0:3]

    return Sensor(
        name="double_position",
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(3) * 0.5,
        meas_dim=3,
    )


def test_measure():
    sensor = double_position_sensor()
    state = jnp.array([1.0, 2.0, 3.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    env = SensorEnvironment()

    assert np.array_equal(sensor.measure(state, env), [2.0, 4.0, 6.0])


def test_jacobian():
    sensor = double_position_sensor()
    state = jnp.array([1.0, 2.0, 3.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    env = SensorEnvironment()

    J = sensor.jacobian(state, env)
    expected = np.zeros((3, 13))
    expected[0, 0] = expected[1, 1] = expected[2, 2] = 2.0

    assert J.shape == (3, 13)
    assert np.allclose(J, expected)


def test_get_noise_cov_array():
    sensor = double_position_sensor()
    assert np.array_equal(sensor.get_noise_cov(), np.eye(3) * 0.5)


def test_get_noise_cov_callable():
    sensor = Sensor(
        name="env_dependent",
        measurement_fn=lambda state, env: state[0:1],
        noise_cov=lambda env: jnp.eye(1) * env.mass,
        meas_dim=1,
    )
    env = SensorEnvironment(mass=4.0)
    assert np.array_equal(sensor.get_noise_cov(env), [[4.0]])


class TestSensorSuite:
    def test_dispatches_by_name(self):
        suite = SensorSuite(sensors={"double_position": double_position_sensor()})
        state = jnp.array([1.0, 2.0, 3.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        env = SensorEnvironment()

        assert np.array_equal(
            suite.measure(state, env, "double_position"), [2.0, 4.0, 6.0]
        )
        assert suite.jacobian(state, env, "double_position").shape == (3, 13)
        assert np.array_equal(
            suite.get_noise_cov("double_position"), np.eye(3) * 0.5
        )

    def test_unknown_sensor_raises(self):
        suite = SensorSuite(sensors={"double_position": double_position_sensor()})
        with pytest.raises(KeyError):
            suite.measure(jnp.zeros(13), SensorEnvironment(), "nonexistent")
