"""Sensor abstraction: a measurement function + noise + (via autodiff) its
Jacobian, the reusable unit sensor models in models.py are built from."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp


@dataclass
class SensorEnvironment:
    """Time-varying/environmental data measurement functions may need, kept
    separate from `state` since it isn't part of the propagated dynamics.
    Extend with more fields (ephemerides, terrain, etc.) as new sensors need
    them."""

    t: float = 0.0
    mass: float = 1.0
    specific_force_body: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3))
    """Non-gravitational specific force in the body frame [m/s2] -- what an
    accelerometer actually reads (it doesn't sense gravity)."""
    target_position: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3))
    """Absolute position of a tracked target (e.g. RPO target) [m]."""
    target_velocity: jnp.ndarray = field(default_factory=lambda: jnp.zeros(3))
    """Absolute velocity of a tracked target [m/s]."""


@dataclass
class Sensor:
    name: str
    measurement_fn: Callable[[jnp.ndarray, SensorEnvironment], jnp.ndarray]
    noise_cov: jnp.ndarray | Callable[[SensorEnvironment], jnp.ndarray]
    meas_dim: int

    def measure(self, state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        """Compute measurement without noise."""
        return self.measurement_fn(state, env)

    def jacobian(self, state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        """Measurement Jacobian w.r.t. state (env held fixed)."""
        return jax.jacfwd(lambda s: self.measurement_fn(s, env))(state)

    def get_noise_cov(self, env: Optional[SensorEnvironment] = None) -> jnp.ndarray:
        if isinstance(self.noise_cov, jnp.ndarray):
            return self.noise_cov
        return self.noise_cov(env) if env is not None else self.noise_cov()


@dataclass
class SensorSuite:
    """Container for multiple sensors, keyed by name."""

    sensors: Dict[str, Sensor]

    def measure(self, state: jnp.ndarray, env: SensorEnvironment, sensor_name: str) -> jnp.ndarray:
        return self._get(sensor_name).measure(state, env)

    def jacobian(self, state: jnp.ndarray, env: SensorEnvironment, sensor_name: str) -> jnp.ndarray:
        return self._get(sensor_name).jacobian(state, env)

    def get_noise_cov(
        self, sensor_name: str, env: Optional[SensorEnvironment] = None
    ) -> jnp.ndarray:
        return self._get(sensor_name).get_noise_cov(env)

    def _get(self, sensor_name: str) -> Sensor:
        if sensor_name not in self.sensors:
            raise KeyError(
                f"Sensor '{sensor_name}' not found in suite. Available: {list(self.sensors.keys())}"
            )
        return self.sensors[sensor_name]
