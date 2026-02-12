"""Hall-effect sensor models."""

import jax
import jax.numpy as jnp

from magdiff.sensors.base import MagneticSensor
from magdiff.system import MagneticSystem


class HallSensor(MagneticSensor):
    """Pose-aware Hall-effect sensor.

    Default behavior is single-axis readout with optional gain, bias, noise,
    and clipping. A Hall sensor observes B and does not generate B.
    """

    def __init__(
        self,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        axis_body=jnp.array([0.0, 0.0, 1.0]),
        sensitivity=1.0,
        bias=0.0,
        noise_std=0.0,
        clip_limits: tuple[float, float] = None,
        name: str = None,
    ):
        """
        :param position: 3D sensor position in parent/world frame (m).
        :param rotation_vector: axis-angle sensor orientation (rad).
        :param axis_body: sensing axis in sensor body frame (3,).
        :param sensitivity: linear gain applied to projected field.
        :param bias: additive offset after gain.
        :param noise_std: Gaussian noise standard deviation in output units.
        :param clip_limits: optional (min, max) clip limits for output.
        :param name: optional human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        axis_body = jnp.asarray(axis_body, dtype=float)
        self.axis_body = axis_body / jnp.maximum(jnp.linalg.norm(axis_body), 1e-12)
        self.sensitivity = jnp.asarray(sensitivity, dtype=float)
        self.bias = jnp.asarray(bias, dtype=float)
        self.noise_std = jnp.asarray(noise_std, dtype=float)
        self.clip_limits = clip_limits

    def measure_from_field(self, b_world: jnp.ndarray, key=None) -> jnp.ndarray:
        """Measure Hall output from world-frame B vector(s).

        B = B_at_point * sensitivity + bias + noise # saturated to clip_limits

        :param b_world: shape (3,) or (N, 3)
        :param key: optional PRNG key for additive Gaussian noise.
        :return: scalar for (3,) input, or (N,) for (N, 3) input.
        """
        b_world = jnp.asarray(b_world, dtype=float)
        axis_world = self.axis_world(self.axis_body)

        if b_world.ndim == 1:
            if b_world.shape[0] != 3:
                raise ValueError(
                    "Single-field input must have shape (3,), "
                    f"got {tuple(b_world.shape)}"
                )
            projected = jnp.dot(b_world, axis_world)
        elif b_world.ndim == 2 and b_world.shape[1] == 3:
            projected = b_world @ axis_world
        else:
            raise ValueError(
                f"b_world must have shape (3,) or (N, 3), got {tuple(b_world.shape)}"
            )

        output = self.sensitivity * projected + self.bias

        if key is not None and float(self.noise_std) > 0.0:
            noise = self.noise_std * jax.random.normal(key, shape=output.shape)
            output = output + noise

        if self.clip_limits is not None:
            lo, hi = self.clip_limits
            output = jnp.clip(output, lo, hi)

        return output

    def measure_from_system(self, system: MagneticSystem, key=None) -> jnp.ndarray:
        """Measure sensor output from a MagneticSystem at sensor pose."""
        b_world = system.field_at(self.position)
        return self.measure_from_field(b_world, key=key)

    def measure_from_system_points(
        self, system: MagneticSystem, points: jnp.ndarray, key=None
    ) -> jnp.ndarray:
        """Measure sensor output from fields sampled at points.

        :param system: MagneticSystem-like object with field_at_points.
        :param points: observation points (N, 3)
        :param key: optional PRNG key for additive Gaussian noise.
        :return: sensor outputs (N,)
        """
        points = jnp.asarray(points, dtype=float)
        b_world = system.field_at_points(points)
        return self.measure_from_field(b_world, key=key)
