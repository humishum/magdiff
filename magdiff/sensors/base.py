"""Base classes for magnetic sensors."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from magdiff.math import rotate_vector


class MagneticSensor(ABC):
    """Pose-aware base class for magnetic sensors.

    Sensors are observers of magnetic fields and are not field sources.
    """

    def __init__(
        self,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """
        :param position: 3D position in the parent/world frame (m).
        :param rotation_vector: axis-angle orientation in parent/world frame (rad).
        :param name: optional human-readable label.
        """
        self.position = jnp.asarray(position, dtype=float)
        self.rotation_vector = jnp.asarray(rotation_vector, dtype=float)
        self.name = name

    def axis_world(self, axis_body=jnp.array([0.0, 0.0, 1.0])) -> jnp.ndarray:
        """Return a body-frame axis rotated into the world frame."""
        axis_body = jnp.asarray(axis_body, dtype=float)
        axis_body = axis_body / jnp.maximum(jnp.linalg.norm(axis_body), 1e-12)
        return rotate_vector(self.rotation_vector, axis_body)

    @abstractmethod
    def measure_from_field(self, b_world: jnp.ndarray, key=None) -> jnp.ndarray:
        """Measure from a provided world-frame field vector/vector batch."""
        raise NotImplementedError
