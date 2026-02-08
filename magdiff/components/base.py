"""Base class for all component types"""

import jax.numpy as jnp
from magdiff.math import quat_normalize
from abc import abstractmethod
from abc import ABC
class MagneticComponent(ABC):
    """Base class for magnetic field sources in the simulation."""

    def __init__(
        self,
        moment=jnp.array([0.0, 0.0, 0.0]), # (3,) array of [mx, my, mz]
        position=jnp.array([0.0, 0.0, 0.0]), # (3,) array of [x, y, z]
        rotation_vector=jnp.array([0.0, 0.0, 0.0]), # (3,) array of [rx, ry, rz]
    ):
        """
        :param moment: 3D vector for magnetic moment (A·m^2 in SI units).
        :param position: 3D position of the magnetic component in 3D space (as a JAX array)
        :param rotation_vector: 3d rotation vector 
        """
        self.position = jnp.array(position, dtype=float)
        self.rotation_vector =jnp.array(rotation_vector, dtype=float)

    @abstractmethod
    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute magnetic B-field vector at a given observation point.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses should implement field_at() for their geometry."
        )
