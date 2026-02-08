"""Base class for all component types"""

import jax.numpy as jnp
from abc import abstractmethod, ABC


class MagneticComponent(ABC):
    """Base class for magnetic field sources in the simulation.

    Every component has a pose (defined by position and rotation_vector) defined
    relative to its parent frame. If there is no parent frame, the component is considered
    to be the world frame.

    Required implemntation by subclasses:
      - field_at()
      - tree_flatten()
      - tree_unflatten()
      - register as a pytree
    """

    def __init__(
        self,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """
        :param position: 3D position relative to parent frame (m).
        :param rotation_vector: axis-angle rotation vector relative to parent frame (rad).
        :param name: optional human-readable label (not differentiated).
        """
        self.position = jnp.asarray(position, dtype=float)
        self.rotation_vector = jnp.asarray(rotation_vector, dtype=float)
        self.name = name

    @abstractmethod
    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute magnetic B-field vector at a given observation point.

        The point is given in the parent's coordinate frame.
        The returned B vector is also in the parent's coordinate frame.
        """
        raise NotImplementedError(
            "Subclasses should implement field_at() for their geometry."
        )
