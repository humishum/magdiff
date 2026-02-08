from typing import Tuple

import jax
import jax.numpy as jnp

from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0
from magdiff.math import rotate_vector


class Dipole(MagneticComponent):
    """
    Magnetic dipole characterized by a magnetic moment vector.

    The moment is specified in the component's body frame. If rotation_vector
    is non-zero, the moment is rotated to the parent frame before computing
    the field.
    """

    def __init__(
        self,
        moment=jnp.array([0.0, 0.0, 1.0]),
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """
        :param moment: 3D magnetic dipole moment vector in body frame (A·m²).
        :param position: 3D position relative to parent frame (m).
        :param rotation_vector: axis-angle rotation vector relative to parent frame (rad).
        :param name: optional human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.moment = jnp.asarray(moment, dtype=float)

    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the magnetic B field (in Tesla) at the given point in the parent frame.

        The stored moment is in the body frame. It is rotated into the parent
        frame before computing the field.
        """

        point = jnp.asarray(point, dtype=float)

        # Rotate moment from body frame to parent frame
        m = rotate_vector(self.rotation_vector, self.moment)

        r = point - self.position
        r_norm = jnp.linalg.norm(r)
        r_hat = r / r_norm

        m_dot_rhat = jnp.dot(m, r_hat)

        B = MU_0 / (4 * jnp.pi * r_norm**3) * (3 * m_dot_rhat * r_hat - m)  # in Tesla
        return B

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[str]]:
        children = (self.position, self.rotation_vector, self.moment)
        aux_data = (self.name,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, moment = children
        (name,) = aux_data
        obj = object.__new__(cls)
        obj.position = position
        obj.rotation_vector = rotation_vector
        obj.moment = moment
        obj.name = name
        return obj


jax.tree_util.register_pytree_node_class(Dipole)
