import jax
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0


class Cylinder(MagneticComponent):
    """Uniformly magnetized solid cylinder (finite length)."""

    def __init__(
        self,
        magnetization=jnp.zeros(3),
        diameter=1.0,
        height=1.0,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """Create a cylinder magnet.

        Parameters
        ----------
        magnetization : array-like, shape (3,)
            Homogeneous magnetization vector (A/m) in the body frame.
        diameter : float
            Cylinder diameter (m).
        height : float
            Cylinder height (m).
        position : array-like, shape (3,)
            Position of the cylinder centre relative to parent frame (m).
        rotation_vector : array-like, shape (3,)
            Axis-angle rotation vector relative to parent frame (rad).
        name : str, optional
            Human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.magnetization = jnp.asarray(magnetization, dtype=float)
        self.diameter = float(diameter)
        self.height = float(height)

    def field_at(self, point):
        """Magnetic flux-density B at an observation point (dipole approximation)."""
        point = jnp.asarray(point, dtype=float)

        # Dipole moment = magnetization * volume
        volume = jnp.pi * (self.diameter / 2) ** 2 * self.height
        m = self.magnetization * volume

        r = point - self.position
        r_norm = jnp.linalg.norm(r)
        r_hat = r / r_norm

        m_dot_rhat = jnp.dot(m, r_hat)
        B = MU_0 / (4 * jnp.pi * r_norm**3) * (3 * m_dot_rhat * r_hat - m)
        return B

    def tree_flatten(self):
        children = (self.position, self.rotation_vector, self.magnetization)
        aux_data = (self.name, self.diameter, self.height)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, magnetization = children
        name, diameter, height = aux_data
        return cls(
            magnetization=magnetization,
            diameter=diameter,
            height=height,
            position=position,
            rotation_vector=rotation_vector,
            name=name,
        )


jax.tree_util.register_pytree_node_class(Cylinder)
