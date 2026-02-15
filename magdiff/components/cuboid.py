import jax
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0


class Cuboid(MagneticComponent):
    """Uniformly magnetized cuboid (rectangular prism)."""

    def __init__(
        self,
        magnetization=jnp.zeros(3),
        dimension=jnp.array([1.0, 1.0, 1.0]),
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """Create a cuboid magnet.

        Parameters
        ----------
        magnetization : array-like, shape (3,)
            Homogeneous magnetization vector (A/m) in the body frame.
        dimension : array-like, shape (3,)
            Side lengths (Lx, Ly, Lz) of the cuboid in meters.
        position : array-like, shape (3,)
            Position of the cuboid centre relative to parent frame (m).
        rotation_vector : array-like, shape (3,)
            Axis-angle rotation vector relative to parent frame (rad).
        name : str, optional
            Human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.magnetization = jnp.asarray(magnetization, dtype=float)
        self.dimension = jnp.asarray(dimension, dtype=float)

    def field_at(self, point):
        """Magnetic flux-density B at an observation point (dipole approximation)."""
        point = jnp.asarray(point, dtype=float)

        # Dipole moment = magnetization * volume
        V = jnp.prod(self.dimension)
        m = self.magnetization * V

        r = point - self.position
        r_norm = jnp.linalg.norm(r)
        r_hat = r / r_norm

        m_dot_rhat = jnp.dot(m, r_hat)
        B = MU_0 / (4 * jnp.pi * r_norm**3) * (3 * m_dot_rhat * r_hat - m)
        return B

    def tree_flatten(self):
        children = (
            self.position,
            self.rotation_vector,
            self.magnetization,
            self.dimension,
        )
        aux_data = (self.name,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, magnetization, dimension = children
        (name,) = aux_data
        obj = object.__new__(cls)
        obj.position = position
        obj.rotation_vector = rotation_vector
        obj.magnetization = magnetization
        obj.dimension = dimension
        obj.name = name
        return obj


jax.tree_util.register_pytree_node_class(Cuboid)
