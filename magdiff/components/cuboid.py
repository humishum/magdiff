import numpy as np
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU0

class Cuboid(MagneticComponent):
    """Uniformly magnetized cuboid (rectangular prism).
    """

    def __init__(self, magnetization=jnp.array([0.0, 0.0, 0.0]),
                 dimension=jnp.array([1.0, 1.0, 1.0]),
                 position=jnp.array([0.0, 0.0, 0.0])):
        """Create a cuboid magnet.

        Parameters
        ----------
        magnetization : array-like, shape (3,)
            Homogeneous magnetization vector (A/m) in the **world** coordinate frame.
        dimension : array-like, shape (3,)
            Side lengths (Lx, Ly, Lz) of the cuboid in metres.
        position : array-like, shape (3,)
            Position of the cuboid centre in metres.
        """
        super().__init__(position=position)
        self.magnetization = jnp.array(magnetization, dtype=float)
        self.dimension = jnp.array(dimension, dtype=float)

        # Pre-compute the total dipole moment
        V = jnp.prod(self.dimension)  # volume in m³
        self._dipole_moment = self.magnetization * V

    def field_at(self, point):
        """
        """
        point = jnp.array(point, dtype=float)
        r = point - self.position
        r_norm = jnp.linalg.norm(r)
        eps = 1e-12
        r_norm = jnp.where(r_norm < eps, eps, r_norm)
        r_hat = r / r_norm
        m = self._dipole_moment
        m_dot_r = jnp.dot(m, r)
        term1 = 3 * m_dot_r * r_hat / (r_norm**3)
        term2 = m / (r_norm**3)
        B = MU0 / (4 * jnp.pi) * (term1 - term2)
        return B
