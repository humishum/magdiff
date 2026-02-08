import numpy as np  # Only for constants like mu0 if needed
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU0

class Cylinder(MagneticComponent):
    """Uniformly magnetized solid cylinder (finite length).
    """

    def __init__(self, magnetization=jnp.array([0.0, 0.0, 0.0]), diameter=1.0,
                 height=1.0, position=jnp.array([0.0, 0.0, 0.0])):
        """Create a cylinder magnet.

        Parameters
        ----------
        magnetization : array-like, shape (3,)
            Homogeneous magnetization vector (A/m) in the **world** coordinate frame.
        diameter : float
            Cylinder diameter (m).
        height : float
            Cylinder height (m).
        position : array-like, shape (3,)
            Position of the cylinder centre in metres.
        """
        super().__init__(position=position)
        self.magnetization = jnp.array(magnetization, dtype=float)
        self.diameter = float(diameter)
        self.height = float(height)

        # Pre-compute dipole moment for fallback
        volume = np.pi * (self.diameter / 2) ** 2 * self.height
        self._dipole_moment = self.magnetization * volume

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def field_at(self, point):
        """Magnetic flux-density **B** at an arbitrary observation point."""
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
        B = MU0 / (4 * np.pi) * (term1 - term2)
        return B
