import numpy as np  # Only for constants like mu0 if needed
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent

MU0 = 4 * np.pi * 1e-7  # vacuum permeability (μ0) in T·m/A (≈ 1.2566e-6). 
# We'll include MU0 for completeness, though in field calculation we might omit or set μ0/(4π)=1 for simplicity.

class Dipole(MagneticComponent):
    """Magnetic dipole characterized by a magnetic moment vector."""
    def __init__(self, moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.array([0.0, 0.0, 0.0])):
        """
        :param moment: 3D vector for magnetic dipole moment (A·m^2 in SI units).
        :param position: 3D position of the dipole (m).
        """
        super().__init__(position=position)
        self.moment = jnp.array(moment, dtype=float)  # magnetic moment vector
    
    def field_at(self, point):
        """
        Compute the magnetic B field (in Tesla) at the given point (array-like of shape (3,))
        due to this dipole, using the analytical dipole field equation.
        """
        # Convert input point to JAX array
        point = jnp.array(point, dtype=float)
        # Displacement vector from dipole to observation point:
        r = point - self.position
        # Distance magnitude (Euclidean norm)
        r_norm = jnp.linalg.norm(r)
        # Prevent division by zero by adding a small epsilon (or handle r=0 separately)
        # (In practice, you might handle the field inside the dipole differently, but for this model we avoid r=0.)
        eps = 1e-9
        r_norm = jnp.where(r_norm < eps, eps, r_norm)
        # Unit vector from dipole to point
        r_hat = r / r_norm
        # Dipole moment vector (already in global coords)
        m = self.moment
        # Compute dot product m·r and magnitude factors
        m_dot_r = jnp.dot(m, r)
        # Using the dipole field formula (μ0/4π factor omitted for now for simplicity):
        term1 = 3 * m_dot_r * r_hat / (r_norm**3)
        term2 = m / (r_norm**3)
        B = MU0/(4*np.pi) * (term1 - term2)  # in Tesla
        return B
