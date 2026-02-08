import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0

class Dipole(MagneticComponent):
    """Magnetic dipole characterized by a magnetic moment vector."""
    def __init__(self, 
            moment=jnp.array([0.0, 0.0, 1.0]), 
            position=jnp.array([0.0, 0.0, 0.0]), 
            rotation_vector=jnp.array([0.0, 0.0, 0.0])
    ):
        """
        :param moment: 3D vector for magnetic dipole moment (A·m^2 in SI units).
        :param position: 3D position of the dipole (m).
        :param quaternion: 4D quaternion of the dipole (as a JAX array)
        """
        super().__init__(moment=moment, position=position, rotation_vector=rotation_vector)
        self.moment = jnp.array(moment, dtype=float)  # magnetic moment vector
    
    def field_at(self, point):
        """
        Compute the magnetic B field (in Tesla) at the given point (array-like of shape (3,))
        due to this dipole, using the analytical dipole field equation.
        """
        
        point = jnp.array(point, dtype=float) # can we just enforce that this is always a jnp array instead of casting? 
        
        r = point - self.position # displacement vector 
        r_norm = jnp.linalg.norm(r)
        r_hat = r / r_norm

        m_dot_rhat = jnp.dot(self.moment, r_hat)

        B = MU_0/(4*jnp.pi*r_norm**3) * (3 * m_dot_rhat * r_hat - self.moment) # in Tesla
        return B
