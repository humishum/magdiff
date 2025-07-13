import jax
import jax.numpy as jnp

class MagneticComponent:
    """Base class for magnetic field sources in the simulation."""
    def __init__(self, position=jnp.array([0.0, 0.0, 0.0])):
        # Position of the magnetic component in 3D space (as a JAX array)
        self.position = jnp.array(position, dtype=float)
    
    def field_at(self, point):
        """
        Compute magnetic B-field vector at a given observation point.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement field_at() for their geometry.")
