import numpy as np  # Only for constants like mu0 if needed
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent

class Cylinder(MagneticComponent):
    """Placeholder for a cylinder magnet."""
    def __init__(self, magnetization=jnp.array([0.0, 0.0, 0.0]), diameter=1.0, height=1.0, position=jnp.array([0.0,0.0,0.0])):
        super().__init__(position=position)
        self.magnetization = jnp.array(magnetization, dtype=float)  # M vector (A/m)
        self.diameter = float(diameter); self.height = float(height)
    def field_at(self, point):
        # In a full implementation, this would use an analytical or numerical solution for a cylinder.
        raise NotImplementedError("Cylinder field calculation not implemented yet.")
