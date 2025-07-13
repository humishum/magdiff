import numpy as np  # Only for constants like mu0 if needed
import jax
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent

class MagneticSystem:
    """A system of multiple magnetic components (dipoles, etc.)."""
    def __init__(self, components=None):
        """
        :param components: list of MagneticComponent objects
        """
        self.components = components[:] if components is not None else []
    
    def add_component(self, component):
        """Add a MagneticComponent (Dipole, Cuboid, etc.) to the system."""
        self.components.append(component)
    
    def field_at(self, point):
        """
        Compute the total magnetic field at the given point by summing contributions
        from all components in the system.
        """
        point = jnp.array(point, dtype=float)
        # Sum up fields from each component
        total_B = jnp.zeros(3, dtype=float)
        for comp in self.components:
            total_B = total_B + comp.field_at(point)
        return total_B
    
    def field_on_grid(self, x_range, y_range, z_range, grid_shape):
        """
        Compute the magnetic field on a 3D grid of points specified by ranges and shape.
        :param x_range, y_range, z_range: tuples (min, max) for each axis.
        :param grid_shape: tuple (Nx, Ny, Nz) indicating number of points along each axis.
        :return: tuple of (X, Y, Z, Bx, By, Bz) flattened arrays for each grid point.
        """
        # Create grid coordinates
        Nx, Ny, Nz = grid_shape
        xs = jnp.linspace(x_range[0], x_range[1], Nx)
        ys = jnp.linspace(y_range[0], y_range[1], Ny)
        zs = jnp.linspace(z_range[0], z_range[1], Nz)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing='ij')
        # Flatten the grid for ease of computation
        points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)  # shape (Npoints, 3)
        # Vectorized field computation at all points:
        # We use vmap to map field_at over all points for efficiency.
        total_field_fn = jax.vmap(self.field_at, in_axes=(0))
        B_vals = total_field_fn(points)  # shape (Npoints, 3)
        # Split B_vals into components and reshape back to grid shape if needed
        Bx = B_vals[:, 0].reshape(grid_shape)
        By = B_vals[:, 1].reshape(grid_shape)
        Bz = B_vals[:, 2].reshape(grid_shape)
        return X, Y, Z, Bx, By, Bz
