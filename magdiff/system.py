import jax
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from typing import List, Dict
from magdiff.constants import MU_0


class MagneticSystem:
    """ A system comprised of multiple components, allowing for the cumulative calculation of magnetic fields."""
    def __init__(self, components: List[MagneticComponent] = None):
        """
        :param components: list of MagneticComponent objects
        """
        self.components: List[MagneticComponent] = components[:] if components is not None else []
    
    def add_component(self, component: MagneticComponent):
        """Add a MagneticComponent  the system."""
        self.components.append(component)
    
    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total magnetic field at the given point by summing contributions
        from all components in the system.
        """
        point = jnp.array(point, dtype=float)
        # Compute all field contributions, then take the sum
        fields = jnp.stack([comp.field_at(point) for comp in self.components])
        return jnp.sum(fields, axis=0)
    
    def get_parameters(self) -> Dict[str, jnp.ndarray]:
        """
        Extract all optimizable parameters from the system as a flat array.
        Returns a dict with parameter structure and the flat parameter array.
        """
        params = []
        param_structure = []
        
        for i, comp in enumerate(self.components):
            comp_params = {}
            if hasattr(comp, 'position'):
                comp_params['position'] = comp.position
                params.append(comp.position)
            if hasattr(comp, 'moment'):
                comp_params['moment'] = comp.moment
                params.append(comp.moment)
            param_structure.append({
                'component_idx': i,
                'component_type': type(comp).__name__,
                'params': comp_params
            })
        
        return {
            'structure': param_structure,
            'flat_params': jnp.concatenate(params) if params else jnp.array([])
        }
    
    def set_parameters(self, param_dict):
        """
        Set system parameters from a parameter dictionary (from get_parameters).
        """
        flat_params = param_dict['flat_params']
        structure = param_dict['structure']
        
        idx = 0
        for comp_info in structure:
            comp = self.components[comp_info['component_idx']]
            for param_name, param_value in comp_info['params'].items():
                param_size = len(param_value)
                new_value = flat_params[idx:idx + param_size]
                setattr(comp, param_name, new_value)
                idx += param_size
    
    def field_at_with_params(self, point, flat_params):
        """
        Compute field at point with given parameters without mutating the system.
        This is the key method for optimization.
        """
        point = jnp.array(point, dtype=float)
        param_dict = self.get_parameters()
        structure = param_dict['structure']
        
        # Reconstruct parameters from flat array
        idx = 0
        total_B = jnp.zeros(3, dtype=float)
        
        for comp_info in structure:
            comp = self.components[comp_info['component_idx']]
            
            # Extract parameters for this component
            comp_params = {}
            for param_name, param_value in comp_info['params'].items():
                param_size = len(param_value)
                new_value = flat_params[idx:idx + param_size]
                comp_params[param_name] = new_value
                idx += param_size
            
            # Compute field contribution from this component with new parameters
            if hasattr(comp, 'field_at_with_params'):
                # If component has its own field_at_with_params method
                field_contrib = comp.field_at_with_params(point, comp_params)
            else:
                # Otherwise, use a functional approach
                field_contrib = self._compute_component_field(comp, point, comp_params)
            
            total_B = total_B + field_contrib
        
        return total_B
    
    def _compute_component_field(self, component, point, params):
        """
        Compute field contribution from a component with given parameters.
        This is a functional approach that doesn't mutate the component.
        """
        from magdiff.components.dipole import Dipole
        
        if isinstance(component, Dipole):
            return self._compute_dipole_field(point, params)
        else:
            # For other component types, could add similar functional implementations
            raise NotImplementedError(f"Functional field computation not implemented for {type(component)}")
    
    def _compute_dipole_field(self, point, params):
        """
        Functional computation of dipole field without mutating the dipole.
        """
        
        point = jnp.array(point, dtype=float)
        position = params['position']
        moment = params['moment']
        
        # Displacement vector from dipole to observation point:
        r = point - position
        # Distance magnitude (Euclidean norm)
        r_norm = jnp.linalg.norm(r)
        # Prevent division by zero by adding a small epsilon
        eps = 1e-9
        r_norm = jnp.where(r_norm < eps, eps, r_norm)
        # Unit vector from dipole to point
        r_hat = r / r_norm
        # Compute dot product m·r and magnitude factors
        m_dot_r = jnp.dot(moment, r)
        # Using the dipole field formula:
        term1 = 3 * m_dot_r * r_hat / (r_norm**3)
        term2 = moment / (r_norm**3)
        B = MU_0/(4*jnp.pi) * (term1 - term2)  # in Tesla
        return B
    
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
