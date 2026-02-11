from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp

from magdiff.components.base import MagneticComponent
from magdiff.math import rotvec_to_matrix

@dataclass
class FieldVolume:
    """Structured sampled field volume over a Cartesian box."""

    X: jnp.ndarray
    Y: jnp.ndarray
    Z: jnp.ndarray
    B: jnp.ndarray
    spacing: jnp.ndarray
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

class MagneticSystem(MagneticComponent):
    """A composable system of magnetic components.

    A system is comprised of a collection of magnetic components. Each component is placed
    in reference to the system's coordinate frame. Each system will have it's own pose, relative to the parent system
    . If there is no parent system, the parent will be the world frame.

    For example:
        sys_a = MagneticSystem([dip1, dip2], position=[1,0,0], name="sys_a")
        sys_b = MagneticSystem([dip3, dip4], position=[-1,0,0], name="sys_b")
        machine = MagneticSystem([sys_a, sys_b], position=[0,0,0], name="machine")
    """

    def __init__(
        self,
        components: List[MagneticComponent] = None,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """
        :param components: list of MagneticComponent objects (dipoles, cuboids, or other systems).
        :param position: 3D position relative to parent frame (m).
        :param rotation_vector: axis-angle rotation vector relative to parent frame (rad).
        :param name: optional human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.components: List[MagneticComponent] = (
            components[:] if components is not None else []
        )

    def add_component(self, component: MagneticComponent):
        """Add a MagneticComponent to the system."""
        self.components.append(component)

    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total magnetic field at the given point by summing
        contributions from all components.

        The point is given in the parent frame. The returned B is also
        in the parent frame.
        """
        point = jnp.asarray(point, dtype=float)

        # Transform observation point: parent frame -> this system's local frame
        R = rotvec_to_matrix(self.rotation_vector)
        local_point = R.T @ (point - self.position)

        # Sum children's field contributions (all computed in local frame)
        fields = jnp.stack([comp.field_at(local_point) for comp in self.components])
        local_B = jnp.sum(fields, axis=0)

        # Rotate B back to parent frame
        return R @ local_B

    def field_at_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """Compute B field at multiple observation points (N, 3) -> (N, 3)."""
        return jax.vmap(self.field_at)(points)


    def field_jacobian_at_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """
        Compute field Jacobians.

        :param points: observation points (N, 3)
        :return: Jacobian (3, 3) for single-point input, else (N, 3, 3).
        """
        points = jnp.asarray(points, dtype=float)
        jac_fn = jax.jacrev(self.field_at)
        return jax.vmap(jac_fn)(points)

    def field_jacobian_on_grid(self, x_range, y_range, z_range, grid_shape):
        """
        Compute field Jacobian on a 3D grid.

        :param x_range, y_range, z_range: tuples (min, max) for each axis.
        :param grid_shape: tuple (Nx, Ny, Nz) number of points along each axis.
        :return: tuple (X, Y, Z, J) where J has shape (Nx, Ny, Nz, 3, 3).
        """
        Nx, Ny, Nz = grid_shape
        xs = jnp.linspace(x_range[0], x_range[1], Nx)
        ys = jnp.linspace(y_range[0], y_range[1], Ny)
        zs = jnp.linspace(z_range[0], z_range[1], Nz)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")

        points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        jac_vals = self.field_jacobian_at_points(points).reshape((Nx, Ny, Nz, 3, 3))
        return X, Y, Z, jac_vals


    def sample_field_volume(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        shape: Tuple[int, int, int],
    ) -> FieldVolume:
        """
        Sample the vector field in a box.

        :param bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        :param shape: (Nx, Ny, Nz) number of sample points per axis.
        :return: FieldVolume with X/Y/Z grids and B shaped (Nx, Ny, Nz, 3).
        """
        (x_range, y_range, z_range) = bounds
        Nx, Ny, Nz = shape

        xs = jnp.linspace(x_range[0], x_range[1], Nx)
        ys = jnp.linspace(y_range[0], y_range[1], Ny)
        zs = jnp.linspace(z_range[0], z_range[1], Nz)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")

        points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        B_vals = self.field_at_points(points).reshape((Nx, Ny, Nz, 3))

        spacing = jnp.asarray(
            [
                (x_range[1] - x_range[0]) / (Nx - 1) if Nx > 1 else 0.0,
                (y_range[1] - y_range[0]) / (Ny - 1) if Ny > 1 else 0.0,
                (z_range[1] - z_range[0]) / (Nz - 1) if Nz > 1 else 0.0,
            ],
            dtype=float,
        )

        return FieldVolume(
            X=X, Y=Y, Z=Z, B=B_vals, spacing=spacing, bounds=bounds
        )

    def field_on_grid(self, x_range, y_range, z_range, grid_shape):
        """
        Compute the magnetic field on a 3D grid of points.

        :param x_range, y_range, z_range: tuples (min, max) for each axis.
        :param grid_shape: tuple (Nx, Ny, Nz) number of points along each axis.
        :return: tuple of (X, Y, Z, Bx, By, Bz) arrays with grid shape.
        """
        volume = self.sample_field_volume(
            bounds=(x_range, y_range, z_range), shape=grid_shape
        )
        return (
            volume.X,
            volume.Y,
            volume.Z,
            volume.B[..., 0],
            volume.B[..., 1],
            volume.B[..., 2],
        )

    def tree_flatten(self):
        children = (self.position, self.rotation_vector, self.components)
        aux_data = (self.name,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, components = children
        (name,) = aux_data
        obj = object.__new__(cls)
        obj.position = position
        obj.rotation_vector = rotation_vector
        obj.components = list(components)
        obj.name = name
        return obj


jax.tree_util.register_pytree_node_class(MagneticSystem)
