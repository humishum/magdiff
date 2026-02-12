from typing import Tuple

import jax
import jax.numpy as jnp

from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0
from magdiff.math import rotvec_to_matrix


class Coil(MagneticComponent):
    """
    Electromagnetic coil component.
    """

    n_seg: int = 100  # Number of line segments used for loop quadrature.

    def __init__(
        self,
        turns: int = 1,
        current: float = 0.0,
        radius: float = 0.01,
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """
        :param turns: number of turns in the coil.
        :param current: coil current in Amps.
        :param radius: coil radius in meters.
        :param position: 3D position relative to parent frame (m).
        :param rotation_vector: axis-angle rotation vector relative to parent frame (rad).
        :param name: optional human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.turns = int(turns)
        self.current = jnp.asarray(current, dtype=float)
        self.radius = float(radius)

    def field_at(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute magnetic flux density B at a point.

        Uses a filamentary circular-loop Biot-Savart quadrature:
            B(r) = (mu0 * I / (4*pi)) * integral[ dl' x (r - r') / |r - r'|^3 ] * N
        or rather in our discretized case since we can't figure out the continuous integral with jax(yet!!!!) TODO!!!!
            B(r) ~= (mu0 * I / (4*pi)) * sum_k [ dl'_k x (r - r'_k) / |r - r'_k|^3 ] * N

        where
            N = number of turns in coil
            I = current in coil
            dl'_k = differential length element of the coil at the k-th segment
            r'_k = source point on the coil at the k-th segment
            r = observation point

        Implementation notes:
            - The loop is discretized into `n_seg` angular segments.
            - Coil is modeled in local frame as a circle in the local x-y plane.
            - `source_points` correspond to r' along the wire.
            - `dl_vectors` correspond to dl' tangent vectors scaled by dtheta.
        """
        point = jnp.asarray(point, dtype=float)
        rotation = rotvec_to_matrix(self.rotation_vector)
        point_local = rotation.T @ (point - self.position)

        theta = jnp.linspace(0.0, 2.0 * jnp.pi, self.n_seg, endpoint=False)
        dtheta = 2.0 * jnp.pi / self.n_seg

        # Source points r' along the circular wire in local frame.
        source_points = jnp.stack(
            [
                self.radius * jnp.cos(theta),
                self.radius * jnp.sin(theta),
                jnp.zeros_like(theta),
            ],
            axis=1,
        )  # (n_seg, 3)

        # Differential line elements dl' **tangent** to the wire.
        dl_vectors = (
            jnp.stack(
                [
                    -self.radius * jnp.sin(theta),
                    self.radius * jnp.cos(theta),
                    jnp.zeros_like(theta),
                ],
                axis=1,
            )
            * dtheta
        )  # (n_seg, 3)

        # Vector from each source point to observation point: (r - r').
        source_to_observer = point_local[None, :] - source_points
        source_dist = jnp.linalg.norm(source_to_observer, axis=1, keepdims=True)
        eps = 1e-10
        integrand = jnp.cross(dl_vectors, source_to_observer) / (source_dist**3 + eps)

        b_local = (
            (MU_0 * self.current / (4.0 * jnp.pi))
            * jnp.sum(integrand, axis=0)
            * self.turns
        )
        return rotation @ b_local

    # Pytree registration
    def tree_flatten(
        self,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[str, int, float]]:
        children = (self.position, self.rotation_vector, self.current)
        aux_data = (self.name, self.turns, self.radius)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, current = children
        name, turns, radius = aux_data
        obj = object.__new__(cls)
        obj.position = position
        obj.rotation_vector = rotation_vector
        obj.current = current
        obj.turns = turns
        obj.radius = radius
        obj.name = name
        return obj


jax.tree_util.register_pytree_node_class(Coil)
