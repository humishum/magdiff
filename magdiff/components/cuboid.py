import jax
import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.constants import MU_0
from magdiff.math import rotvec_to_matrix


class Cuboid(MagneticComponent):
    """Uniformly magnetized cuboid (rectangular prism).

    Field model
    ----------
    Uses the closed-form finite-prism expressions (log/atan corner sums)
    for a uniformly magnetized rectangular magnet:

    - Camacho et al., Rev. Mex. Fis. E 59 (2013), Eqs. (2)-(4)
      https://rmf.smf.mx/ojs/index.php/rmf-e/article/view/2055
    - By-term correction in the 2024 erratum:
      https://rmf.smf.mx/ojs/index.php/rmf-e/article/view/7526

    The implemented kernel gives the demagnetizing contribution
    (equivalent to mu0 * H from magnetic charges). For points inside the
    magnet, we add mu0 * M to return B = mu0 (H + M).
    """

    def __init__(
        self,
        magnetization=jnp.zeros(3),
        dimension=jnp.array([1.0, 1.0, 1.0]),
        position=jnp.zeros(3),
        rotation_vector=jnp.zeros(3),
        name=None,
    ):
        """Create a cuboid magnet.

        Parameters
        ----------
        magnetization : array-like, shape (3,)
            Homogeneous magnetization vector (A/m) in the body frame.
        dimension : array-like, shape (3,)
            Side lengths (Lx, Ly, Lz) of the cuboid in meters.
        position : array-like, shape (3,)
            Position of the cuboid centre relative to parent frame (m).
        rotation_vector : array-like, shape (3,)
            Axis-angle rotation vector relative to parent frame (rad).
        name : str, optional
            Human-readable label.
        """
        super().__init__(position=position, rotation_vector=rotation_vector, name=name)
        self.magnetization = jnp.asarray(magnetization, dtype=float)
        self.dimension = jnp.asarray(dimension, dtype=float)

    @staticmethod
    def _field_magnetized_along_z(
        point_body: jnp.ndarray,
        dimension: jnp.ndarray,
        magnetization_z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Closed-form demag field for M = [0, 0, magnetization_z] in body frame."""
        x, y, z = point_body
        Lx, Ly, Lz = dimension
        a = 0.5 * Lx
        b = 0.5 * Ly
        c = 0.5 * Lz
        eps = 1e-18

        def _safe_signed(v: jnp.ndarray) -> jnp.ndarray:
            sign = jnp.where(v >= 0.0, 1.0, -1.0)
            return jnp.where(jnp.abs(v) < eps, sign * eps, v)

        def _safe_pos(v: jnp.ndarray) -> jnp.ndarray:
            return jnp.maximum(v, eps)

        # Eq. (4): atan-kernel
        def f1(xv: jnp.ndarray, yv: jnp.ndarray, zv: jnp.ndarray) -> jnp.ndarray:
            xp = xv + a
            yp = yv + b
            zp = zv + c
            r = jnp.sqrt(xp * xp + yp * yp + zp * zp)
            return jnp.arctan((xp * yp) / _safe_signed(zp * r))

        # Eq. (4): log-kernels (using log-difference for numerical stability)
        def log_f2(xv: jnp.ndarray, yv: jnp.ndarray, zv: jnp.ndarray) -> jnp.ndarray:
            num = jnp.sqrt((xv + a) ** 2 + (yv - b) ** 2 + (zv + c) ** 2) + b - yv
            den = jnp.sqrt((xv + a) ** 2 + (yv + b) ** 2 + (zv + c) ** 2) - b - yv
            return jnp.log(_safe_pos(num)) - jnp.log(_safe_pos(den))

        def log_f3(xv: jnp.ndarray, yv: jnp.ndarray, zv: jnp.ndarray) -> jnp.ndarray:
            num = jnp.sqrt((xv - a) ** 2 + (yv + b) ** 2 + (zv + c) ** 2) + a - xv
            den = jnp.sqrt((xv + a) ** 2 + (yv + b) ** 2 + (zv + c) ** 2) - a - xv
            return jnp.log(_safe_pos(num)) - jnp.log(_safe_pos(den))

        pref = MU_0 * magnetization_z / (4.0 * jnp.pi)

        # Eq. (2), with 2024 erratum correction for By:
        bx = pref * (
            log_f2(-x, y, -z)
            + log_f2(x, y, z)
            - log_f2(x, y, -z)
            - log_f2(-x, y, z)
        )
        by = pref * (
            log_f3(x, -y, -z)
            + log_f3(x, y, z)
            - log_f3(x, y, -z)
            - log_f3(x, -y, z)
        )
        bz = -pref * (
            f1(-x, y, z)
            + f1(-x, y, -z)
            + f1(-x, -y, z)
            + f1(-x, -y, -z)
            + f1(x, y, z)
            + f1(x, y, -z)
            + f1(x, -y, z)
            + f1(x, -y, -z)
        )
        return jnp.array([bx, by, bz], dtype=point_body.dtype)

    def field_at(self, point):
        """Magnetic flux-density B (Tesla) at an observation point."""
        point = jnp.asarray(point, dtype=float)

        # Parent frame -> body frame
        R = rotvec_to_matrix(self.rotation_vector)
        point_body = R.T @ (point - self.position)

        Mx, My, Mz = self.magnetization
        Lx, Ly, Lz = self.dimension

        # Superpose principal-axis contributions via coordinate permutations.
        # Mz term (identity mapping)
        Bz_term = self._field_magnetized_along_z(
            point_body=point_body,
            dimension=jnp.array([Lx, Ly, Lz], dtype=point_body.dtype),
            magnetization_z=Mz,
        )

        # Mx term: (x', y', z') = (y, z, x), dims' = (Ly, Lz, Lx)
        p_x = jnp.array([point_body[1], point_body[2], point_body[0]], dtype=point_body.dtype)
        B_xp = self._field_magnetized_along_z(
            point_body=p_x,
            dimension=jnp.array([Ly, Lz, Lx], dtype=point_body.dtype),
            magnetization_z=Mx,
        )
        Bx_term = jnp.array([B_xp[2], B_xp[0], B_xp[1]], dtype=point_body.dtype)

        # My term: (x', y', z') = (z, x, y), dims' = (Lz, Lx, Ly)
        p_y = jnp.array([point_body[2], point_body[0], point_body[1]], dtype=point_body.dtype)
        B_yp = self._field_magnetized_along_z(
            point_body=p_y,
            dimension=jnp.array([Lz, Lx, Ly], dtype=point_body.dtype),
            magnetization_z=My,
        )
        By_term = jnp.array([B_yp[1], B_yp[2], B_yp[0]], dtype=point_body.dtype)

        B_body = Bx_term + By_term + Bz_term

        # Inside magnet: B = mu0 * (H + M). Outside: B = mu0 * H.
        # this can probably be changed to make the gradient smoother, right now we do a hard step function 
        half_dim = 0.5 * self.dimension
        inside = jnp.all(jnp.abs(point_body) <= half_dim)
        B_body = B_body + jnp.where(inside, MU_0 * self.magnetization, jnp.zeros(3, dtype=B_body.dtype))

        # Body frame -> parent frame
        return R @ B_body

    def tree_flatten(self):
        children = (
            self.position,
            self.rotation_vector,
            self.magnetization,
            self.dimension,
        )
        aux_data = (self.name,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        position, rotation_vector, magnetization, dimension = children
        (name,) = aux_data
        obj = object.__new__(cls)
        obj.position = position
        obj.rotation_vector = rotation_vector
        obj.magnetization = magnetization
        obj.dimension = dimension
        obj.name = name
        return obj


jax.tree_util.register_pytree_node_class(Cuboid)
