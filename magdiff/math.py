"""Math/geometry utilities"""

import jax.numpy as jnp


def quat_normalize(quat_wxyz: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Normalize a quaternion to unit length, and clamp to a small epsilon to avoid zero"""
    return quat_wxyz / jnp.maximum(jnp.linalg.norm(quat_wxyz), eps)


# Todo: need to understand this better
def rotvec_to_matrix(rotvec: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation vector (3,) to 3x3 rotation matrix using a numerically stable
    Rodrigues / expmap form. We do this to avoid k = r/||r||.

    Uses:
        R = I + a(theta) [r]_x + b(theta) [r]_x^2
    where:
        a(theta) = sin(theta)/theta
        b(theta) = (1-cos(theta))/theta^2
    and switches to Taylor expansions near theta=0 for stability + good gradients.
    """
    r = rotvec
    theta2 = jnp.dot(r, r)

    # Skew-symmetric matrix [r]_x (no normalization needed)
    Rx = jnp.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ],
        dtype=r.dtype,
    )

    # Small-angle stable coefficients via Taylor expansion:
    #  a = sin(theta)/theta  ->  1 - theta^2/6 + theta^4/120 - ...
    #  b = (1-cos(theta))/theta^2 -> 1/2 - theta^2/24 + theta^4/720 - ...
    small = theta2 < 1e-8
    theta4 = theta2 * theta2

    a_small = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_small = 0.5 - theta2 / 24.0 + theta4 / 720.0

    # Clamp theta2 for the large branch so its gradient is never 0/0.
    # jnp.where evaluates gradients of BOTH branches, so the large branch
    # must be safe even when theta2 == 0 (it just won't be selected).
    safe_theta2 = jnp.where(small, 1.0, theta2)
    safe_theta = jnp.sqrt(safe_theta2)

    a_large = jnp.sin(safe_theta) / safe_theta
    b_large = (1.0 - jnp.cos(safe_theta)) / safe_theta2

    a = jnp.where(small, a_small, a_large)
    b = jnp.where(small, b_small, b_large)

    I = jnp.eye(3, dtype=r.dtype)
    return I + a * Rx + b * (Rx @ Rx)


def pose_to_matrix(position, rotation_vector):
    """Build a 4x4 homogeneous transform from position (3,) + rotvec (3,)."""
    R = rotvec_to_matrix(rotation_vector)
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(position)
    return T


def rotate_vector(rotvec, v_body):
    """Rotate vector v_body (3,) from local frame to world frame."""
    return rotvec_to_matrix(rotvec) @ v_body
