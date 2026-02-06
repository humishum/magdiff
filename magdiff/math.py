""" Math/geometry utilities """ 

import jax 
import jax.numpy as jnp

def quat_normalize(quat_wxyz: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """ Normalize a quaternion to unit length, and clamp to a small epsilon to avoid zero """ 
    return quat_wxyz / jnp.maximum(jnp.linalg.norm(quat_wxyz), eps)

def rotvec_to_matrix(rotvec: jnp.ndarray) -> jnp.ndarray:
    """Convert a rotation vector (3,) to a 3x3 rotation matrix via Rodrigues."""
    theta = jnp.linalg.norm(rotvec)
    # Safe normalize (avoid 0/0 — near identity, R ≈ I + [k]×)
    k = rotvec / jnp.maximum(theta, 1e-10)
    K = jnp.array([[0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]])
    R = (jnp.eye(3)
         + jnp.sin(theta) * K
         + (1 - jnp.cos(theta)) * (K @ K))
    return R

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
