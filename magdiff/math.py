""" Math/geometry utilities """ 

import jax 
import jax.numpy as jnp

def quat_normalize(quat_wxyz: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """ Normalize a quaternion to unit length, and clamp to a small epsilon to avoid zero """ 
    return quat_wxyz / jnp.maximum(jnp.linalg.norm(quat_wxyz), eps)

