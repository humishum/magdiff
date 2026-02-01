""" Scientific constants """ 
# prefer to use jax.numpy as much as possible
import jax.numpy as jnp


# vacuum permeability (μ0) in T·m/A (≈ 1.2566e-6). 
MU0 = 4 * jnp.pi * 1e-7 