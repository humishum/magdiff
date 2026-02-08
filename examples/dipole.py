"""
This examples shows an example of setting up a single dipole, and a target point
in space we want to maximize the field strength at, without changing the position or moment of the dipole.
This is a watered down example, but shows how one can use magdiff for optimization problems. 
"""

import jax
import jax.numpy as jnp

from magdiff.math import rotate_vector
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.utils import print_tree
from magdiff.visualize import visualize_field

# Target point where we want |B| maximal
target = jnp.array([5.0, 5.0, 5.0])

# Create dipole and system
dip = Dipole(name="main", moment=jnp.array([50.0, 0.0, 0.0]), position=jnp.array([1.0, 0.0, 0.0]), rotation_vector=jnp.array([0.0, 0.0, 0.0]))
system = MagneticSystem([dip], name="single_dipole_sys")

print("=== Initial system ===")
print_tree(system)
print(f"Initial Field at target: {system.field_at(target)}")
print(f"|B| at target: {jnp.linalg.norm(system.field_at(target)):.4e} T\n")

# Define the loss function we want to minimize
def loss(system):
    """The loss function is defined as the negative of the field magnitude at the target point. 
        Since we don't want to move the position or modify the moment, we use jax.lax.stop_gradient to freeze those params, 
        and only optimize for the rotation_vector    
    """
    frozen_components = []
    for comp in system.components:
        frozen_components.append(Dipole(
            moment=jax.lax.stop_gradient(comp.moment),
            position=jax.lax.stop_gradient(comp.position),
            rotation_vector=comp.rotation_vector,  
            name=comp.name,
        ))
    frozen_sys = MagneticSystem(components=frozen_components, name=system.name)
    return -jnp.linalg.norm(frozen_sys.field_at(target))

grad_fn = jax.grad(loss)

# Show what the gradient looks like
grads = grad_fn(system)
print("=== Gradient structure ===")
print(f"  grad position: {grads.components[0].position}")
print(f"  grad moment:   {grads.components[0].moment}")
print(f"  grad rot_vec:  {grads.components[0].rotation_vector}\n")

# In the ideal world this is the final position we want that maximized |B| 
r = target - system.components[0].position
r_hat = r / jnp.linalg.norm(r)
print(f"Initial moment direction:    {system.components[0].moment / jnp.linalg.norm(system.components[0].moment)}")
print(f"Ideal direction to target (r_hat): {r_hat}")


# Use normalised gradient steps to decouple step size from field magnitude.
lr = 1e6
n_steps = 500
current = system

for i in range(1, n_steps + 1):
    g = grad_fn(current)
    current = jax.tree.map(lambda p, gp: p - lr * gp, current, g)

    if i % 100 == 0 or i == 1:
        field_mag = jnp.linalg.norm(current.field_at(target))
        m_world = rotate_vector(current.components[0].rotation_vector, current.components[0].moment)
        m_dir = m_world / jnp.linalg.norm(m_world)
        print(f"Step {i:4d}: |B| = {field_mag:.4e} T  moment_dir={m_dir}  rotvec={current.components[0].rotation_vector}")


print("=== After optimisation ===")
print_tree(current)
m_world = rotate_vector(current.components[0].rotation_vector, current.components[0].moment)
print(f"Moment in world frame: {m_world}")
print(f"Field at target: {current.field_at(target)}")
print(f"|B| at target:   {jnp.linalg.norm(current.field_at(target)):.4e} T")
print(f"Error in moment direction: {jnp.linalg.norm(m_dir - r_hat)}")

fig = visualize_field(current)
fig.show()