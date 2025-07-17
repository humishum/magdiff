import jax, jax.numpy as jnp
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualize import visualize_field

# Target where we want |B| maximal
target = jnp.array([-0.5, 0.0, 0.0])

# Create dipole and system
dip = Dipole(moment=jnp.array([0,0,50]), position=jnp.array([-1.0,0,0]))
system = MagneticSystem([dip])

# Get initial system parameters
param_dict = system.get_parameters()
print("Initial system parameters:", param_dict)

def neg_Bmag_system(flat_params):
    """
    Objective function that operates on the entire system.
    Returns negative field magnitude at target point.
    """
    B_field = system.field_at_with_params(target, flat_params)
    return -jnp.linalg.norm(B_field)

# Create gradient function for system-level optimization
grad_fn = jax.grad(neg_Bmag_system)

# Extract initial parameters
initial_params = param_dict['flat_params']
print("Initial flat parameters shape:", initial_params.shape)
print("Initial flat parameters:", initial_params)

initial_gradient = grad_fn(initial_params)
print(f"Initial gradient: {initial_gradient}")
print(f"Initial gradient magnitude: {jnp.linalg.norm(initial_gradient):.4e}")

# Optimization loop
params = initial_params
lr = 0.1
for i in range(5):
    g = grad_fn(params)
    print(f"Gradient magnitude: {jnp.linalg.norm(g):.4e}")
    params = params - lr * g  # gradient ascent for maximization
    field_mag = -neg_Bmag_system(params)
    print(f"{i+1}: params={params}, |B|={field_mag:.4e} T")

# Update the system with optimized parameters
final_param_dict = param_dict.copy()
final_param_dict['flat_params'] = params
system.set_parameters(final_param_dict)

# Verify the system has been updated
print("\nFinal system state:")
print("Dipole position:", system.components[0].position)
print("Dipole moment:", system.components[0].moment)
print("Field at target:", system.field_at(target))
print("Field magnitude:", jnp.linalg.norm(system.field_at(target)))

# Visualize the optimized system
fig = visualize_field(system)
fig.show()
