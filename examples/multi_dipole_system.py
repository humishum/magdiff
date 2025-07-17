import jax, jax.numpy as jnp
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualize import visualize_field

# Target where we want |B| maximal
target = jnp.array([1.0, 1.0, 1.0])

# Create a system with multiple dipoles
dip1 = Dipole(moment=jnp.array([1.0, 0.0, 0.0]), position=jnp.array([-1.0, 0.0, 0.0]))
dip2 = Dipole(moment=jnp.array([0.0, 1.0, 0.0]), position=jnp.array([1.0, 0.0, 0.0]))
dip3 = Dipole(moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.array([0.0, 1.0, 0.0]))

system = MagneticSystem([dip1, dip2, dip3])

# Get initial system parameters
param_dict = system.get_parameters()
print("Initial system parameters structure:")
for i, comp_info in enumerate(param_dict['structure']):
    print(f"  Component {i}: {comp_info}")
print("Initial flat parameters shape:", param_dict['flat_params'].shape)
print("Initial flat parameters:", param_dict['flat_params'])

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
print(f"\nInitial field at target: {system.field_at(target)}")
print(f"Initial field magnitude: {jnp.linalg.norm(system.field_at(target)):.4e} T")

# Optimization loop
params = initial_params
lr = 0.01  # Smaller learning rate for multiple components
for i in range(20):
    g = grad_fn(params)
    params = params - lr * g  # gradient ascent for maximization
    field_mag = -neg_Bmag_system(params)
    if i % 5 == 0:  # Print every 5th iteration
        print(f"Iteration {i+1}: |B|={field_mag:.4e} T")

# Update the system with optimized parameters
final_param_dict = param_dict.copy()
final_param_dict['flat_params'] = params
system.set_parameters(final_param_dict)

# Verify the system has been updated
print("\nFinal system state:")
for i, comp in enumerate(system.components):
    print(f"Dipole {i+1}: position={comp.position}, moment={comp.moment}")

print(f"\nFinal field at target: {system.field_at(target)}")
print(f"Final field magnitude: {jnp.linalg.norm(system.field_at(target)):.4e} T")

# Show improvement
initial_field_mag = jnp.linalg.norm(system.field_at_with_params(target, initial_params))
final_field_mag = jnp.linalg.norm(system.field_at(target))
print(f"Improvement factor: {final_field_mag / initial_field_mag:.2f}x")

# Visualize the optimized system
fig = visualize_field(system)
fig.show() 