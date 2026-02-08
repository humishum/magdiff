import jax, jax.numpy as jnp
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualize import visualize_field
from matplotlib import pyplot as plt
# Target where we want |B| maximal
target = jnp.array([-5.0, 0.0, 1.0])

# Create a system with multiple dipoles
dip1 = Dipole(moment=jnp.array([1.0, 0.0, 0.0]), position=jnp.array([1.0, 0.0, 0.0]), rotation_vector=jnp.zeros(3))
dip2 = Dipole(moment=jnp.array([1.0, 1.0, 0.0]), position=jnp.array([0.0, 1.0, 0.0]), rotation_vector=jnp.zeros(3))
dip3 = Dipole(moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.array([0.0, 1.0, 1.0]), rotation_vector=jnp.zeros(3))

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

# ----------------------------
# Optimization helpers
# ----------------------------
def _split_dipole_params(flat_params):
    """Assumes flat_params packs [pos(3), moment(3)] per dipole."""
    x = flat_params.reshape((-1, 6))
    positions = x[:, :3]
    moments = x[:, 3:]
    return positions, moments


def _replace_positions(flat_params_template, new_positions_flat):
    """Return a full flat_params vector with positions replaced and moments unchanged."""
    n = flat_params_template.shape[0] // 6
    x = flat_params_template.reshape((n, 6))
    new_pos = new_positions_flat.reshape((n, 3))
    x = x.at[:, :3].set(new_pos)
    return x.reshape((-1,))


def loss_fn(pos_params):
    """
    Scalar loss to MINIMIZE.

    We optimize POSITIONS ONLY (moments are held fixed at their initial values).
    Maximizing ||B|| at a point is still effectively unbounded if positions can move arbitrarily
    close to the target (until the r_norm epsilon hits), so we keep it well-behaved with
    boundary conditions + a soft min-distance barrier.
    """
    full_flat = _replace_positions(flat_params_template, pos_params)
    B_field = system.field_at_with_params(target, full_flat)
    field_mag = jnp.linalg.norm(B_field)

    positions = pos_params.reshape((-1, 3))

    # L2 regularization keeps positions from drifting too far from the initial config
    pos_delta = positions - init_positions
    pos_l2 = jnp.mean(jnp.sum(pos_delta**2, axis=-1))

    # Soft barrier discourages diving into the near-singularity at the target
    dists = jnp.linalg.norm(positions - target[None, :], axis=-1)
    dist_barrier = jnp.mean(jax.nn.softplus(min_dist - dists) ** 2)

    # MINIMIZE: negative of what we want + penalties
    loss = (
        -field_mag
        + reg_position * pos_l2
        + reg_min_dist * dist_barrier
    )
    aux = (field_mag, pos_l2, dist_barrier)
    return loss, aux


value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

# Extract initial parameters
initial_params = param_dict['flat_params']
print(f"\nInitial field at target: {system.field_at(target)}")
print(f"Initial field magnitude: {jnp.linalg.norm(system.field_at(target)):.4e} T")

# Optimization loop
params = initial_params
# Template full parameter vector; we will only change positions
flat_params_template = initial_params
init_positions, init_moments = _split_dipole_params(flat_params_template)
init_positions = init_positions  # shape (n_dipoles, 3)
pos_params0 = init_positions.reshape((-1,))

# Optimizer + schedule knobs (tune these)
n_steps = int(2_000)
lr0 = 1e2
lr_min = 1.0
beta1 = 0.9
beta2 = 0.999
adam_eps = 1e-8
grad_clip = 1e-4  # clip global grad norm (raise if updates are too timid)

# Hard bounds (projection) to prevent pathological solutions:
# - flying far away (field -> ~0)
# - diving into the near-singularity (field spikes)
pos_bound = 10.0  # positions clipped to [-pos_bound, +pos_bound] per coordinate

# Regularization knobs
reg_position = 1e-3  # penalize deviation from initial positions
min_dist = 0.5
reg_min_dist = 1e-9


def lr_schedule(i):
    # Cosine decay from lr0 -> lr_min over n_steps
    t = i / jnp.maximum(1, (n_steps - 1))
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + jnp.cos(jnp.pi * t))


@jax.jit
def _adam_scan(init_pos_params):
    def _project(pos_params):
        positions = pos_params.reshape((-1, 3))
        positions = jnp.clip(positions, -pos_bound, pos_bound)
        return positions.reshape((-1,))

    def _body(carry, i):
        pos_params, m, v, best_pos_params, best_field = carry

        (loss, (field_mag, pos_l2, dist_barrier)), g = value_and_grad_fn(pos_params)

        # Track best params by true objective (field_mag) at the *current* params.
        # Important: field_mag is computed BEFORE we update params.
        # Previously we compared pre-update field_mag but stored the post-update params,
        # which can make the saved "best" params unrelated (and often much worse).
        best_better = field_mag > best_field
        best_field = jnp.where(best_better, field_mag, best_field)
        best_pos_params = jnp.where(best_better, pos_params, best_pos_params)

        # Clip gradient to avoid rare blow-ups near the r_norm epsilon
        g_norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, grad_clip / (g_norm + 1e-12))
        g = g * scale

        # Adam update
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** (i + 1))
        vhat = v / (1.0 - beta2 ** (i + 1))

        lr = lr_schedule(i)
        step = lr * mhat / (jnp.sqrt(vhat) + adam_eps)
        pos_params = pos_params - step  # gradient descent on loss
        pos_params = _project(pos_params)

        metrics = (loss, field_mag, g_norm, jnp.linalg.norm(step), pos_l2, dist_barrier, lr)
        return (pos_params, m, v, best_pos_params, best_field), metrics

    init_m = jnp.zeros_like(init_pos_params)
    init_v = jnp.zeros_like(init_pos_params)
    init_best_field = -jnp.inf
    init_best_pos_params = init_pos_params
    init_carry = (init_pos_params, init_m, init_v, init_best_pos_params, init_best_field)

    (final_pos_params, _m, _v, best_pos_params, best_field), metrics = jax.lax.scan(
        _body, init_carry, jnp.arange(n_steps)
    )
    return final_pos_params, best_pos_params, best_field, metrics


pos_params, best_pos_params, best_field_mag, metrics = _adam_scan(pos_params0)
losses, field_mags, grad_norms, step_norms, pos_l2s, dist_barriers, lrs = metrics

# Host-side logging / plotting
losses_np = jax.device_get(losses)
field_mags_np = jax.device_get(field_mags)
grad_norms_np = jax.device_get(grad_norms)
step_norms_np = jax.device_get(step_norms)
lrs_np = jax.device_get(lrs)
pos_l2s_np = jax.device_get(pos_l2s)
dist_barriers_np = jax.device_get(dist_barriers)
best_field_mag_np = float(jax.device_get(best_field_mag))
best_pos_params_np = jax.device_get(best_pos_params)

print_every = int(1e2)
for i in range(0, n_steps, print_every):
    print(
        f"Iteration {i+1}: |B|={field_mags_np[i]:.4e} T | "
        f"loss={losses_np[i]:.4e} | lr={lrs_np[i]:.2e} | "
        f"||grad||={grad_norms_np[i]:.2e} | ||step||={step_norms_np[i]:.2e}"
    )
print(f"Best |B| seen during run: {best_field_mag_np:.4e} T")
# Update the system with optimized parameters
final_param_dict = param_dict.copy()
final_param_dict['flat_params'] = _replace_positions(flat_params_template, best_pos_params_np)
print(initial_params)
print("---")
print(final_param_dict["flat_params"])
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

best_so_far_np = jnp.maximum.accumulate(field_mags).astype(field_mags.dtype)
best_so_far_np = jax.device_get(best_so_far_np)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))

# Left: objective over time
ax0.plot(field_mags_np, label="|B|(iter)")
ax0.plot(best_so_far_np, label="best so far")
ax0.set_title("Target field magnitude")
ax0.set_xlabel("step")
ax0.set_ylabel("|B| [T]")
ax0.legend()

# Right: loss/regularizers + LR
ax1.plot(losses_np, label="loss")
ax1.plot(pos_l2s_np, label="pos_l2 (Δpos²)")
ax1.plot(dist_barriers_np, label="min-dist barrier")
ax1.set_title("Optimization metrics")
ax1.set_xlabel("step")
ax1.set_ylabel("value")
ax1.set_yscale("symlog", linthresh=1e-12)

ax1b = ax1.twinx()
ax1b.plot(lrs_np, color="k", alpha=0.35, linewidth=1.0, label="lr")
ax1b.set_ylabel("learning rate")

# Combine legends from both right axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

ax0.grid(True)
ax1.grid(True)
plt.tight_layout()
plt.show()
# Visualize the optimized system
fig = visualize_field(system)
fig.show() 