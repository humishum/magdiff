"""
Multi-dipole optimization example. 
In this example, we explore how we can build a system of multiple dipoles placed in some arbitrary configuration, 
and then optimize their positions to maximize field strenght at a point in space. 

The optimization in this is somewhat overkill, 

Demonstrates:
  - Freezing moments with stop_gradient (optimize positions only)
  - Position bounds via projection (clipping)
  - Minimum-distance soft barrier
  - Adam optimizer with cosine LR schedule via jax.lax.scan
"""

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.utils import print_tree

# ---------------------------------------------------------------------------
# 1.  Build the system
# ---------------------------------------------------------------------------
target = jnp.array([-5.0, 0.0, 1.0])

system = MagneticSystem(
    components=[
        Dipole(name="dip_x",  moment=jnp.array([1.0, 0.0, 0.0]), position=jnp.array([1.0, 0.0, 0.0])),
        Dipole(name="dip_xy", moment=jnp.array([1.0, 1.0, 0.0]), position=jnp.array([0.0, 1.0, 0.0])),
        Dipole(name="dip_z",  moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.array([0.0, 1.0, 1.0])),
    ],
    name="dipole_array",
)

print("=== Initial system ===")
print_tree(system)
print(f"\nTarget point: {target}")
print(f"Initial |B| at target: {jnp.linalg.norm(system.field_at(target)):.4e} T\n")

# ---------------------------------------------------------------------------
# 2.  Store the initial system so we can freeze moments
# ---------------------------------------------------------------------------
init_system = system  # reference for regularisation


# ---------------------------------------------------------------------------
# 3.  Loss function — takes the full system pytree
# ---------------------------------------------------------------------------
# Hyperparameters
pos_bound = 10.0        # hard bound on positions
min_dist = 0.5          # soft barrier: minimum distance to target
reg_position = 1e-3     # L2 penalty on position drift
reg_min_dist = 1e-9     # weight for distance barrier


def loss_fn(system):
    """
    Scalar loss to MINIMISE.  Optimises positions only, while magnetic moments are frozen.

    - Primary objective: maximise |B| at the target  (negated for minimisation)
    - L2 regularisation
    - distance barrier prevents positions from collapsing onto the target
    """
    frozen_components = []
    for comp in system.components:
        frozen_components.append(
            Dipole(
                moment=jax.lax.stop_gradient(comp.moment),
                position=comp.position,
                rotation_vector=jax.lax.stop_gradient(comp.rotation_vector),
                name=comp.name,
            )
        )
    frozen_system = MagneticSystem(
        components=frozen_components,
        position=jax.lax.stop_gradient(system.position),
        rotation_vector=jax.lax.stop_gradient(system.rotation_vector),
        name=system.name,
    )

    # Field magnitude at target
    B = frozen_system.field_at(target)
    field_mag = jnp.linalg.norm(B)

    # Collect positions for regularisation
    positions = jnp.stack([c.position for c in system.components])
    init_positions = jnp.stack([c.position for c in init_system.components])

    # L2 drift penalty
    pos_delta = positions - init_positions
    pos_l2 = jnp.mean(jnp.sum(pos_delta ** 2, axis=-1))

    # min-distance barrier (keeps dipoles from moving to exacty the target point)
    dists = jnp.linalg.norm(positions - target[None, :], axis=-1)
    dist_barrier = jnp.mean(jax.nn.softplus(min_dist - dists) ** 2)

    loss = -field_mag + reg_position * pos_l2 + reg_min_dist * dist_barrier
    aux = (field_mag, pos_l2, dist_barrier)
    return loss, aux


# ---------------------------------------------------------------------------
# 4.  Adam optimiser via jax.lax.scan (all in XLA, no Python-loop overhead)
# ---------------------------------------------------------------------------
n_steps = 2_000
lr0 = 1e2
lr_min = 1.0
beta1 = 0.9
beta2 = 0.999
adam_eps = 1e-8
grad_clip = 1e-4


def lr_schedule(i):
    """Cosine decay from lr0 → lr_min over n_steps."""
    t = i / jnp.maximum(1, n_steps - 1)
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + jnp.cos(jnp.pi * t))


value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


@jax.jit
def run_optimisation(system):
    # Extract only the position leaves we want to optimise
    # We'll work directly with a flat positions array for the scan body
    init_pos = jnp.stack([c.position for c in system.components])  # (N, 3)
    pos_flat = init_pos.reshape(-1)

    def _rebuild_system(pos_flat):
        """Rebuild the system pytree with updated positions."""
        positions = pos_flat.reshape((-1, 3))
        new_components = []
        for i, comp in enumerate(system.components):
            new_components.append(
                Dipole(
                    moment=comp.moment,
                    position=positions[i],
                    rotation_vector=comp.rotation_vector,
                    name=comp.name,
                )
            )
        return MagneticSystem(
            components=new_components,
            position=system.position,
            rotation_vector=system.rotation_vector,
            name=system.name,
        )

    def _loss_of_positions(pos_flat):
        """Wrap loss_fn so it's a function of the flat position vector only."""
        sys = _rebuild_system(pos_flat)
        return loss_fn(sys)

    pos_value_and_grad = jax.value_and_grad(_loss_of_positions, has_aux=True)

    def _project(pos_flat):
        """Clip positions to [-pos_bound, pos_bound]."""
        return jnp.clip(pos_flat, -pos_bound, pos_bound)

    def _body(carry, i):
        pos, m, v, best_pos, best_field = carry

        (loss, (field_mag, pos_l2, dist_barrier)), g = pos_value_and_grad(pos)

        # Track best
        better = field_mag > best_field
        best_field = jnp.where(better, field_mag, best_field)
        best_pos = jnp.where(better, pos, best_pos)

        # Clip gradient norm
        g_norm = jnp.linalg.norm(g)
        g = g * jnp.minimum(1.0, grad_clip / (g_norm + 1e-12))

        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        mhat = m / (1 - beta1 ** (i + 1))
        vhat = v / (1 - beta2 ** (i + 1))
        lr = lr_schedule(i)
        step = lr * mhat / (jnp.sqrt(vhat) + adam_eps)

        pos = _project(pos - step)

        metrics = (loss, field_mag, g_norm, jnp.linalg.norm(step), pos_l2, dist_barrier, lr)
        return (pos, m, v, best_pos, best_field), metrics

    init_carry = (
        pos_flat,
        jnp.zeros_like(pos_flat),       # m
        jnp.zeros_like(pos_flat),       # v
        pos_flat,                        # best_pos
        -jnp.inf,                        # best_field
    )

    (final_pos, _, _, best_pos, best_field), metrics = jax.lax.scan(
        _body, init_carry, jnp.arange(n_steps)
    )
    return best_pos, best_field, metrics


# ---------------------------------------------------------------------------
# 5.  Run
# ---------------------------------------------------------------------------
best_pos, best_field_mag, metrics = run_optimisation(system)
losses, field_mags, grad_norms, step_norms, pos_l2s, dist_barriers, lrs = metrics

# Rebuild the optimised system
best_positions = best_pos.reshape((-1, 3))
optimised_components = []
for i, comp in enumerate(system.components):
    optimised_components.append(
        Dipole(
            moment=comp.moment,
            position=best_positions[i],
            rotation_vector=comp.rotation_vector,
            name=comp.name,
        )
    )
optimised_system = MagneticSystem(components=optimised_components, name="dipole_array (optimised)")

# ---------------------------------------------------------------------------
# 6.  Report
# ---------------------------------------------------------------------------
print("=== Optimised system ===")
print_tree(optimised_system)

initial_field_mag = jnp.linalg.norm(system.field_at(target))
final_field_mag = jnp.linalg.norm(optimised_system.field_at(target))

print(f"\nInitial |B| at target: {initial_field_mag:.4e} T")
print(f"Best |B| at target:   {float(best_field_mag):.4e} T")
print(f"Final |B| at target:  {final_field_mag:.4e} T")
print(f"Improvement factor:   {final_field_mag / initial_field_mag:.2f}x")

# ---------------------------------------------------------------------------
# 7.  Plot
# ---------------------------------------------------------------------------
field_mags_np = jax.device_get(field_mags)
losses_np = jax.device_get(losses)
pos_l2s_np = jax.device_get(pos_l2s)
dist_barriers_np = jax.device_get(dist_barriers)
lrs_np = jax.device_get(lrs)

best_so_far = jnp.maximum.accumulate(field_mags)
best_so_far_np = jax.device_get(best_so_far.astype(field_mags.dtype))

# print(jax.make_jaxpr(run_optimisation.__wrapped__)(system))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))

ax0.plot(field_mags_np, label="|B|(iter)")
ax0.plot(best_so_far_np, label="best so far")
ax0.set_title("Target field magnitude")
ax0.set_xlabel("step")
ax0.set_ylabel("|B| [T]")
ax0.legend()
ax0.grid(True)

ax1.plot(losses_np, label="loss")
ax1.plot(pos_l2s_np, label="pos L2 drift")
ax1.plot(dist_barriers_np, label="min-dist barrier")
ax1.set_title("Optimisation metrics")
ax1.set_xlabel("step")
ax1.set_ylabel("value")
ax1.set_yscale("symlog", linthresh=1e-12)

ax1b = ax1.twinx()
ax1b.plot(lrs_np, color="k", alpha=0.35, linewidth=1.0, label="lr")
ax1b.set_ylabel("learning rate")

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
ax1.grid(True)

plt.tight_layout()
plt.show()
