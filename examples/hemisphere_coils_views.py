"""
Hemisphere coil array visualization.

Creates a hemispherical arrangement of coils, applies a user-editable current array,
and renders 6 subplots:
  - Row 1: |B| heatmaps for x-z, y-z, and y-x views.
  - Row 2: vector fields (quiver) for the same views.

Index mapping:
  - current at index k in `coil_currents` is applied to coil `C{k:02d}`.

Optional mode:
  - optimize only coil currents to maximize |B| at target points, with constraints:
      * per-coil bounds: -5 A <= I_k <= 5 A
      * total current budget: sum(abs(I_k)) <= 25 A
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from magdiff.components.coil import Coil
from magdiff.system import MagneticSystem


def rotvec_from_z_to_direction(direction: jnp.ndarray) -> jnp.ndarray:
    """Axis-angle rotvec that rotates local +z onto `direction`."""
    z_axis = jnp.array([0.0, 0.0, 1.0], dtype=float)
    d = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-12)

    cross = jnp.cross(z_axis, d)
    sin_theta = jnp.linalg.norm(cross)
    cos_theta = jnp.dot(z_axis, d)

    # Parallel to +z: no rotation.
    if float(sin_theta) < 1e-12 and float(cos_theta) > 0.0:
        return jnp.zeros(3)

    # Anti-parallel to +z: rotate pi around x-axis.
    if float(sin_theta) < 1e-12 and float(cos_theta) < 0.0:
        return jnp.array([jnp.pi, 0.0, 0.0])

    axis = cross / sin_theta
    angle = jnp.arctan2(sin_theta, cos_theta)
    return axis * angle


def hemisphere_points(n_points: int, radius: float) -> jnp.ndarray:
    """Approximate-uniform points on upper hemisphere (z >= 0)."""
    i = jnp.arange(n_points, dtype=float)
    golden_angle = jnp.pi * (3.0 - jnp.sqrt(5.0))

    # z in (0, 1): keep all points on upper hemisphere.
    z = (i + 0.5) / n_points
    r_xy = jnp.sqrt(jnp.maximum(0.0, 1.0 - z**2))
    phi = i * golden_angle

    x = r_xy * jnp.cos(phi)
    y = r_xy * jnp.sin(phi)
    pts = jnp.stack([x, y, z], axis=1)
    return radius * pts


def build_hemisphere_coil_system(
    n_coils: int,
    sphere_radius: float,
    coil_radius: float,
    turns: int,
    currents: jnp.ndarray,
) -> MagneticSystem:
    """Build a MagneticSystem with coils arranged on a hemisphere."""
    positions = hemisphere_points(n_coils, sphere_radius)
    coils = []
    for idx in range(n_coils):
        pos = positions[idx]
        # Aim coil local +z axis toward the hemisphere center.
        # normal_inward = -pos / jnp.maximum(jnp.linalg.norm(pos), 1e-12)
        normal_outward = pos / jnp.maximum(jnp.linalg.norm(pos), 1e-12)
        rotvec = rotvec_from_z_to_direction(normal_outward)
        coils.append(
            Coil(
                turns=turns,
                current=currents[idx],
                radius=coil_radius,
                position=pos,
                rotation_vector=rotvec,
                name=f"C{idx:02d}",
            )
        )
    return MagneticSystem(components=coils, name="hemisphere_coils")


def print_coil_mapping_table(coil_positions: np.ndarray, currents: np.ndarray) -> None:
    """Print explicit mapping: current index -> coil label -> position/current."""
    print("\n=== Coil Current Mapping ===")
    print("index  coil   current[A]      x[m]       y[m]       z[m]")
    for idx, (pos, cur) in enumerate(zip(coil_positions, currents)):
        print(
            f"{idx:>5d}  C{idx:02d}  {cur:>10.4f}  "
            f"{pos[0]:>8.4f}  {pos[1]:>8.4f}  {pos[2]:>8.4f}"
        )
    print()


def build_system_with_currents(
    template_system: MagneticSystem, currents: jnp.ndarray
) -> MagneticSystem:
    """Create a system copy with fixed coil pose/geometry and updated currents only."""
    new_components = []
    for idx, comp in enumerate(template_system.components):
        new_components.append(
            Coil(
                turns=comp.turns,
                current=currents[idx],
                radius=comp.radius,
                position=comp.position,
                rotation_vector=comp.rotation_vector,
                name=comp.name,
            )
        )
    return MagneticSystem(
        components=new_components,
        position=template_system.position,
        rotation_vector=template_system.rotation_vector,
        name=template_system.name,
    )


def constrained_currents(
    latent_currents: jnp.ndarray,
    per_coil_abs_max: float,
    total_abs_current_max: float,
) -> jnp.ndarray:
    """Map unconstrained latent variables to feasible coil currents.

    1) Box bound:      I_k in [-per_coil_abs_max, per_coil_abs_max] via tanh.
    2) L1 budget:      sum(abs(I_k)) <= total_abs_current_max via global scaling.
    """
    boxed = per_coil_abs_max * jnp.tanh(latent_currents)
    abs_sum = jnp.sum(jnp.abs(boxed))
    scale = jnp.minimum(1.0, total_abs_current_max / (abs_sum + 1e-12))
    return boxed * scale

@partial(jax.jit, static_argnames=("n_steps",))
def optimize_currents_for_targets(
    template_system: MagneticSystem,
    initial_currents: jnp.ndarray,
    target_points: jnp.ndarray,
    per_coil_abs_max: float = 5.0,
    total_abs_current_max: float = 25.0,
    n_steps: int = 250,
    learning_rate: float = 0.15,
):
    """Optimize coil currents for max mean |B| at target points under constraints."""
    target_points = jnp.asarray(target_points, dtype=float)
    initial_currents = jnp.asarray(initial_currents, dtype=float)

    # Latent parameterization used for constrained mapping.
    init_ratio = jnp.clip(initial_currents / per_coil_abs_max, -0.999, 0.999)
    latent = jnp.arctanh(init_ratio)

    def loss_fn(latent_currents):
        currents = constrained_currents(
            latent_currents, per_coil_abs_max, total_abs_current_max
        )
        system = build_system_with_currents(template_system, currents)
        b = system.field_at_points(target_points)  # (M, 3)
        b_mag = jnp.linalg.norm(b, axis=1)
        # Maximize mean |B| <=> minimize negative mean |B|.
        return -jnp.mean(b_mag)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Adam on latent variables.
    beta1, beta2 = 0.9, 0.999
    adam_eps = 1e-8
    def scan_step(carry, step_idx):
        latent_currents, m, v = carry

        loss_val, g = value_and_grad_fn(latent_currents)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)

        t = step_idx + 1
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        latent_currents = latent_currents - learning_rate * mhat / (
            jnp.sqrt(vhat) + adam_eps
        )

        currents = constrained_currents(
            latent_currents, per_coil_abs_max, total_abs_current_max
        )
        metrics = (
            loss_val,
            -loss_val,  # mean |B| at targets
            jnp.sum(jnp.abs(currents)),
        )
        return (latent_currents, m, v), metrics

    init_carry = (latent, jnp.zeros_like(latent), jnp.zeros_like(latent))
    (latent, _, _), metrics = jax.lax.scan(
        scan_step, init_carry, jnp.arange(n_steps)
    )
    loss_hist, mean_b_hist, total_abs_i_hist = metrics

    final_currents = constrained_currents(
        latent, per_coil_abs_max, total_abs_current_max
    )
    history = {
        "loss": loss_hist,
        "mean_b": mean_b_hist,
        "total_abs_current": total_abs_i_hist,
    }
    return final_currents, history


def random_feasible_currents(
    key: jax.Array,
    n_coils: int,
    per_coil_abs_max: float,
    total_abs_current_max: float,
) -> jnp.ndarray:
    """Draw random currents and project them to the feasible set."""
    raw = jax.random.uniform(
        key,
        shape=(n_coils,),
        minval=-per_coil_abs_max,
        maxval=per_coil_abs_max,
    )
    abs_sum = jnp.sum(jnp.abs(raw))
    scale = jnp.minimum(1.0, total_abs_current_max / (abs_sum + 1e-12))
    return raw * scale


def optimize_currents_multi_start(
    template_system: MagneticSystem,
    initial_currents: jnp.ndarray,
    target_points: jnp.ndarray,
    per_coil_abs_max: float,
    total_abs_current_max: float,
    n_steps: int,
    learning_rate: float,
    n_restarts: int,
    seed: int = 0,
):
    """Run multiple restarts and return the best optimization result."""
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

    best_currents = None
    best_history = None
    best_score = -np.inf  # maximize mean |B|
    summary = []

    key = jax.random.PRNGKey(seed)
    for restart_idx in range(n_restarts):
        if restart_idx == 0:
            start_currents = initial_currents
            start_tag = "user_init"
        else:
            key, subkey = jax.random.split(key)
            start_currents = random_feasible_currents(
                subkey,
                n_coils=initial_currents.shape[0],
                per_coil_abs_max=per_coil_abs_max,
                total_abs_current_max=total_abs_current_max,
            )
            start_tag = "random"

        currents_opt, history = optimize_currents_for_targets(
            template_system=template_system,
            initial_currents=start_currents,
            target_points=target_points,
            per_coil_abs_max=per_coil_abs_max,
            total_abs_current_max=total_abs_current_max,
            n_steps=n_steps,
            learning_rate=learning_rate,
        )

        final_mean_b = float(history["mean_b"][-1])
        final_total_abs_i = float(jnp.sum(jnp.abs(currents_opt)))
        summary.append(
            {
                "restart": restart_idx,
                "type": start_tag,
                "final_mean_b": final_mean_b,
                "final_total_abs_current": final_total_abs_i,
            }
        )

        if final_mean_b > best_score:
            best_score = final_mean_b
            best_currents = currents_opt
            best_history = history

    return best_currents, best_history, summary


def sample_plane_field(
    system: MagneticSystem,
    axis_u: int,
    axis_v: int,
    axis_fixed: int,
    fixed_value: float,
    u_lim: tuple[float, float],
    v_lim: tuple[float, float],
    n: int,
):
    """Sample B field on one 2D plane."""
    u = jnp.linspace(u_lim[0], u_lim[1], n)
    v = jnp.linspace(v_lim[0], v_lim[1], n)
    U, V = jnp.meshgrid(u, v, indexing="xy")  # shape (n, n)

    points = jnp.zeros((U.size, 3), dtype=float)
    points = points.at[:, axis_u].set(U.ravel())
    points = points.at[:, axis_v].set(V.ravel())
    points = points.at[:, axis_fixed].set(fixed_value)

    B = system.field_at_points(points).reshape(U.shape + (3,))
    Bu = B[..., axis_u]
    Bv = B[..., axis_v]
    Bmag = jnp.linalg.norm(B, axis=-1)
    return U, V, Bu, Bv, Bmag


def plot_plane(
    ax_heat,
    ax_vec,
    U,
    V,
    Bu,
    Bv,
    Bmag,
    coil_positions: np.ndarray,
    currents: np.ndarray,
    axis_u: int,
    axis_v: int,
    title: str,
    label_u: str,
    label_v: str,
    show_indices: bool = True,
    clip_percentile: float = 99.2,
    normalize_vectors: bool = True,
    quiver_scale: float = 32.0,
):
    """Render one column: heatmap (top) + quiver (bottom)."""
    U_np = np.asarray(U)
    V_np = np.asarray(V)
    Bu_np = np.asarray(Bu)
    Bv_np = np.asarray(Bv)
    Bmag_np = np.asarray(Bmag)
    eps = 1e-12

    # Clip extreme peaks from filamentary near-singular regions so plots stay interpretable.
    finite_mask = np.isfinite(Bmag_np)
    if np.any(finite_mask):
        b_clip = np.nanpercentile(Bmag_np[finite_mask], clip_percentile)
    else:
        b_clip = np.nanmax(Bmag_np)
    b_clip = max(float(b_clip), eps)

    singular_like_mask = (Bmag_np > b_clip) | (~finite_mask)
    Bmag_vis = np.where(singular_like_mask, np.nan, np.minimum(Bmag_np, b_clip))

    # Heatmap of log10 magnitude for dynamic range visibility.
    heat = np.log10(Bmag_vis + 1e-20)
    hm = ax_heat.pcolormesh(U_np, V_np, heat, shading="auto", cmap="viridis")
    plt.colorbar(
        hm,
        ax=ax_heat,
        shrink=0.82,
        label=f"log10(|B| [T], clipped @{clip_percentile:.1f}th pct)",
    )

    # Show coil centers projected into this view.
    cu = coil_positions[:, axis_u]
    cv = coil_positions[:, axis_v]
    ccol = np.where(currents >= 0.0, "tab:red", "tab:blue")
    ax_heat.scatter(cu, cv, c=ccol, edgecolor="black", s=45, zorder=4)
    if show_indices:
        for idx, (u_i, v_i) in enumerate(zip(cu, cv)):
            ax_heat.text(
                u_i + 0.004,
                v_i + 0.004,
                f"{idx}",
                fontsize=7,
                color="white",
                ha="left",
                va="bottom",
                zorder=5,
            )
    ax_heat.set_title(f"{title}: field strength")
    ax_heat.set_xlabel(f"{label_u} (m)")
    ax_heat.set_ylabel(f"{label_v} (m)")
    ax_heat.set_aspect("equal")

    # Vector field (optionally direction-normalized to avoid giant arrows near singularities).
    step = max(1, U_np.shape[0] // 18)
    Bu_vis = np.where(singular_like_mask, np.nan, Bu_np)
    Bv_vis = np.where(singular_like_mask, np.nan, Bv_np)
    Buv_norm = np.hypot(Bu_vis, Bv_vis)

    if normalize_vectors:
        U_plot = Bu_vis / (Buv_norm + eps)
        V_plot = Bv_vis / (Buv_norm + eps)
    else:
        U_plot = Bu_vis
        V_plot = Bv_vis

    ax_vec.quiver(
        U_np[::step, ::step],
        V_np[::step, ::step],
        U_plot[::step, ::step],
        V_plot[::step, ::step],
        np.minimum(Buv_norm[::step, ::step], b_clip),
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale if normalize_vectors else None,
        width=0.004,
    )
    ax_vec.scatter(cu, cv, c=ccol, edgecolor="black", s=40, zorder=4)
    if show_indices:
        for idx, (u_i, v_i) in enumerate(zip(cu, cv)):
            ax_vec.text(
                u_i + 0.004,
                v_i + 0.004,
                f"{idx}",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
                zorder=5,
            )
    if normalize_vectors:
        ax_vec.set_title(f"{title}: vector field (direction-normalized)")
    else:
        ax_vec.set_title(f"{title}: vector field")
    ax_vec.set_xlabel(f"{label_u} (m)")
    ax_vec.set_ylabel(f"{label_v} (m)")
    ax_vec.set_aspect("equal")


def main():
    # ----------------------------
    # User-editable coil settings
    # ----------------------------
    n_coils = 20  # keep between 16 and 25 as requested
    sphere_radius = 0.11
    coil_radius = 0.015
    turns = 120
    show_coil_indices = True
    optimize_currents = True  # Single flag to enable/disable current optimization.

    # Current constraints used when optimization is enabled.
    per_coil_abs_max = 5.0       # A, i_k in [-5, 5]
    total_abs_current_max = 25.0 # A, sum(abs(i_k)) <= 25
    opt_n_steps = 2200
    opt_learning_rate = 0.12
    opt_n_restarts = 8
    opt_seed = 7

    # Input current array (A):
    #   index k here -> coil C{k:02d} in the hemisphere layout.
    # Keep length equal to n_coils.
    coil_currents = jnp.array(
        [
            2.0, -1.0, 1.2, -2.2, 0.9,
            -1.6, 1.8, -0.7, 2.1, -1.9,
            1.0, -0.8, 1.4, -2.0, 0.6,
            -1.3, 1.7, -0.5, 1.1, -1.5,
        ],
        dtype=float,
    )
    # Optional baseline override:
    # coil_currents = jnp.ones_like(coil_currents)
    if coil_currents.shape[0] != n_coils:
        raise ValueError(
            f"coil_currents length ({coil_currents.shape[0]}) must match n_coils ({n_coils})."
        )

    base_system = build_hemisphere_coil_system(
        n_coils=n_coils,
        sphere_radius=sphere_radius,
        coil_radius=coil_radius,
        turns=turns,
        currents=coil_currents,
    )

    opt_history = None
    baseline_mean_b = None
    final_mean_b = None
    improvement_factor = None
    if optimize_currents:
        # Arbitrary target points to maximize |B| at (edit as needed).
        # target_points = jnp.array(
        #     [
        #         [0.00, 0.00, 0.08],
        #         [0.03, 0.00, 0.07],
        #         [-0.02, 0.02, 0.09],
        #     ],
        #     dtype=float,
        # )
        target_points = jnp.array([[0.00, 0.00, 0.0]])
        baseline_b = base_system.field_at_points(target_points)
        baseline_mean_b = float(jnp.mean(jnp.linalg.norm(baseline_b, axis=1)))

        coil_currents, opt_history, opt_summary = optimize_currents_multi_start(
            template_system=base_system,
            initial_currents=coil_currents,
            target_points=target_points,
            per_coil_abs_max=per_coil_abs_max,
            total_abs_current_max=total_abs_current_max,
            n_steps=opt_n_steps,
            learning_rate=opt_learning_rate,
            n_restarts=opt_n_restarts,
            seed=opt_seed,
        )
        print("Optimization enabled.")
        print(f"  per-coil bound: [-{per_coil_abs_max:.1f}, {per_coil_abs_max:.1f}] A")
        print(f"  total abs current bound: {total_abs_current_max:.1f} A")
        print(f"  restarts: {opt_n_restarts} (seed={opt_seed})")
        print("  restart summary:")
        for row in opt_summary:
            print(
                f"    r{row['restart']:02d} [{row['type']}] "
                f"mean|B|={row['final_mean_b']:.6e} T  "
                f"sum|I|={row['final_total_abs_current']:.4f} A"
            )
        print(f"  final total abs current: {float(jnp.sum(jnp.abs(coil_currents))):.4f} A")
        final_mean_b = float(opt_history["mean_b"][-1])
        improvement_factor = final_mean_b / max(baseline_mean_b, 1e-20)
        print(f"  baseline mean |B| at targets: {baseline_mean_b:.6e} T")
        print(f"  final mean |B| at targets:    {final_mean_b:.6e} T")
        print(f"  improvement factor:            {improvement_factor:.3f}x")

    system = build_system_with_currents(base_system, coil_currents)

    coil_positions = np.asarray(jnp.stack([c.position for c in system.components]))
    currents_np = np.asarray(coil_currents)
    print_coil_mapping_table(coil_positions, currents_np)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    n_grid = 55
    lim = 0.16

    # Column 1: x-z view (y = 0)
    U, V, Bu, Bv, Bmag = sample_plane_field(
        system=system,
        axis_u=0, axis_v=2, axis_fixed=1, fixed_value=0.0,
        u_lim=(-lim, lim), v_lim=(-lim, lim), n=n_grid,
    )
    plot_plane(
        axes[0, 0], axes[1, 0],
        U, V, Bu, Bv, Bmag,
        coil_positions=coil_positions, currents=currents_np,
        axis_u=0, axis_v=2,
        title="x-z", label_u="x", label_v="z",
        show_indices=show_coil_indices,
    )

    # Column 2: y-z view (x = 0)
    U, V, Bu, Bv, Bmag = sample_plane_field(
        system=system,
        axis_u=1, axis_v=2, axis_fixed=0, fixed_value=0.0,
        u_lim=(-lim, lim), v_lim=(-lim, lim), n=n_grid,
    )
    plot_plane(
        axes[0, 1], axes[1, 1],
        U, V, Bu, Bv, Bmag,
        coil_positions=coil_positions, currents=currents_np,
        axis_u=1, axis_v=2,
        title="y-z", label_u="y", label_v="z",
        show_indices=show_coil_indices,
    )

    # Column 3: y-x view (z = 0)
    U, V, Bu, Bv, Bmag = sample_plane_field(
        system=system,
        axis_u=1, axis_v=0, axis_fixed=2, fixed_value=0.0,
        u_lim=(-lim, lim), v_lim=(-lim, lim), n=n_grid,
    )
    plot_plane(
        axes[0, 2], axes[1, 2],
        U, V, Bu, Bv, Bmag,
        coil_positions=coil_positions, currents=currents_np,
        axis_u=1, axis_v=0,
        title="y-x", label_u="y", label_v="x",
        show_indices=show_coil_indices,
    )

    if optimize_currents and improvement_factor is not None:
        fig.suptitle(
            "Hemispherical Coil Array: |B| Heatmaps (top) and Vector Fields (bottom)\n"
            f"Target mean |B| improvement: {improvement_factor:.3f}x",
            fontsize=14,
        )
    else:
        fig.suptitle(
            "Hemispherical Coil Array: |B| Heatmaps (top) and Vector Fields (bottom)",
            fontsize=14,
        )

    if opt_history is not None:
        fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        loss_np = np.asarray(opt_history["loss"])
        mean_b_np = np.asarray(opt_history["mean_b"])
        best_loss_so_far = np.minimum.accumulate(loss_np)
        best_mean_b_so_far = np.maximum.accumulate(mean_b_np)

        ax_loss.plot(loss_np, label="loss = -mean(|B| at targets)", alpha=0.7)
        ax_loss.plot(best_loss_so_far, label="best loss so far", linewidth=2.0)
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Current Optimization Loss Curve")
        ax_loss.grid(True, alpha=0.3)

        ax_aux = ax_loss.twinx()
        ax_aux.plot(
            np.asarray(opt_history["total_abs_current"]),
            color="tab:orange",
            alpha=0.7,
            label="sum(|I|)",
        )
        ax_aux.axhline(total_abs_current_max, color="tab:red", linestyle="--", linewidth=1.0, label="|I| budget")
        ax_aux.set_ylabel("current [A]")
        ax_obj = ax_loss.twinx()
        ax_obj.spines["right"].set_position(("axes", 1.12))
        ax_obj.plot(mean_b_np, color="tab:green", alpha=0.35, label="mean(|B|)")
        ax_obj.plot(best_mean_b_so_far, color="tab:green", linewidth=2.0, label="best mean(|B|) so far")
        ax_obj.set_ylabel("objective [T]")

        h1, l1 = ax_loss.get_legend_handles_labels()
        h2, l2 = ax_aux.get_legend_handles_labels()
        h3, l3 = ax_obj.get_legend_handles_labels()
        ax_aux.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right")

        if (
            baseline_mean_b is not None
            and final_mean_b is not None
            and improvement_factor is not None
        ):
            summary_text = (
                f"baseline mean |B|: {baseline_mean_b:.3e} T\n"
                f"final mean |B|: {final_mean_b:.3e} T\n"
                f"improvement: {improvement_factor:.3f}x"
            )
            ax_loss.text(
                0.02,
                0.02,
                summary_text,
                transform=ax_loss.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
            )

    plt.show()


if __name__ == "__main__":
    main()
