"""
LaPoint-like radial magnet-array field visualization (2D slice).
see: https://patents.google.com/patent/US8638186B1/en for more details
This example builds a ing-style magnet array in the y-z plane and plots:
  - background heatmap of log10(|B|)
  - magnetic field lines (streamplot) using in-plane components (By, Bz)

Note:
  - Field sources here use dipole approximation.
  - This is a qualitative visualization inspired by LaPoint-style radial layouts.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem


def build_lapoint_like_radial_array(
    n_arms: int = 16,
    mags_per_arm: int = 6,
    inner_radius: float = 0.035,
    radial_pitch: float = 0.014,
    mount_on_hemisphere: bool = True,
    hemisphere_radius: float = 0.13,
    moment_mag: float = 1.3,
    moment_mode: str = "inward_normal",
    polarity_pattern: str = "checker",
) -> MagneticSystem:
    """Create a dense radial/spoked array in the y-z plane.

    Magnets are positioned along `n_arms` radial spokes, each with `mags_per_arm` magnets.
    If `mount_on_hemisphere=True`, each magnet is placed on x>0 hemisphere:
      x = sqrt(R^2 - y^2 - z^2)
    so the visible y-z layout is the projection of a hemispherical mount.

    Moment direction options:
      - inward_normal:  towards hemisphere center
      - outward_normal: away from hemisphere center
      - x_axis:         fixed +/-x direction
    """
    if n_arms < 3:
        raise ValueError(f"n_arms must be >= 3, got {n_arms}")
    if mags_per_arm < 1:
        raise ValueError(f"mags_per_arm must be >= 1, got {mags_per_arm}")

    magnets = []
    idx = 0
    for arm in range(n_arms):
        theta = 2.0 * jnp.pi * arm / n_arms
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        for step in range(mags_per_arm):
            radius = inner_radius + step * radial_pitch
            if mount_on_hemisphere and radius >= hemisphere_radius:
                raise ValueError(
                    f"Radial distance {float(radius):.4f} exceeds hemisphere radius "
                    f"{hemisphere_radius:.4f}. Increase hemisphere_radius or reduce array size."
                )
            y = float(radius * ct)
            z = float(radius * st)
            if mount_on_hemisphere:
                x = float(jnp.sqrt(jnp.maximum(0.0, hemisphere_radius**2 - radius**2)))
            else:
                x = 0.0

            pos = jnp.array([x, y, z], dtype=float)
            pos_norm = jnp.maximum(jnp.linalg.norm(pos), 1e-12)
            n_out = pos / pos_norm
            n_in = -n_out

            if polarity_pattern == "checker":
                polarity = -1.0 if (arm + step) % 2 else 1.0
            elif polarity_pattern == "arm_alternating":
                polarity = -1.0 if arm % 2 else 1.0
            elif polarity_pattern == "radial_alternating":
                polarity = -1.0 if step % 2 else 1.0
            else:
                raise ValueError(f"Unknown polarity_pattern '{polarity_pattern}'.")
            polarity = 1.0
            if moment_mode == "inward_normal":
                base_dir = n_in
            elif moment_mode == "outward_normal":
                base_dir = n_out
            elif moment_mode == "x_axis":
                base_dir = jnp.array([1.0, 0.0, 0.0], dtype=float)
            else:
                raise ValueError(f"Unknown moment_mode '{moment_mode}'.")

            magnets.append(
                Dipole(
                    moment=polarity * moment_mag * base_dir,
                    position=pos,
                    name=f"M{idx:03d}",
                )
            )
            idx += 1
    return MagneticSystem(magnets, name="lapoint_like_radial")


def main():
    # ----------------------------
    # User-tunable parameters
    # ----------------------------
    n_arms = 16
    mags_per_arm = 6
    inner_radius = 0.035
    radial_pitch = 0.014
    mount_on_hemisphere = True
    hemisphere_radius = 0.13
    moment_mag = 1.3
    moment_mode = "inward_normal"  # inward_normal, outward_normal, x_axis
    polarity_pattern = "checker"  # options: checker, arm_alternating, radial_alternating

    # y-z plane at x = 0
    y_lim = (-0.14, 0.14)
    z_lim = (-0.14, 0.14)
    n = 180
    exclusion_radius = 0.007
    clip_percentile = 99.5

    system = build_lapoint_like_radial_array(
        n_arms=n_arms,
        mags_per_arm=mags_per_arm,
        inner_radius=inner_radius,
        radial_pitch=radial_pitch,
        mount_on_hemisphere=mount_on_hemisphere,
        hemisphere_radius=hemisphere_radius,
        moment_mag=moment_mag,
        moment_mode=moment_mode,
        polarity_pattern=polarity_pattern,
    )

    ys = jnp.linspace(y_lim[0], y_lim[1], n)
    zs = jnp.linspace(z_lim[0], z_lim[1], n)
    Y, Z = jnp.meshgrid(ys, zs, indexing="xy")  # (n, n)
    X = jnp.zeros_like(Y)

    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    B = system.field_at_points(points).reshape((n, n, 3))
    By = np.asarray(B[..., 1])
    Bz = np.asarray(B[..., 2])
    Bmag = np.asarray(jnp.linalg.norm(B, axis=-1))

    # Mask near singular points around each magnet center.
    singular_mask = np.zeros_like(Bmag, dtype=bool)
    for m in system.components:
        my = float(m.position[1])
        mz = float(m.position[2])
        singular_mask |= np.sqrt((np.asarray(Y) - my) ** 2 + (np.asarray(Z) - mz) ** 2) < exclusion_radius

    # Clip extreme peaks to improve visibility.
    finite = np.isfinite(Bmag)
    if np.any(finite):
        b_clip = np.nanpercentile(Bmag[finite], clip_percentile)
    else:
        b_clip = np.nanmax(Bmag)
    b_clip = max(float(b_clip), 1e-12)

    field_mask = singular_mask | (Bmag > b_clip)
    By_plot = np.where(field_mask, np.nan, By)
    Bz_plot = np.where(field_mask, np.nan, Bz)
    Bmag_plot = np.where(field_mask, np.nan, np.minimum(Bmag, b_clip))

    fig, ax = plt.subplots(figsize=(8.8, 8.0), constrained_layout=True)

    heat = ax.pcolormesh(
        np.asarray(ys),
        np.asarray(zs),
        np.log10(Bmag_plot + 1e-20),
        shading="auto",
        cmap="coolwarm",
        alpha=0.75,
    )
    cbar = fig.colorbar(heat, ax=ax, shrink=0.92)
    cbar.set_label("log10(|B| [T])")

    # Streamlines over in-plane components.
    ax.streamplot(
        np.asarray(ys),
        np.asarray(zs),
        np.nan_to_num(By_plot, nan=0.0),
        np.nan_to_num(Bz_plot, nan=0.0),
        density=1.9,
        color="midnightblue",
        linewidth=1.1,
        arrowsize=1.4,
    )

    # Explicit arrow overlay for readability.
    by_vis = np.nan_to_num(By_plot, nan=0.0)
    bz_vis = np.nan_to_num(Bz_plot, nan=0.0)
    b_inplane = np.hypot(by_vis, bz_vis)
    b95 = np.nanpercentile(b_inplane, 95.0)
    bmask = b_inplane <= b95
    by_q = np.where(bmask, by_vis, np.nan)
    bz_q = np.where(bmask, bz_vis, np.nan)
    bnorm = np.hypot(by_q, bz_q) + 1e-12
    Uy = by_q / bnorm
    Uz = bz_q / bnorm
    step = max(1, n // 24)
    ax.quiver(
        np.asarray(Y)[::step, ::step],
        np.asarray(Z)[::step, ::step],
        Uy[::step, ::step],
        Uz[::step, ::step],
        color="navy",
        alpha=0.55,
        angles="xy",
        scale_units="xy",
        scale=36,
        width=0.0025,
        pivot="mid",
        zorder=4,
    )

    # Draw magnet markers as circles (polarity shown by fill color).
    marker_radius = 0.0048
    for m in system.components:
        y = float(m.position[1])
        z = float(m.position[2])
        n_in = -m.position / jnp.maximum(jnp.linalg.norm(m.position), 1e-12)
        polarity = float(jnp.sign(jnp.dot(m.moment, n_in)))
        face = "black" if polarity >= 0.0 else "white"
        ax.add_patch(
            Circle(
                (y, z),
                marker_radius,
                facecolor=face,
                edgecolor="black",
                zorder=5,
            )
        )

    ax.set_title(
        "Magnetic field of LaPoint-like radial array "
        f"({n_arms} arms x {mags_per_arm} magnets, {polarity_pattern}, "
        f"{'hemisphere' if mount_on_hemisphere else 'flat'}, {moment_mode})"
    )
    ax.set_xlabel("y-position (m)")
    ax.set_ylabel("z-position (m)")
    ax.set_xlim(y_lim)
    ax.set_ylim(z_lim)
    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
