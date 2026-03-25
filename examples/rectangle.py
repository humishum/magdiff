""" 
An example showcasing 
field map of a uniformly magnetized rectangular magnet (cuboid)."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from magdiff.components.cuboid import Cuboid
from magdiff.system import MagneticSystem


def _add_magnet_patch(ax, length: float, height: float) -> None:
    """Draw the cuboid cross-section and N/S labels on the x-z plane."""
    ax.add_patch(
        Rectangle(
            (-length / 2, -height / 2),
            length / 2,
            height,
            edgecolor="black",
            facecolor="#1f77b4",
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, -height / 2),
            length / 2,
            height,
            edgecolor="black",
            facecolor="#d62728",
            zorder=5,
        )
    )
    ax.text(-length * 0.25, 0.0, "S", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(+length * 0.25, 0.0, "N", ha="center", va="center", fontsize=14, weight="bold")


def main() -> None:
    # Rectangular magnet centered at the origin, magnetized along +x.
    # N pole is on +x face; S pole is on -x face.
    dimension = jnp.array([0.08, 0.03, 0.03])  # (Lx, Ly, Lz) [m]
    system = MagneticSystem(
        [
            Cuboid(
                magnetization=jnp.array([8.0e5, 0.0, 0.0]),  # [A/m]
                dimension=dimension,
                position=jnp.array([0.0, 0.0, 0.0]),
                name="rectangle_magnet",
            )
        ],
        name="rectangle_system",
    )

    # Sample the field on an x-z slice (y=0).
    bounds = ((-0.18, 0.18), (-0.04, 0.04), (-0.12, 0.12))
    shape = (181, 11, 141)  # (Nx, Ny, Nz)
    volume = system.sample_field_volume(bounds=bounds, shape=shape)
    y_idx = shape[1] // 2

    xs = np.array(volume.X[:, y_idx, 0])              # (Nx,)
    zs = np.array(volume.Z[0, y_idx, :])              # (Nz,)
    B_slice = np.array(volume.B[:, y_idx, :, :])      # (Nx, Nz, 3)
    Bx = B_slice[:, :, 0].T                            # (Nz, Nx)
    Bz = B_slice[:, :, 2].T                            # (Nz, Nx)
    Bmag = np.hypot(Bx, Bz)

    # Plot magnitude + streamlines in one view.
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=140, constrained_layout=True)
    im = ax.pcolormesh(
        xs,
        zs,
        np.log10(Bmag + 1e-20),
        shading="auto",
        cmap="Greys",
        alpha=0.45,
    )
    ax.streamplot(xs, zs, Bx, Bz, density=1.8, color="black", linewidth=1.0, arrowsize=1.0)
    _add_magnet_patch(ax, float(dimension[0]), float(dimension[2]))

    ax.set_title("Rectangular Magnet Field (x-z plane, y=0)")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[2])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="log10(|B| [T])")

    plt.show()


if __name__ == "__main__":
    main()
