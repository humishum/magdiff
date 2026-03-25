from __future__ import annotations

from magdiff.visualization import GlyphFieldSpec, visualize_system


def visualize_field(
    system,
    region=((-1, 1), (-1, 1), (-1, 1)),
    grid_count=(10, 10, 10),
):
    """
    Compatibility wrapper around the new scene-based visualization stack.

    The legacy API used Plotly streamtubes directly. The new implementation
    returns a SceneViewer and approximates the legacy field display with
    sparse glyphs over the requested region.

    :param system: MagneticSystem object or MagneticComponent
    :param region: tuple of (x_range, y_range, z_range) each itself a (min, max)
    :param grid_count: tuple (Nx, Ny, Nz) number of grid points in each dimension
    :return: SceneViewer
    """
    bounds = tuple(tuple(axis) for axis in region)
    glyph_scale = min(
        (axis[1] - axis[0]) / max(count, 1)
        for axis, count in zip(bounds, grid_count)
    )
    return visualize_system(
        system,
        field=GlyphFieldSpec(bounds=bounds, shape=grid_count, scale=glyph_scale * 0.6),
        metadata={"title": "Magnetic Field Visualization"},
    )
