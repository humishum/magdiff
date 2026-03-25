"""Field overlay specifications and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from plotly.colors import sample_colorscale

from magdiff.components.dipole import Dipole
from magdiff.math import pose_to_matrix
from magdiff.system import MagneticSystem
from magdiff.visualization.scene import Primitive, SceneNode, VisualStyle


@dataclass
class PlaneFieldSpec:
    """Axis-aligned 2D field sample rendered as a 3D plane overlay."""

    plane: str
    fixed: float
    u_lim: tuple[float, float]
    v_lim: tuple[float, float]
    resolution: tuple[int, int] | int = (81, 81)
    quantity: str = "norm"
    show_vectors: bool = False
    vector_stride: int = 10
    vector_scale: float = 0.14
    vector_min_scale: float = 0.03
    vector_normalize: bool = True
    vector_color_by_magnitude: bool = True
    vector_percentile: float = 97.5
    color_percentile: float | None = 99.0
    exclusion_radius: float | None = None
    colorscale: str = "Viridis"
    opacity: float = 0.65

    @classmethod
    def xz(
        cls,
        *,
        y: float = 0.0,
        xlim: tuple[float, float] = (-0.1, 0.1),
        zlim: tuple[float, float] = (-0.1, 0.1),
        resolution: tuple[int, int] | int = (81, 81),
        quantity: str = "norm",
        show_vectors: bool = False,
        **kwargs,
    ) -> "PlaneFieldSpec":
        return cls(
            plane="xz",
            fixed=y,
            u_lim=xlim,
            v_lim=zlim,
            resolution=resolution,
            quantity=quantity,
            show_vectors=show_vectors,
            **kwargs,
        )

    @classmethod
    def yz(
        cls,
        *,
        x: float = 0.0,
        ylim: tuple[float, float] = (-0.1, 0.1),
        zlim: tuple[float, float] = (-0.1, 0.1),
        resolution: tuple[int, int] | int = (81, 81),
        quantity: str = "norm",
        show_vectors: bool = False,
        **kwargs,
    ) -> "PlaneFieldSpec":
        return cls(
            plane="yz",
            fixed=x,
            u_lim=ylim,
            v_lim=zlim,
            resolution=resolution,
            quantity=quantity,
            show_vectors=show_vectors,
            **kwargs,
        )

    @classmethod
    def xy(
        cls,
        *,
        z: float = 0.0,
        xlim: tuple[float, float] = (-0.1, 0.1),
        ylim: tuple[float, float] = (-0.1, 0.1),
        resolution: tuple[int, int] | int = (81, 81),
        quantity: str = "norm",
        show_vectors: bool = False,
        **kwargs,
    ) -> "PlaneFieldSpec":
        return cls(
            plane="xy",
            fixed=z,
            u_lim=xlim,
            v_lim=ylim,
            resolution=resolution,
            quantity=quantity,
            show_vectors=show_vectors,
            **kwargs,
        )

    @property
    def fixed_axis(self) -> str:
        return {"xz": "y", "yz": "x", "xy": "z"}[self.plane]


@dataclass
class GlyphFieldSpec:
    """Sparse vector glyph overlay inside a 3D box."""

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    shape: tuple[int, int, int] = (6, 6, 6)
    normalize: bool = True
    scale: float = 0.18
    min_scale: float = 0.04
    scale_by_magnitude: bool = True
    magnitude_percentile: float = 97.5
    exclusion_radius: float | None = None
    color: str = "#7c3aed"
    color_by_magnitude: bool = True
    colorscale: str = "Viridis"


@dataclass
class MarkerSpec:
    """Simple point markers with optional labels."""

    points: np.ndarray
    color: str = "#e63946"
    size: float = 0.12
    labels: list[str] | None = None


@dataclass
class StreamlineSpec:
    """Simple streamline overlay traced directly through the field."""

    seeds: np.ndarray
    step_size: float = 0.01
    n_steps: int = 48
    color: str = "#111111"
    line_width: float = 2.0


def sample_plane_field(system: MagneticSystem, spec: PlaneFieldSpec) -> dict[str, np.ndarray]:
    """Sample magnetic field values over one axis-aligned plane."""
    nu, nv = _normalize_resolution(spec.resolution)
    u = jnp.linspace(spec.u_lim[0], spec.u_lim[1], nu)
    v = jnp.linspace(spec.v_lim[0], spec.v_lim[1], nv)
    U, V = jnp.meshgrid(u, v, indexing="xy")

    axis_u, axis_v, axis_fixed = _plane_axes(spec.plane)
    points = jnp.zeros((U.size, 3), dtype=float)
    points = points.at[:, axis_u].set(U.ravel())
    points = points.at[:, axis_v].set(V.ravel())
    points = points.at[:, axis_fixed].set(spec.fixed)

    B = system.field_at_points(points).reshape(U.shape + (3,))
    quantity = _plane_quantity(B, spec.quantity)
    valid_mask = _point_valid_mask(points, system, spec.exclusion_radius).reshape(U.shape)

    coords = np.zeros(U.shape + (3,), dtype=float)
    coords[..., axis_u] = np.asarray(U)
    coords[..., axis_v] = np.asarray(V)
    coords[..., axis_fixed] = spec.fixed

    return {
        "coords": coords,
        "B": np.asarray(B),
        "quantity": np.asarray(quantity),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
    }


def build_overlay_node(system: MagneticSystem, spec: object, key: str) -> SceneNode:
    """Build a scene node for a field overlay specification."""
    if isinstance(spec, PlaneFieldSpec):
        return build_plane_overlay(system, spec, key)
    if isinstance(spec, GlyphFieldSpec):
        return build_glyph_overlay(system, spec, key)
    if isinstance(spec, MarkerSpec):
        return build_marker_overlay(spec, key)
    if isinstance(spec, StreamlineSpec):
        return build_streamline_overlay(system, spec, key)
    raise TypeError(f"Unsupported field overlay spec {type(spec).__name__}")


def build_plane_overlay(system: MagneticSystem, spec: PlaneFieldSpec, key: str) -> SceneNode:
    data = sample_plane_field(system, spec)
    coords = data["coords"]
    quantity = np.asarray(data["quantity"], dtype=float)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    quantity_label = _plane_quantity_label(spec.quantity)
    color_range = _color_range_for_quantity(
        quantity,
        valid_mask=valid_mask,
        quantity=spec.quantity,
        percentile=spec.color_percentile,
    )
    display_quantity = quantity.copy()
    if color_range is not None:
        display_quantity = np.clip(display_quantity, color_range[0], color_range[1])
    display_quantity[~valid_mask] = np.nan
    display_coords = np.asarray(coords, dtype=float).copy()
    display_coords[~valid_mask] = np.nan
    primitives = [
        Primitive(
            kind="surface",
            params={
                "x": display_coords[..., 0],
                "y": display_coords[..., 1],
                "z": display_coords[..., 2],
                "surfacecolor": display_quantity,
            },
            style=VisualStyle(
                colorscale=spec.colorscale,
                opacity=spec.opacity,
                showscale=True,
                colorbar_title=f"{quantity_label} (T)",
                colorbar_tickformat=".2e",
                color_range=color_range,
            ),
        )
    ]

    if spec.show_vectors:
        B = data["B"]
        stride = max(1, int(spec.vector_stride))
        sample_points = coords[::stride, ::stride, :].reshape(-1, 3)
        sample_vectors = B[::stride, ::stride, :].reshape(-1, 3)
        sample_valid = valid_mask[::stride, ::stride].reshape(-1)
        sample_points = sample_points[sample_valid]
        sample_vectors = sample_vectors[sample_valid]
        relative = _relative_magnitudes(sample_vectors, spec.vector_percentile)
        scaled = _scaled_vectors(
            sample_vectors,
            normalize=spec.vector_normalize,
            max_scale=spec.vector_scale,
            min_scale=spec.vector_min_scale,
            relative=relative,
        )
        for point, direction, rel in zip(sample_points, scaled, relative):
            primitives.append(
                Primitive(
                    kind="arrow",
                    params={
                        "origin": np.asarray(point, dtype=float),
                        "direction": np.asarray(direction, dtype=float),
                        "show_head": False,
                    },
                    style=VisualStyle(
                        color=_color_for_relative(rel, spec.colorscale)
                        if spec.vector_color_by_magnitude
                        else "#111111",
                        line_width=2.0,
                    ),
                )
            )

    return SceneNode(
        key=key,
        kind="field_plane",
        label=f"{quantity_label} on {spec.plane.upper()} @ {spec.fixed_axis}={spec.fixed:.3f} m",
        primitives=primitives,
        metadata={
            "spec": spec,
            "quantity": spec.quantity,
            "quantity_label": quantity_label,
            "plane_label": f"{spec.plane.upper()} slice at {spec.fixed_axis.upper()} = {spec.fixed:.3f} m",
            "exclusion_radius": spec.exclusion_radius,
        },
    )


def build_glyph_overlay(system: MagneticSystem, spec: GlyphFieldSpec, key: str) -> SceneNode:
    (x_range, y_range, z_range) = spec.bounds
    nx, ny, nz = spec.shape
    xs = jnp.linspace(x_range[0], x_range[1], nx)
    ys = jnp.linspace(y_range[0], y_range[1], ny)
    zs = jnp.linspace(z_range[0], z_range[1], nz)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    points_np = np.asarray(points, dtype=float)
    valid_mask = _point_valid_mask(points_np, system, spec.exclusion_radius)
    points_np = points_np[valid_mask]
    if len(points_np) == 0:
        return SceneNode(
            key=key,
            kind="field_glyphs",
            label="field_glyphs",
            primitives=[],
            metadata={"spec": spec, "magnitude_relative": np.asarray([], dtype=float)},
        )
    B = np.asarray(system.field_at_points(jnp.asarray(points_np, dtype=float)))
    relative = _relative_magnitudes(B, spec.magnitude_percentile)
    vectors = _scaled_vectors(
        B,
        normalize=spec.normalize,
        max_scale=spec.scale,
        min_scale=spec.min_scale,
        relative=relative if spec.scale_by_magnitude else np.ones_like(relative),
    )

    primitives = [
        Primitive(
            kind="arrow",
            params={"origin": np.asarray(point), "direction": np.asarray(direction), "show_head": False},
            style=VisualStyle(
                color=_color_for_relative(rel, spec.colorscale) if spec.color_by_magnitude else spec.color,
                line_width=2.0,
            ),
        )
        for point, direction, rel in zip(points_np, vectors, relative)
    ]
    return SceneNode(
        key=key,
        kind="field_glyphs",
        label="field_glyphs",
        primitives=primitives,
        metadata={"spec": spec, "magnitude_relative": relative},
    )


def build_marker_overlay(spec: MarkerSpec, key: str) -> SceneNode:
    points = np.asarray(spec.points, dtype=float)
    primitives = [
        Primitive(
            kind="points",
            params={"points": points},
            style=VisualStyle(color=spec.color, point_size=spec.size),
        )
    ]
    if spec.labels:
        for point, label in zip(points, spec.labels):
            primitives.append(
                Primitive(
                    kind="text",
                    params={"text": label, "position": np.asarray(point, dtype=float)},
                    style=VisualStyle(color=spec.color),
                )
            )
    return SceneNode(
        key=key,
        kind="markers",
        label="markers",
        primitives=primitives,
        metadata={"spec": spec},
    )


def build_streamline_overlay(
    system: MagneticSystem, spec: StreamlineSpec, key: str
) -> SceneNode:
    primitives: list[Primitive] = []
    for seed in np.asarray(spec.seeds, dtype=float):
        points = [np.asarray(seed, dtype=float)]
        current = jnp.asarray(seed, dtype=float)
        for _ in range(int(spec.n_steps)):
            field = np.asarray(system.field_at(current), dtype=float)
            norm = np.linalg.norm(field)
            if not np.isfinite(norm) or norm < 1e-12:
                break
            current = current + jnp.asarray(field / norm * spec.step_size, dtype=float)
            points.append(np.asarray(current, dtype=float))
        if len(points) > 1:
            primitives.append(
                Primitive(
                    kind="polyline",
                    params={"points": np.asarray(points, dtype=float)},
                    style=VisualStyle(color=spec.color, line_width=spec.line_width),
                )
            )
    return SceneNode(
        key=key,
        kind="streamlines",
        label="streamlines",
        primitives=primitives,
        metadata={"spec": spec},
    )


def normalize_field_specs(field: object | Sequence[object] | None) -> list[object]:
    if field is None:
        return []
    if isinstance(field, (PlaneFieldSpec, GlyphFieldSpec, MarkerSpec, StreamlineSpec)):
        return [field]
    return list(field)


def _relative_magnitudes(vectors: np.ndarray, percentile: float) -> np.ndarray:
    magnitudes = np.linalg.norm(np.asarray(vectors, dtype=float), axis=1)
    finite = np.isfinite(magnitudes)
    if not np.any(finite):
        return np.zeros_like(magnitudes)
    clip_value = np.nanpercentile(magnitudes[finite], percentile)
    clip_value = max(float(clip_value), 1e-12)
    clipped = np.clip(magnitudes, 0.0, clip_value)
    return clipped / clip_value


def _color_range_for_quantity(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    quantity: str,
    percentile: float | None,
) -> tuple[float, float] | None:
    finite_mask = np.isfinite(values) & valid_mask
    if not np.any(finite_mask):
        return None

    valid_values = np.asarray(values[finite_mask], dtype=float)
    if quantity == "norm":
        lo = float(np.nanmin(valid_values))
        hi = float(np.nanmax(valid_values))
        if percentile is not None:
            hi = float(np.nanpercentile(valid_values, percentile))
        hi = max(hi, lo + 1e-12)
        return lo, hi

    limit = float(np.nanmax(np.abs(valid_values)))
    if percentile is not None:
        limit = float(np.nanpercentile(np.abs(valid_values), percentile))
    limit = max(limit, 1e-12)
    return -limit, limit


def _point_valid_mask(
    points: jnp.ndarray | np.ndarray,
    system: MagneticSystem,
    exclusion_radius: float | None,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if exclusion_radius is None or exclusion_radius <= 0.0:
        return np.ones(points.shape[0], dtype=bool)

    centers = _collect_world_dipole_centers(system)
    if centers.size == 0:
        return np.ones(points.shape[0], dtype=bool)

    deltas = points[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return np.all(distances >= float(exclusion_radius), axis=1)


def _collect_world_dipole_centers(system: MagneticSystem) -> np.ndarray:
    centers: list[np.ndarray] = []

    def _visit(obj, parent_transform: np.ndarray) -> None:
        local = np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float)
        world = parent_transform @ local
        if isinstance(obj, Dipole):
            centers.append(world[:3, 3].copy())
        if isinstance(obj, MagneticSystem):
            for child in obj.components:
                _visit(child, world)

    _visit(system, np.eye(4, dtype=float))
    if not centers:
        return np.empty((0, 3), dtype=float)
    return np.stack(centers, axis=0)


def _scaled_vectors(
    vectors: np.ndarray,
    *,
    normalize: bool,
    max_scale: float,
    min_scale: float,
    relative: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.where(norms > 0.0, vectors / np.maximum(norms, 1e-12), 0.0)
    lengths = float(min_scale) + (float(max_scale) - float(min_scale)) * np.sqrt(relative)
    if normalize:
        return unit * lengths[:, None]
    scale = np.maximum(relative, 1e-12)[:, None] * float(max_scale)
    return unit * scale


def _color_for_relative(relative: float, colorscale: str) -> str:
    return sample_colorscale(colorscale, [float(np.clip(relative, 0.0, 1.0))])[0]


def _normalize_resolution(resolution: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(resolution, int):
        return resolution, resolution
    return resolution


def _plane_axes(plane: str) -> tuple[int, int, int]:
    mapping = {
        "xz": (0, 2, 1),
        "yz": (1, 2, 0),
        "xy": (0, 1, 2),
    }
    if plane not in mapping:
        raise ValueError(f"Unsupported plane {plane!r}")
    return mapping[plane]


def _plane_quantity(B: jnp.ndarray, quantity: str) -> jnp.ndarray:
    if quantity == "norm":
        return jnp.linalg.norm(B, axis=-1)
    if quantity == "x":
        return B[..., 0]
    if quantity == "y":
        return B[..., 1]
    if quantity == "z":
        return B[..., 2]
    raise ValueError(f"Unsupported plane quantity {quantity!r}")


def _plane_quantity_label(quantity: str) -> str:
    if quantity == "norm":
        return "|B|"
    return f"B{quantity}"
