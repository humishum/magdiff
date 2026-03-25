"""Renderer-agnostic scene models used by visualization backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class VisualStyle:
    """Basic styling shared across renderers."""

    color: str | tuple[float, float, float] | None = None
    opacity: float = 1.0
    wireframe: bool = False
    line_width: float = 3.0
    point_size: float = 0.02
    visible: bool = True
    colorscale: str | None = None
    showscale: bool = False
    colorbar_title: str | None = None
    colorbar_tickformat: str | None = None
    color_range: tuple[float, float] | None = None


@dataclass
class Primitive:
    """Smallest renderable unit emitted by adapters and overlay builders."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    style: VisualStyle = field(default_factory=VisualStyle)


@dataclass
class SceneNode:
    """Hierarchical scene node with a local transform and render primitives."""

    key: str
    kind: str
    label: str | None = None
    local_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    primitives: list[Primitive] = field(default_factory=list)
    children: list["SceneNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    style: VisualStyle | None = None

    def add_child(self, child: "SceneNode") -> None:
        self.children.append(child)


@dataclass
class Scene:
    """Single resolved snapshot of scene geometry and overlays."""

    root: SceneNode
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_nodes(self) -> Iterator[tuple[SceneNode, np.ndarray]]:
        """Yield nodes with accumulated world transforms."""
        yield from _iter_nodes(self.root, np.eye(4, dtype=float))

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute loose bounds from primitive control points."""
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for node, world_transform in self.iter_nodes():
            for primitive in node.primitives:
                points = primitive_reference_points(primitive)
                if points.size == 0:
                    continue
                world_points = apply_transform(points, world_transform)
                mins.append(world_points.min(axis=0))
                maxs.append(world_points.max(axis=0))
        if not mins:
            return None
        return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def _iter_nodes(
    node: SceneNode, parent_transform: np.ndarray
) -> Iterator[tuple[SceneNode, np.ndarray]]:
    world_transform = parent_transform @ np.asarray(node.local_transform, dtype=float)
    yield node, world_transform
    for child in node.children:
        yield from _iter_nodes(child, world_transform)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to 3D point sets."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
    return (np.asarray(transform, dtype=float) @ hom.T).T[:, :3]


def merge_style(node_style: VisualStyle | None, primitive_style: VisualStyle) -> VisualStyle:
    """Merge a node-level style with a primitive-level style."""
    if node_style is None:
        return primitive_style
    return VisualStyle(
        color=primitive_style.color if primitive_style.color is not None else node_style.color,
        opacity=primitive_style.opacity if primitive_style.opacity != 1.0 else node_style.opacity,
        wireframe=primitive_style.wireframe or node_style.wireframe,
        line_width=primitive_style.line_width
        if primitive_style.line_width != 3.0
        else node_style.line_width,
        point_size=primitive_style.point_size
        if primitive_style.point_size != 0.02
        else node_style.point_size,
        visible=primitive_style.visible and node_style.visible,
        colorscale=primitive_style.colorscale or node_style.colorscale,
        showscale=primitive_style.showscale or node_style.showscale,
        colorbar_title=primitive_style.colorbar_title or node_style.colorbar_title,
        colorbar_tickformat=primitive_style.colorbar_tickformat or node_style.colorbar_tickformat,
        color_range=primitive_style.color_range or node_style.color_range,
    )


def primitive_reference_points(primitive: Primitive) -> np.ndarray:
    """Reference points used for bounds and coarse renderer sizing."""
    kind = primitive.kind
    params = primitive.params
    if kind == "box":
        size = np.asarray(params["size"], dtype=float)
        hx, hy, hz = 0.5 * size
        return np.asarray(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=float,
        )
    if kind == "cylinder":
        radius = float(params["radius"])
        height = float(params["height"])
        z = 0.5 * height
        return np.asarray(
            [
                [-radius, -radius, -z],
                [radius, radius, z],
            ],
            dtype=float,
        )
    if kind == "sphere":
        radius = float(params["radius"])
        return np.asarray(
            [
                [-radius, -radius, -radius],
                [radius, radius, radius],
            ],
            dtype=float,
        )
    if kind in {"tube", "polyline", "points"}:
        return np.asarray(params["points"], dtype=float)
    if kind == "arrow":
        origin = np.asarray(params["origin"], dtype=float)
        direction = np.asarray(params["direction"], dtype=float)
        return np.vstack([origin, origin + direction])
    if kind == "text":
        return np.asarray(params["position"], dtype=float)[None, :]
    if kind == "axes":
        scale = float(params.get("scale", 1.0))
        return np.asarray(
            [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
            dtype=float,
        )
    if kind == "surface":
        x = np.asarray(params["x"], dtype=float)
        y = np.asarray(params["y"], dtype=float)
        z = np.asarray(params["z"], dtype=float)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        finite = np.all(np.isfinite(points), axis=1)
        return points[finite]
    return np.empty((0, 3), dtype=float)
