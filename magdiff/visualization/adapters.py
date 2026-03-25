"""Adapters that map magnetic objects into scene nodes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import numpy as np

from magdiff.components.coil import Coil
from magdiff.components.cuboid import Cuboid
from magdiff.components.cylinder import Cylinder
from magdiff.components.dipole import Dipole
from magdiff.math import pose_to_matrix
from magdiff.system import MagneticSystem
from magdiff.visualization.scene import Primitive, SceneNode, VisualStyle

VisualAdapter = Callable[[object, str, "VisualContext"], SceneNode]

DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "system": {"color": "#666666"},
    "Cuboid": {"color": "#d95f02", "opacity": 0.9},
    "Cylinder": {"color": "#1f77b4", "opacity": 0.9},
    "Coil": {"color": "#b87333", "line_width": 4.0},
    "Dipole": {"color": "#2a9d8f", "line_width": 4.0},
}
VISUAL_STYLE_FIELDS = {field.name for field in fields(VisualStyle)}


@dataclass
class VisualContext:
    """Context used while adapting objects into scene nodes."""

    registry: "AdapterRegistry"
    styles: Mapping[str, Mapping[str, Any]] | None = None

    def resolve_options(
        self,
        obj: object,
        *,
        kind: str,
        fallback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if fallback is not None:
            options.update(fallback)
        name = getattr(obj, "name", None)
        if kind in DEFAULT_STYLES:
            options.update(DEFAULT_STYLES[kind])
        cls_name = type(obj).__name__
        if cls_name in DEFAULT_STYLES:
            options.update(DEFAULT_STYLES[cls_name])
        if self.styles:
            if kind in self.styles:
                options.update(self.styles[kind])
            if cls_name in self.styles:
                options.update(self.styles[cls_name])
            if name and name in self.styles:
                options.update(self.styles[name])
        return options

    def resolve_style(
        self,
        obj: object,
        *,
        kind: str,
        fallback: Mapping[str, Any] | None = None,
    ) -> VisualStyle:
        style_data = self.resolve_options(obj, kind=kind, fallback=fallback)
        return VisualStyle(
            **{
                key: value
                for key, value in style_data.items()
                if key in VISUAL_STYLE_FIELDS
            }
        )


class AdapterRegistry:
    """Registry that resolves scene adapters by Python type."""

    def __init__(self):
        self._adapters: dict[type, VisualAdapter] = {}

    def register(self, cls: type, adapter: VisualAdapter) -> None:
        self._adapters[cls] = adapter

    def get_adapter(self, obj: object) -> VisualAdapter:
        for cls in type(obj).__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        raise TypeError(f"No visualization adapter registered for {type(obj).__name__}")

    def adapt(self, obj: object, key: str, context: VisualContext) -> SceneNode:
        return self.get_adapter(obj)(obj, key, context)


def create_default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(MagneticSystem, adapt_system)
    registry.register(Cuboid, adapt_cuboid)
    registry.register(Cylinder, adapt_cylinder)
    registry.register(Coil, adapt_coil)
    registry.register(Dipole, adapt_dipole)
    return registry


def adapt_system(obj: MagneticSystem, key: str, context: VisualContext) -> SceneNode:
    return SceneNode(
        key=key,
        kind="system",
        label=obj.name,
        local_transform=np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float),
        metadata={"component_count": len(obj.components), "component_type": type(obj).__name__},
        style=context.resolve_style(obj, kind="system"),
    )


def adapt_cuboid(obj: Cuboid, key: str, context: VisualContext) -> SceneNode:
    return SceneNode(
        key=key,
        kind="cuboid",
        label=obj.name,
        local_transform=np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float),
        primitives=[
            Primitive(
                kind="box",
                params={"size": np.asarray(obj.dimension, dtype=float)},
                style=context.resolve_style(obj, kind="cuboid"),
            )
        ],
        metadata={
            "component_type": type(obj).__name__,
            "magnetization": np.asarray(obj.magnetization, dtype=float),
        },
    )


def adapt_cylinder(obj: Cylinder, key: str, context: VisualContext) -> SceneNode:
    return SceneNode(
        key=key,
        kind="cylinder",
        label=obj.name,
        local_transform=np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float),
        primitives=[
            Primitive(
                kind="cylinder",
                params={"radius": 0.5 * float(obj.diameter), "height": float(obj.height)},
                style=context.resolve_style(obj, kind="cylinder"),
            )
        ],
        metadata={
            "component_type": type(obj).__name__,
            "magnetization": np.asarray(obj.magnetization, dtype=float),
            "diameter": float(obj.diameter),
            "height": float(obj.height),
        },
    )


def adapt_coil(obj: Coil, key: str, context: VisualContext) -> SceneNode:
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=True)
    circle = np.stack(
        [
            obj.radius * np.cos(theta),
            obj.radius * np.sin(theta),
            np.zeros_like(theta),
        ],
        axis=1,
    )
    direction_sign = 1.0 if float(np.asarray(obj.current)) >= 0.0 else -1.0
    arrow_origin = np.array([obj.radius, 0.0, 0.0], dtype=float)
    arrow_direction = np.array([0.0, direction_sign * obj.radius * 0.6, 0.0], dtype=float)
    return SceneNode(
        key=key,
        kind="coil",
        label=obj.name,
        local_transform=np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float),
        primitives=[
            Primitive(
                kind="tube",
                params={"points": circle, "closed": True, "radius": obj.radius * 0.04},
                style=context.resolve_style(obj, kind="coil"),
            ),
            Primitive(
                kind="arrow",
                params={"origin": arrow_origin, "direction": arrow_direction, "show_head": True},
                style=context.resolve_style(obj, kind="coil"),
            ),
        ],
        metadata={
            "component_type": type(obj).__name__,
            "current": float(np.asarray(obj.current)),
            "turns": int(obj.turns),
            "radius": float(obj.radius),
        },
    )


def adapt_dipole(obj: Dipole, key: str, context: VisualContext) -> SceneNode:
    options = context.resolve_options(obj, kind="dipole")
    dipole_style = context.resolve_style(obj, kind="dipole")
    moment = np.asarray(obj.moment, dtype=float)
    norm = np.linalg.norm(moment)
    arrow_length = float(options.get("arrow_length", 0.45))
    sphere_radius = float(options.get("sphere_radius", 0.12))
    body_color = options.get("body_color", "#f4a261")
    body_opacity = float(options.get("body_opacity", 0.95))
    label_color = options.get("label_color", "#264653")
    scaled = moment if norm == 0.0 else moment / norm * arrow_length
    primitives = [
        Primitive(
            kind="sphere",
            params={"radius": sphere_radius},
            style=VisualStyle(color=body_color, opacity=body_opacity),
        ),
        Primitive(
            kind="arrow",
            params={"origin": np.zeros(3, dtype=float), "direction": scaled, "show_head": True},
            style=dipole_style,
        ),
    ]
    if obj.name:
        primitives.append(
            Primitive(
                kind="text",
                params={"text": obj.name, "position": np.array([0.0, 0.0, sphere_radius * 1.8], dtype=float)},
                style=VisualStyle(color=label_color),
            )
        )
    return SceneNode(
        key=key,
        kind="dipole",
        label=obj.name,
        local_transform=np.asarray(pose_to_matrix(obj.position, obj.rotation_vector), dtype=float),
        primitives=primitives,
        metadata={
            "component_type": type(obj).__name__,
            "moment": moment,
        },
    )
