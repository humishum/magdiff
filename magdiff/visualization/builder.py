"""Scene construction entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

from magdiff.system import MagneticSystem
from magdiff.visualization.adapters import VisualContext, create_default_registry
from magdiff.visualization.overlays import build_overlay_node, normalize_field_specs
from magdiff.visualization.scene import Scene, SceneNode


def build_scene(
    obj: object,
    *,
    field: object | list[object] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    show_axes: bool = True,
) -> Scene:
    """Build one fully-resolved scene from a magnetic component or system."""
    registry = create_default_registry()
    context = VisualContext(registry=registry, styles=styles)
    scene_root = SceneNode(key="scene", kind="scene", label="scene")
    scene_root.add_child(_build_node(obj, key="root", context=context))

    if isinstance(obj, MagneticSystem):
        for idx, overlay_spec in enumerate(normalize_field_specs(field)):
            scene_root.add_child(build_overlay_node(obj, overlay_spec, key=f"scene/overlay_{idx}"))

    scene_metadata = dict(metadata or {})
    scene_metadata.setdefault("show_axes", show_axes)
    return Scene(root=scene_root, metadata=scene_metadata)


def _build_node(obj: object, *, key: str, context: VisualContext):
    node = context.registry.adapt(obj, key, context)
    if isinstance(obj, MagneticSystem):
        for idx, child in enumerate(obj.components):
            node.add_child(_build_node(child, key=f"{key}/{idx}", context=context))
    return node
