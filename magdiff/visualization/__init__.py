"""Public visualization API."""

from magdiff.visualization.builder import build_scene
from magdiff.visualization.overlays import GlyphFieldSpec, MarkerSpec, PlaneFieldSpec, StreamlineSpec
from magdiff.visualization.scene import Scene, SceneNode, VisualStyle
from magdiff.visualization.viewer import SceneViewer, visualize_system

__all__ = [
    "GlyphFieldSpec",
    "MarkerSpec",
    "PlaneFieldSpec",
    "Scene",
    "SceneNode",
    "SceneViewer",
    "StreamlineSpec",
    "VisualStyle",
    "build_scene",
    "visualize_system",
]
