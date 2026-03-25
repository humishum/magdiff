"""Interactive single-scene viewer built on top of the scene model."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import warnings
import webbrowser
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from magdiff.system import MagneticSystem
from magdiff.visualization.builder import build_scene
from magdiff.visualization.overlays import PlaneFieldSpec
from magdiff.visualization.scene import (
    Primitive,
    Scene,
    VisualStyle,
    apply_transform,
    merge_style,
)

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - optional dependency path
    widgets = None


class _FallbackSlider:
    """Minimal slider-like object for environments without ipywidgets."""

    def __init__(
        self,
        *,
        description: str,
        min: float,
        max: float,
        step: float,
        value: float,
    ):
        self.description = description
        self.min = min
        self.max = max
        self.step = step
        self._value = value
        self._observers: list[Any] = []

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        old_value = self._value
        self._value = new_value
        change = {"name": "value", "old": old_value, "new": new_value, "owner": self}
        for handler in list(self._observers):
            handler(change)

    def observe(self, handler, names="value") -> None:  # noqa: ARG002 - matches widget API
        self._observers.append(handler)


class SceneViewer:
    """Interactive viewer that rebuilds and rerenders a single scene."""

    def __init__(
        self,
        obj: object,
        *,
        field: object | list[object] | None = None,
        styles: Mapping[str, Mapping[str, Any]] | None = None,
        controls: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        show_axes: bool = True,
        backend: str = "auto",
    ):
        self._obj = obj
        self._field = deepcopy(field)
        self._styles = styles
        self._metadata = metadata or {}
        self._show_axes = show_axes
        self._backend = self._resolve_backend(backend)
        self._render_count = 0
        self._last_export_path: Path | None = None
        self._control_specs: dict[str, Mapping[str, Any]] = {}
        self._control_widgets: dict[str, Any] = {}
        self._plot_obj: Any = None
        self._output = widgets.Output() if widgets is not None else None
        self._controls_box = widgets.VBox([]) if widgets is not None else None
        self.scene: Scene = build_scene(
            obj,
            field=self._field,
            styles=self._styles,
            metadata=self._metadata,
            show_axes=self._show_axes,
        )
        self._render_current_scene()
        if controls:
            self.set_controls(controls)

    @property
    def widget(self):
        """Notebook-friendly render payload."""
        if widgets is None:
            return self._plot_obj
        if self._controls_box is not None and len(self._controls_box.children) > 0:
            return widgets.VBox([self._controls_box, self._output])
        return self._output

    def show(self):
        """Display the viewer in notebooks or open the plot in scripts."""
        if widgets is not None and _in_ipython_kernel():
            try:
                from IPython.display import display

                display(self.widget)
                return self.widget
            except ImportError:  # pragma: no cover - script fallback
                pass
        if self._backend == "plotly":
            return self._open_html(self._plot_obj.to_html(include_plotlyjs="cdn"), prefix="magdiff_plotly_")
        if self._backend == "k3d":
            return self._open_html(self._plot_obj.get_snapshot(), prefix="magdiff_k3d_")
        return self._plot_obj

    def refresh(self):
        """Rebuild scene state from the current object and rerender it."""
        self.scene = build_scene(
            self._obj,
            field=self._field,
            styles=self._styles,
            metadata=self._metadata,
            show_axes=self._show_axes,
        )
        self._render_current_scene()
        return self

    def set_controls(self, controls: Mapping[str, Mapping[str, Any]]):
        """Attach numeric controls that mutate the source object or field spec."""
        self._control_specs = dict(controls)
        built_widgets = []
        for name, spec in self._control_specs.items():
            target = spec["target"]
            min_value = float(spec["min"])
            max_value = float(spec["max"])
            step = float(spec.get("step", max((max_value - min_value) / 100.0, 1e-3)))
            default_value = float(spec.get("value", self._get_target_value(target)))
            if widgets is None:
                slider = _FallbackSlider(
                    description=str(name),
                    min=min_value,
                    max=max_value,
                    step=step,
                    value=default_value,
                )
            else:
                slider = widgets.FloatSlider(
                    description=str(name),
                    min=min_value,
                    max=max_value,
                    step=step,
                    value=default_value,
                    continuous_update=False,
                    readout_format=spec.get("readout_format", ".3f"),
                )
            slider.observe(self._make_control_handler(target), names="value")
            built_widgets.append(slider)
            self._control_widgets[name] = slider

        if self._controls_box is not None:
            self._controls_box.children = tuple(built_widgets)
        return self

    def _make_control_handler(self, target: str):
        def _handler(change):
            if change.get("name") != "value":
                return
            self._set_target_value(target, change["new"])
            self.refresh()

        return _handler

    def _get_target_value(self, target: str) -> float:
        if target.startswith("field."):
            if self._field is None:
                raise ValueError("Field controls require a field overlay specification.")
            if not isinstance(self._field, PlaneFieldSpec):
                raise ValueError("V1 field controls currently support PlaneFieldSpec only.")
            field_target = target.split(".", 1)[1]
            if field_target in {"fixed", self._field.fixed_axis}:
                return float(self._field.fixed)
            raise ValueError(f"Unsupported field target {target!r}")

        component_name, attr_path = target.split(".", 1)
        component = _find_component_by_name(self._obj, component_name)
        if component is None:
            raise ValueError(f"Could not resolve component {component_name!r}")
        return float(np.asarray(_resolve_attr(component, attr_path)))

    def _set_target_value(self, target: str, value: float) -> None:
        if target.startswith("field."):
            if self._field is None or not isinstance(self._field, PlaneFieldSpec):
                raise ValueError("Field controls require PlaneFieldSpec.")
            field_target = target.split(".", 1)[1]
            if field_target in {"fixed", self._field.fixed_axis}:
                self._field.fixed = float(value)
                return
            raise ValueError(f"Unsupported field target {target!r}")

        component_name, attr_path = target.split(".", 1)
        component = _find_component_by_name(self._obj, component_name)
        if component is None:
            raise ValueError(f"Could not resolve component {component_name!r}")
        _assign_attr(component, attr_path, value)

    def _resolve_backend(self, backend: str) -> str:
        if backend not in {"auto", "plotly", "k3d"}:
            raise ValueError(f"Unsupported visualization backend {backend!r}")
        if backend == "plotly":
            return "plotly"
        if backend == "k3d" and not _has_module("k3d"):
            warnings.warn("k3d is not installed; falling back to Plotly.", stacklevel=2)
            return "plotly"
        if backend == "auto":
            return "k3d" if _has_module("k3d") else "plotly"
        return backend

    def _render_current_scene(self) -> None:
        self._render_count += 1
        if self._backend == "k3d":
            try:
                self._plot_obj = _render_k3d_scene(self.scene)
            except Exception as exc:  # pragma: no cover - k3d not installed in CI
                warnings.warn(
                    f"k3d rendering failed ({exc}); falling back to Plotly.",
                    stacklevel=2,
                )
                self._backend = "plotly"
                self._plot_obj = _render_plotly_scene(self.scene)
        else:
            self._plot_obj = _render_plotly_scene(self.scene)

        if self._output is not None:
            if _in_ipython_kernel():
                try:
                    from IPython.display import display
                except ImportError:  # pragma: no cover - script fallback
                    return
                with self._output:
                    self._output.clear_output(wait=True)
                    display(self._plot_obj)

    def _repr_mimebundle_(self, include=None, exclude=None):
        target = self.widget
        if hasattr(target, "_repr_mimebundle_"):
            return target._repr_mimebundle_(include=include, exclude=exclude)
        if hasattr(self._plot_obj, "_repr_mimebundle_"):
            return self._plot_obj._repr_mimebundle_(include=include, exclude=exclude)
        return None

    def _open_html(self, html: str, *, prefix: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=prefix,
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(html)
            path = Path(handle.name)
        self._last_export_path = path
        webbrowser.open(path.as_uri())
        return path


def visualize_system(obj: object, **kwargs) -> SceneViewer:
    """Public convenience entrypoint for one-line visualization."""
    return SceneViewer(obj, **kwargs)


def _find_component_by_name(obj: object, name: str):
    if getattr(obj, "name", None) == name:
        return obj
    if isinstance(obj, MagneticSystem):
        for child in obj.components:
            found = _find_component_by_name(child, name)
            if found is not None:
                return found
    return None


def _resolve_attr(component: object, attr_path: str):
    parts = attr_path.split(".")
    value = getattr(component, parts[0])
    for part in parts[1:]:
        idx = _axis_or_index(part)
        value = np.asarray(value)[idx]
    return value


def _assign_attr(component: object, attr_path: str, value: float) -> None:
    parts = attr_path.split(".")
    attr = parts[0]
    existing = getattr(component, attr)
    if len(parts) == 1:
        setattr(component, attr, _coerce_value(existing, value))
        return

    array = np.asarray(existing, dtype=float).copy()
    array[_axis_or_index(parts[1])] = value
    setattr(component, attr, _coerce_value(existing, array))


def _coerce_value(existing: Any, value: Any):
    if isinstance(existing, jnp.ndarray):
        return jnp.asarray(value, dtype=existing.dtype)
    if isinstance(existing, np.ndarray):
        return np.asarray(value, dtype=existing.dtype)
    if isinstance(existing, int):
        return int(round(float(np.asarray(value))))
    if isinstance(existing, float):
        return float(np.asarray(value))
    return value


def _axis_or_index(token: str) -> int:
    if token in {"x", "y", "z"}:
        return {"x": 0, "y": 1, "z": 2}[token]
    return int(token)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - optional path
        return False
    shell = get_ipython()
    return bool(shell is not None and getattr(shell, "kernel", None) is not None)


def _render_plotly_scene(scene: Scene) -> go.Figure:
    traces: list[Any] = []
    for node, world in scene.iter_nodes():
        for primitive in node.primitives:
            style = merge_style(node.style, primitive.style)
            if not style.visible:
                continue
            traces.extend(_primitive_to_plotly_traces(primitive, world, style, label=node.label))

    fig = go.Figure(data=traces)
    subtitle = _scene_subtitle(scene)
    title = scene.metadata.get("title", "MagDiff Visualization")
    if subtitle:
        title = f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        uirevision="magdiff-scene",
    )
    bounds = scene.bounds()
    if bounds is not None:
        mins, maxs = bounds
        pad = np.maximum((maxs - mins) * 0.1, 1e-3)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[mins[0] - pad[0], maxs[0] + pad[0]]),
                yaxis=dict(range=[mins[1] - pad[1], maxs[1] + pad[1]]),
                zaxis=dict(range=[mins[2] - pad[2], maxs[2] + pad[2]]),
                aspectmode="data",
            )
        )
    return fig


def _primitive_to_plotly_traces(
    primitive: Primitive, world_transform: np.ndarray, style: VisualStyle, *, label: str | None
) -> list[Any]:
    kind = primitive.kind
    params = primitive.params
    if kind == "box":
        return [_plotly_mesh_trace(*_box_mesh(np.asarray(params["size"], dtype=float), world_transform), style, label)]
    if kind == "cylinder":
        return [
            _plotly_mesh_trace(
                *_cylinder_mesh(float(params["radius"]), float(params["height"]), world_transform),
                style,
                label,
            )
        ]
    if kind == "sphere":
        return [
            _plotly_mesh_trace(
                *_sphere_mesh(float(params["radius"]), world_transform),
                style,
                label,
            )
        ]
    if kind == "tube":
        points = np.asarray(params["points"], dtype=float)
        if params.get("closed", False) and len(points) > 0:
            points = np.vstack([points, points[0]])
        world_points = apply_transform(points, world_transform)
        return [_plotly_line_trace(world_points, style, label)]
    if kind == "polyline":
        world_points = apply_transform(np.asarray(params["points"], dtype=float), world_transform)
        return [_plotly_line_trace(world_points, style, label)]
    if kind == "points":
        world_points = apply_transform(np.asarray(params["points"], dtype=float), world_transform)
        return [_plotly_points_trace(world_points, style, label)]
    if kind == "axes":
        return _plotly_axes_traces(float(params.get("scale", 1.0)), world_transform)
    if kind == "arrow":
        origin = apply_transform(np.asarray(params["origin"], dtype=float), world_transform)[0]
        direction = _rotate_vector(np.asarray(params["direction"], dtype=float), world_transform)
        endpoint = origin + direction
        traces = [_plotly_line_trace(np.vstack([origin, endpoint]), style, label)]
        if params.get("show_head", True) and np.linalg.norm(direction) > 0.0:
            traces.append(
                go.Cone(
                    x=[endpoint[0]],
                    y=[endpoint[1]],
                    z=[endpoint[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    sizemode="absolute",
                    sizeref=max(np.linalg.norm(direction), 1e-3),
                    anchor="tip",
                    showscale=False,
                    colorscale=[[0.0, _plotly_color(style.color)], [1.0, _plotly_color(style.color)]],
                    name=label,
                )
            )
        return traces
    if kind == "text":
        position = apply_transform(np.asarray(params["position"], dtype=float), world_transform)[0]
        return [
            go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode="text",
                text=[params["text"]],
                textposition="top center",
                textfont=dict(color=_plotly_color(style.color), size=12),
                name=label,
            )
        ]
    if kind == "surface":
        kwargs: dict[str, Any] = {
            "x": np.asarray(params["x"], dtype=float),
            "y": np.asarray(params["y"], dtype=float),
            "z": np.asarray(params["z"], dtype=float),
            "surfacecolor": np.asarray(params["surfacecolor"], dtype=float),
            "colorscale": style.colorscale or "Viridis",
            "opacity": style.opacity,
            "showscale": style.showscale,
            "name": label,
        }
        if style.showscale:
            kwargs["colorbar"] = {
                "title": style.colorbar_title,
                "tickformat": style.colorbar_tickformat,
            }
        if style.color_range is not None:
            kwargs["cmin"] = float(style.color_range[0])
            kwargs["cmax"] = float(style.color_range[1])
        return [go.Surface(**kwargs)]
    return []


def _plotly_mesh_trace(vertices, faces, style: VisualStyle, label: str | None):
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        opacity=style.opacity,
        color=_plotly_color(style.color),
        name=label,
    )


def _plotly_line_trace(points: np.ndarray, style: VisualStyle, label: str | None):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line=dict(color=_plotly_color(style.color), width=style.line_width),
        name=label,
    )


def _plotly_points_trace(points: np.ndarray, style: VisualStyle, label: str | None):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(color=_plotly_color(style.color), size=max(style.point_size * 100.0, 2.0)),
        name=label,
    )


def _plotly_axes_traces(scale: float, transform: np.ndarray):
    origins = np.zeros((4, 3), dtype=float)
    origins[1, 0] = scale
    origins[2, 1] = scale
    origins[3, 2] = scale
    world = apply_transform(origins, transform)
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    traces = []
    for idx, color in enumerate(colors):
        traces.append(
            go.Scatter3d(
                x=[world[0, 0], world[idx + 1, 0]],
                y=[world[0, 1], world[idx + 1, 1]],
                z=[world[0, 2], world[idx + 1, 2]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
            )
        )
    return traces


def _plotly_color(color: str | tuple[float, float, float] | None) -> str:
    if color is None:
        return "#4c78a8"
    if isinstance(color, tuple):
        rgb = tuple(int(max(0.0, min(1.0, channel)) * 255.0) for channel in color)
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    return color


def _box_mesh(size: np.ndarray, world_transform: np.ndarray):
    hx, hy, hz = 0.5 * np.asarray(size, dtype=float)
    vertices = np.asarray(
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
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )
    return apply_transform(vertices, world_transform), faces


def _cylinder_mesh(radius: float, height: float, world_transform: np.ndarray, segments: int = 24):
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    bottom = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), -0.5 * height * np.ones_like(theta)],
        axis=1,
    )
    top = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), 0.5 * height * np.ones_like(theta)],
        axis=1,
    )
    vertices = np.vstack([bottom, top, [[0.0, 0.0, -0.5 * height]], [[0.0, 0.0, 0.5 * height]]])
    bottom_center = 2 * segments
    top_center = 2 * segments + 1
    faces = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt])
        faces.append([idx, segments + nxt, segments + idx])
        faces.append([bottom_center, nxt, idx])
        faces.append([top_center, segments + idx, segments + nxt])
    return apply_transform(vertices, world_transform), np.asarray(faces, dtype=int)


def _sphere_mesh(radius: float, world_transform: np.ndarray, lat_steps: int = 12, lon_steps: int = 24):
    phi = np.linspace(0.0, np.pi, lat_steps)
    theta = np.linspace(0.0, 2.0 * np.pi, lon_steps, endpoint=False)
    vertices = []
    for p in phi:
        for t in theta:
            vertices.append(
                [
                    radius * np.sin(p) * np.cos(t),
                    radius * np.sin(p) * np.sin(t),
                    radius * np.cos(p),
                ]
            )
    vertices = np.asarray(vertices, dtype=float)
    faces = []
    for i in range(lat_steps - 1):
        for j in range(lon_steps):
            nxt = (j + 1) % lon_steps
            a = i * lon_steps + j
            b = i * lon_steps + nxt
            c = (i + 1) * lon_steps + nxt
            d = (i + 1) * lon_steps + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return apply_transform(vertices, world_transform), np.asarray(faces, dtype=int)


def _rotate_vector(vector: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=float)[:3, :3] @ np.asarray(vector, dtype=float)


def _render_k3d_scene(scene: Scene):  # pragma: no cover - exercised only when k3d is installed
    import k3d

    show_axes = bool(scene.metadata.get("show_axes", True))
    plot = k3d.plot(grid_visible=show_axes, camera_up_axis="z")
    plot.axes = ["X (m)", "Y (m)", "Z (m)"]
    plot.additional_js_code = _k3d_legend_script(scene)
    bounds = scene.bounds()
    if bounds is not None:
        mins, maxs = bounds
        plot.grid = [
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        ]
        plot.axes_helper = max(float(np.max(maxs - mins)) * 0.15, 0.5) if show_axes else 0.0
    else:
        plot.axes_helper = 1.0 if show_axes else 0.0
    for node, world in scene.iter_nodes():
        for primitive in node.primitives:
            style = merge_style(node.style, primitive.style)
            if not style.visible:
                continue
            for obj in _primitive_to_k3d_objects(primitive, world, style):
                plot += obj
                if primitive.kind == "surface" and style.showscale:
                    plot.colorbar_object_id = obj.id
                    plot.colorbar_scientific = True
    overlay_note = _scene_overlay_note(scene)
    if overlay_note:
        plot += k3d.text2d(
            overlay_note,
            position=(0.02, 0.98),
            color=0x111111,
            size=0.8,
            reference_point="lt",
            label_box=True,
            is_html=True,
        )
    return plot


def _primitive_to_k3d_objects(
    primitive: Primitive, world_transform: np.ndarray, style: VisualStyle
):  # pragma: no cover - exercised only when k3d is installed
    import k3d

    color = _k3d_color(style.color)
    kind = primitive.kind
    params = primitive.params
    if kind == "box":
        vertices, faces = _box_mesh(np.asarray(params["size"], dtype=float), world_transform)
        return [k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=color, opacity=style.opacity)]
    if kind == "cylinder":
        vertices, faces = _cylinder_mesh(float(params["radius"]), float(params["height"]), world_transform)
        return [k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=color, opacity=style.opacity)]
    if kind == "sphere":
        vertices, faces = _sphere_mesh(float(params["radius"]), world_transform)
        return [k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=color, opacity=style.opacity)]
    if kind in {"tube", "polyline"}:
        points = np.asarray(params["points"], dtype=float)
        if params.get("closed", False) and len(points) > 0:
            points = np.vstack([points, points[0]])
        points = apply_transform(points, world_transform)
        return [k3d.line(points.astype(np.float32), color=color, width=float(style.line_width))]
    if kind == "points":
        points = apply_transform(np.asarray(params["points"], dtype=float), world_transform)
        return [k3d.points(points.astype(np.float32), color=color, point_size=float(style.point_size))]
    if kind == "arrow":
        origin = apply_transform(np.asarray(params["origin"], dtype=float), world_transform)
        direction = _rotate_vector(np.asarray(params["direction"], dtype=float), world_transform)
        return [k3d.vectors(origin.astype(np.float32), direction[None, :].astype(np.float32), color=color)]
    if kind == "text":
        position = apply_transform(np.asarray(params["position"], dtype=float), world_transform)[0]
        return [k3d.label(params["text"], position=tuple(position.tolist()), color=color, size=0.8)]
    if kind == "surface":
        x = np.asarray(params["x"], dtype=float)
        y = np.asarray(params["y"], dtype=float)
        z = np.asarray(params["z"], dtype=float)
        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        vertex_valid = np.all(np.isfinite(vertices), axis=1)
        if not np.any(vertex_valid):
            return []
        remap = -np.ones(len(vertices), dtype=int)
        remap[vertex_valid] = np.arange(np.count_nonzero(vertex_valid))
        filtered_vertices = vertices[vertex_valid]
        faces = []
        rows, cols = x.shape
        for row in range(rows - 1):
            for col in range(cols - 1):
                idx = row * cols + col
                a, b, c, d = idx, idx + 1, idx + cols + 1, idx + cols
                if not (vertex_valid[a] and vertex_valid[b] and vertex_valid[c] and vertex_valid[d]):
                    continue
                faces.append([remap[a], remap[b], remap[c]])
                faces.append([remap[a], remap[c], remap[d]])
        if not faces:
            return []
        kwargs = {"color": color, "opacity": style.opacity}
        surfacecolor = params.get("surfacecolor")
        if surfacecolor is not None:
            values = np.asarray(surfacecolor, dtype=np.float32).ravel()[vertex_valid]
            kwargs["attribute"] = values
            kwargs["color_map"] = _k3d_colormap(style.colorscale)
            if style.color_range is not None:
                kwargs["color_range"] = [float(style.color_range[0]), float(style.color_range[1])]
            else:
                kwargs["color_range"] = [float(np.nanmin(values)), float(np.nanmax(values))]
        return [k3d.mesh(filtered_vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32), **kwargs)]
    return []


def _k3d_color(color: str | tuple[float, float, float] | None) -> int:
    color_str = _plotly_color(color)
    if color_str.startswith("#"):
        return int(color_str[1:], 16)
    if color_str.startswith("rgb"):
        values = color_str[color_str.find("(") + 1 : color_str.find(")")].split(",")
        r, g, b = [int(v.strip()) for v in values]
        return (r << 16) + (g << 8) + b
    return 0x4C78A8


def _k3d_colormap(name: str | None):
    import k3d

    if not name:
        return k3d.matplotlib_color_maps.Viridis
    target = str(name).lower()
    for attr in dir(k3d.matplotlib_color_maps):
        if attr.lower() == target:
            return getattr(k3d.matplotlib_color_maps, attr)
    return k3d.matplotlib_color_maps.Viridis


def _scene_subtitle(scene: Scene) -> str | None:
    plane_nodes = [node for node, _ in scene.iter_nodes() if node.kind == "field_plane"]
    if not plane_nodes:
        return None
    node = plane_nodes[0]
    spec = node.metadata.get("spec")
    quantity_label = node.metadata.get("quantity_label", "B")
    parts = [node.metadata.get("plane_label", "Field slice"), f"surface color: {quantity_label} (T)"]
    if getattr(spec, "show_vectors", False):
        parts.append("arrows: B direction, display length scaled for readability")
    return " | ".join(parts)


def _scene_overlay_note(scene: Scene) -> str | None:
    lines: list[str] = []
    plane_nodes = [node for node, _ in scene.iter_nodes() if node.kind == "field_plane"]
    if plane_nodes:
        node = plane_nodes[0]
        spec = node.metadata.get("spec")
        quantity_label = node.metadata.get("quantity_label", "|B|")
        lines.append(f"<b>{node.metadata.get('plane_label', 'Field slice')}</b>")
        lines.append(f"Surface color: {quantity_label} (T)")
        if getattr(spec, 'show_vectors', False):
            lines.append("Arrows: B direction; length is scaled for visualization")
    if any(node.kind == "dipole" for node, _ in scene.iter_nodes()):
        lines.append("Dipole arrow: magnetic moment direction")
    if not lines:
        return None
    return "<br/>".join(lines)


def _k3d_legend_script(scene: Scene) -> str:
    title = None
    for node, _ in scene.iter_nodes():
        for primitive in node.primitives:
            style = merge_style(node.style, primitive.style)
            if primitive.kind == "surface" and style.showscale:
                title = style.colorbar_title
                break
        if title:
            break
    if not title:
        return ""

    title_js = json.dumps(title)
    return f"""
(function() {{
  var plot = typeof K3DInstance !== 'undefined'
    ? K3DInstance
    : ((typeof this !== 'undefined' && this && this.K3DInstance) ? this.K3DInstance : null);
  if (!plot) return;
  var legendTitle = {title_js};

  function applyLegendPatch(attempt) {{
    if (!plot) return;
    try {{
      if (plot.setColorbarScientific) {{
        plot.setColorbarScientific(true);
      }}
      if (plot.parameters && plot.parameters.colorbarObjectId > 0 && plot.setColorMapLegend) {{
        plot.setColorMapLegend(plot.parameters.colorbarObjectId);
      }}

      var svg = plot.colorMapNode;
      if (!svg) {{
        if (attempt < 12) {{
          setTimeout(function() {{ applyLegendPatch(attempt + 1); }}, 50);
        }}
        return;
      }}

      var groups = svg.querySelectorAll('g[pos_y]');
      groups.forEach(function(group) {{
        var texts = group.querySelectorAll('text');
        if (!texts.length) return;
        var raw = texts[texts.length - 1].textContent;
        var value = Number(raw);
        if (!Number.isFinite(value)) return;
        var formatted = value.toExponential(2);
        texts.forEach(function(textNode) {{
          textNode.textContent = formatted;
        }});
      }});

      var existing = svg.querySelector('.magdiff-colorbar-title');
      if (!existing) {{
        existing = document.createElementNS(svg.namespaceURI, 'text');
        existing.setAttribute('class', 'magdiff-colorbar-title');
        existing.setAttribute('x', '18');
        existing.setAttribute('y', '8');
        existing.setAttribute('alignment-baseline', 'middle');
        existing.setAttribute('text-anchor', 'start');
        existing.setAttribute('font-size', '5');
        existing.setAttribute('fill', 'rgb(68, 68, 68)');
        svg.appendChild(existing);
      }}
      existing.textContent = legendTitle;
    }} catch (error) {{
      console.log(error);
    }}
  }}

  setTimeout(function() {{ applyLegendPatch(0); }}, 0);
}})();
""".strip()
