import unittest

import jax.numpy as jnp
import numpy as np

from magdiff.components.coil import Coil
from magdiff.components.cuboid import Cuboid
from magdiff.components.cylinder import Cylinder
from magdiff.components.dipole import Dipole
from magdiff.system import MagneticSystem
from magdiff.visualization import PlaneFieldSpec, SceneViewer, build_scene, visualize_system
from magdiff.visualization.overlays import build_plane_overlay, sample_plane_field
from magdiff.visualize import visualize_field


def build_sample_system():
    coil = Coil(
        name="C03",
        current=1.0,
        radius=0.02,
        position=jnp.array([0.2, 0.0, 0.0]),
    )
    nested = MagneticSystem(
        components=[
            Dipole(
                name="nested_dipole",
                moment=jnp.array([0.0, 0.0, 1.0]),
                position=jnp.array([0.5, 0.0, 0.0]),
            )
        ],
        position=jnp.array([0.0, 2.0, 0.0]),
        name="nested",
    )
    root = MagneticSystem(
        components=[
            Cuboid(
                name="cube",
                magnetization=jnp.array([1.0, 0.0, 0.0]),
                dimension=jnp.array([1.0, 2.0, 3.0]),
                position=jnp.array([1.0, 0.0, 0.0]),
            ),
            Cylinder(
                name="cyl",
                magnetization=jnp.array([0.0, 1.0, 0.0]),
                diameter=0.5,
                height=1.5,
                position=jnp.array([0.0, -1.0, 0.0]),
            ),
            coil,
            nested,
        ],
        position=jnp.array([1.0, 0.0, 0.0]),
        name="root_system",
    )
    return root


class VisualizationTests(unittest.TestCase):
    def test_build_scene_emits_expected_hierarchy_and_primitives(self):
        system = build_sample_system()
        scene = build_scene(system)

        nodes = {node.key: (node, world) for node, world in scene.iter_nodes()}

        self.assertEqual(scene.root.kind, "scene")
        self.assertEqual(scene.root.children[0].kind, "system")
        self.assertEqual(
            {child.kind for child in scene.root.children[0].children},
            {"cuboid", "cylinder", "coil", "system"},
        )
        self.assertEqual(
            [primitive.kind for primitive in scene.root.children[0].children[0].primitives],
            ["box"],
        )
        self.assertEqual(
            [primitive.kind for primitive in scene.root.children[0].children[1].primitives],
            ["cylinder"],
        )
        self.assertEqual(
            [primitive.kind for primitive in scene.root.children[0].children[2].primitives],
            ["tube", "arrow"],
        )

        nested_node, nested_world = nodes["root/3"]
        self.assertEqual(nested_node.kind, "system")
        np.testing.assert_allclose(nested_world[:3, 3], np.array([1.0, 2.0, 0.0]), atol=1e-7)

        nested_child_node, nested_child_world = nodes["root/3/0"]
        self.assertEqual(nested_child_node.kind, "dipole")
        self.assertEqual(
            [primitive.kind for primitive in nested_child_node.primitives],
            ["sphere", "arrow", "text"],
        )
        np.testing.assert_allclose(
            nested_child_world[:3, 3],
            np.array([1.5, 2.0, 0.0]),
            atol=1e-7,
        )

    def test_plane_field_sampling_matches_direct_field_query(self):
        system = MagneticSystem(
            [Dipole(name="main", moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.zeros(3))],
            name="single",
        )
        spec = PlaneFieldSpec.xz(
            y=0.0,
            xlim=(-0.2, 0.2),
            zlim=(-0.3, 0.3),
            resolution=(7, 9),
            quantity="norm",
            show_vectors=True,
        )

        sampled = sample_plane_field(system, spec)
        center = (sampled["coords"].shape[0] // 2, sampled["coords"].shape[1] // 2)
        point = sampled["coords"][center]
        expected = np.asarray(system.field_at(jnp.asarray(point)))
        actual = sampled["B"][center]

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)

        overlay = build_plane_overlay(system, spec, key="overlay")
        self.assertEqual(overlay.kind, "field_plane")
        self.assertEqual(overlay.primitives[0].kind, "surface")
        self.assertEqual(overlay.primitives[0].style.colorbar_title, "|B| (T)")
        self.assertEqual(overlay.primitives[0].style.colorbar_tickformat, ".2e")
        self.assertTrue(any(primitive.kind == "arrow" for primitive in overlay.primitives[1:]))
        custom = PlaneFieldSpec.xz(y=0.0, vector_scale=0.22, vector_stride=4)
        self.assertEqual(custom.vector_scale, 0.22)
        self.assertEqual(custom.vector_stride, 4)

    def test_plane_field_exclusion_radius_masks_dipole_core_and_clips_colors(self):
        system = MagneticSystem(
            [Dipole(name="main", moment=jnp.array([50.0, 0.0, 0.0]), position=jnp.zeros(3))],
            name="single",
        )
        spec = PlaneFieldSpec.xz(
            y=0.0,
            xlim=(-0.1, 0.1),
            zlim=(-0.1, 0.1),
            resolution=(41, 41),
            quantity="norm",
            show_vectors=True,
            exclusion_radius=0.03,
            color_percentile=95.0,
        )

        sampled = sample_plane_field(system, spec)
        self.assertTrue(np.any(~sampled["valid_mask"]))

        overlay = build_plane_overlay(system, spec, key="overlay")
        surface = overlay.primitives[0]
        self.assertIsNotNone(surface.style.color_range)
        self.assertLess(surface.style.color_range[1], 1e6)
        finite_surface = np.asarray(surface.params["surfacecolor"], dtype=float)
        finite_surface = finite_surface[np.isfinite(finite_surface)]
        self.assertLessEqual(finite_surface.max(), surface.style.color_range[1] + 1e-12)

    def test_field_overlays_stay_in_world_frame(self):
        system = MagneticSystem(
            [Dipole(name="main", moment=jnp.array([0.0, 0.0, 1.0]), position=jnp.zeros(3))],
            position=jnp.array([1.0, 2.0, 3.0]),
            name="offset_system",
        )
        scene = build_scene(system, field=PlaneFieldSpec.xz(y=0.0, xlim=(-0.1, 0.1), zlim=(-0.2, 0.2)))

        overlay_node = next(child for child in scene.root.children if child.kind == "field_plane")
        world = next(world for node, world in scene.iter_nodes() if node.key == overlay_node.key)
        first_surface = overlay_node.primitives[0]
        points = np.stack(
            [
                np.asarray(first_surface.params["x"]).ravel(),
                np.asarray(first_surface.params["y"]).ravel(),
                np.asarray(first_surface.params["z"]).ravel(),
            ],
            axis=1,
        )
        world_points = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=float)],
            axis=1,
        )
        world_points = (world @ world_points.T).T[:, :3]
        self.assertTrue(np.allclose(world_points[:, 1], 0.0))

    def test_visualize_system_and_component_method_return_viewer(self):
        system = build_sample_system()

        viewer = visualize_system(system)
        viewer_from_method = system.visualize()

        self.assertIsInstance(viewer, SceneViewer)
        self.assertIsInstance(viewer_from_method, SceneViewer)
        self.assertEqual(viewer.scene.root.kind, "scene")
        self.assertIsNotNone(viewer.widget)

    def test_controls_rebuild_once_and_update_targets(self):
        system = build_sample_system()
        spec = PlaneFieldSpec.xz(y=0.0)
        viewer = visualize_system(
            system,
            field=spec,
            controls={
                "coil_current": {"target": "C03.current", "min": -5.0, "max": 5.0},
                "plane_y": {"target": "field.y", "min": -0.05, "max": 0.05},
            },
        )

        initial_count = viewer._render_count
        viewer._control_widgets["coil_current"].value = 2.5
        self.assertEqual(viewer._render_count, initial_count + 1)
        self.assertEqual(float(np.asarray(system.components[2].current)), 2.5)

        next_count = viewer._render_count
        viewer._control_widgets["plane_y"].value = 0.02
        self.assertEqual(viewer._render_count, next_count + 1)
        self.assertTrue(np.isclose(viewer._field.fixed, 0.02))

    def test_legacy_visualize_field_returns_scene_viewer(self):
        system = build_sample_system()
        viewer = visualize_field(
            system,
            region=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
            grid_count=(4, 4, 4),
        )

        self.assertIsInstance(viewer, SceneViewer)
        overlay_kinds = [
            child.kind for child in viewer.scene.root.children if child.kind.startswith("field_")
        ]
        self.assertIn("field_glyphs", overlay_kinds)


if __name__ == "__main__":
    unittest.main()
