"""Utility functions for magdiff."""

from typing import List

import jax.numpy as jnp
from magdiff.components.base import MagneticComponent
from magdiff.system import MagneticSystem


def _fmt_array(arr: jnp.ndarray) -> str:
    """Format a JAX array as a compact string like [1.0, 0.0, 0.0]."""
    return "[" + ", ".join(f"{float(x):.4g}" for x in arr) + "]"


def _component_label(comp: MagneticComponent) -> str:
    """Build a one-line label for a component with its key properties."""

    type_name = type(comp).__name__

    # Use name if set, otherwise just the class name
    if comp.name is not None:
        label = f"{comp.name} ({type_name})"
    else:
        label = type_name

    parts = [label]
    parts.append(f"pos={_fmt_array(comp.position)}")

    # Only show rotation if non-zero
    if jnp.any(comp.rotation_vector != 0):
        parts.append(f"rot={_fmt_array(comp.rotation_vector)}")

    # Component-specific properties
    if hasattr(comp, "moment"):
        parts.append(f"moment={_fmt_array(comp.moment)}")
    if hasattr(comp, "magnetization"):
        parts.append(f"M={_fmt_array(comp.magnetization)}")
    if hasattr(comp, "dimension"):
        parts.append(f"dim={_fmt_array(comp.dimension)}")
    if hasattr(comp, "diameter"):
        parts.append(f"d={comp.diameter:.4g}")
    if hasattr(comp, "height"):
        parts.append(f"h={comp.height:.4g}")
    if isinstance(comp, MagneticSystem):
        parts.append(f"({len(comp.components)} children)")

    return "  ".join(parts)


def print_tree(system: MagneticSystem, name: str = None):
    """Print a tree-formatted view of the system hierarchy.

    Example output:
        system (MagneticSystem)  pos=[0, 0, 0]  (2 children)
        ├── dip_left (Dipole)  pos=[0.5, 0, 0]  moment=[0, 0, 1]
        └── dip_right (Dipole)  pos=[-0.5, 0, 0]  moment=[0, 0, 1]

    :param system: a MagneticSystem to display.
    :param name: override label for the root node (defaults to system's own label).
    """
    if name is not None:
        root_label = name
    else:
        root_label = _component_label(system)

    lines = [root_label]
    _build_tree_lines(system, lines, prefix="")
    print("\n".join(lines))


def _build_tree_lines(system: MagneticSystem, lines: List[str], prefix: str):
    """Recursively build tree lines for a system's children."""

    children = system.components
    for i, comp in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + _component_label(comp))

        # If this child is itself a system, recurse into it
        if isinstance(comp, MagneticSystem) and comp.components:
            child_prefix = prefix + ("    " if is_last else "│   ")
            _build_tree_lines(comp, lines, child_prefix)
