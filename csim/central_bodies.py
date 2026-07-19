"""Central body (Earth, Moon, ...) sphere rendering for Plotly 3D scenes.

Two independent techniques for turning a `CentralBody` into a Plotly trace are
provided, since they trade off differently:

- `sphere_surface_trace`: a UV-parametrized `go.Surface`. Cheap to build, good
  default. Triangle density bunches up at the poles (UV pinching), which shows
  at low resolution.
- `sphere_mesh_trace`: a `go.Mesh3d` built from a subdivided icosahedron
  ("icosphere"). Near-uniform triangle size everywhere, so shading looks even
  at low triangle counts, at the cost of a pricier Python-side build.

`add_central_bodies` adds one or more bodies (e.g. Earth + Moon) to a figure
in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from .world import R_EARTH

DEFAULT_LIGHTING = dict(ambient=0.45, diffuse=0.6, specular=0.15, roughness=0.9)
DEFAULT_LIGHTPOSITION = dict(x=1e5, y=1e5, z=1e5)
"""Plotly caps lightposition to [-1e5, 1e5]; it's independent of data-coordinate scale."""


@dataclass
class CentralBody:
    name: str
    radius: float
    """meters"""
    color: str = "#888888"
    shadow_color: str | None = None
    """Darker shade for the surface colorscale; derived from `color` if omitted."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Position of the body center in the plot frame, meters."""

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        if self.shadow_color is None:
            self.shadow_color = _darken(self.color, 0.5)


def _darken(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


EARTH = CentralBody(name="Earth", radius=R_EARTH, color="#3a6ea5")
MOON = CentralBody(name="Moon", radius=1737.4e3, color="#8a8a8a")


def sphere_surface_trace(
    body: CentralBody, resolution: int = 40, opacity: float = 1.0
) -> go.Surface:
    """UV-parametrized sphere as a go.Surface."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = body.radius * np.outer(np.cos(u), np.sin(v)) + body.center[0]
    y = body.radius * np.outer(np.sin(u), np.sin(v)) + body.center[1]
    z = body.radius * np.outer(np.ones_like(u), np.cos(v)) + body.center[2]

    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[[0, body.shadow_color], [1, body.color]],
        showscale=False,
        opacity=opacity,
        name=body.name,
        hoverinfo="skip",
        lighting=DEFAULT_LIGHTING,
        lightposition=DEFAULT_LIGHTPOSITION,
    )


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-radius (vertices, faces) via subdivided icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base_verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ]
    )
    base_verts = base_verts / np.linalg.norm(base_verts, axis=1, keepdims=True)

    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    verts = list(base_verts)
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (verts[a] + verts[b]) / 2
        mid /= np.linalg.norm(mid)
        idx = len(verts)
        verts.append(mid)
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        next_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            next_faces += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        faces = next_faces

    return np.array(verts), np.array(faces)


def sphere_mesh_trace(
    body: CentralBody, subdivisions: int = 3, opacity: float = 1.0
) -> go.Mesh3d:
    """Subdivided-icosahedron sphere as a go.Mesh3d."""
    verts, faces = _icosphere(subdivisions)
    x = verts[:, 0] * body.radius + body.center[0]
    y = verts[:, 1] * body.radius + body.center[1]
    z = verts[:, 2] * body.radius + body.center[2]

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=body.color,
        opacity=opacity,
        name=body.name,
        hoverinfo="skip",
        flatshading=False,
        lighting=DEFAULT_LIGHTING,
        lightposition=DEFAULT_LIGHTPOSITION,
    )


def central_body_trace(
    body: CentralBody, method: str = "surface", **kwargs
) -> go.Surface | go.Mesh3d:
    if method == "surface":
        return sphere_surface_trace(body, **kwargs)
    if method == "mesh":
        return sphere_mesh_trace(body, **kwargs)
    raise ValueError(f"Unknown method {method!r}; choose 'surface' or 'mesh'")


def add_central_bodies(
    fig: go.Figure, bodies: list[CentralBody], method: str = "surface", **kwargs
) -> list[go.Surface | go.Mesh3d]:
    """Add one or more central bodies (e.g. Earth + Moon) to `fig`. Returns the added traces."""
    traces = [central_body_trace(body, method=method, **kwargs) for body in bodies]
    for trace in traces:
        fig.add_trace(trace)
    return traces
