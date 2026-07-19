import numpy as np
import plotly.graph_objects as go

from .central_bodies import EARTH, CentralBody, add_central_bodies

_TRAJECTORY_COLORS = ["blue", "gold", "deepskyblue", "orchid", "lightseagreen", "coral"]


def _plot_trajectory(
    r: np.ndarray | list[np.ndarray],
    title: str,
    body: CentralBody | list[CentralBody] = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
    max_frames: int = 300,
):
    """`r` is either one [N, >=3] state/position array, or a list of them to plot several
    propagated spacecraft at once (e.g. `[sim1.X, sim2.X]`) -- only columns 0:3 (position) are
    used. All trajectories must share the same number of rows/timesteps (as they would if
    propagated with the same `dt`/`n_steps`), since the slider walks them together.

    `width`/`height` in px; pass None for either to let Plotly fill its container instead of
    using a fixed size. `slider` adds a time slider (+ play/pause) that scrubs a marker along each
    trajectory; `downsample_rate` controls how many rows get drawn (defaults to roughly 200
    evenly-spaced points -- plotting every raw row of a long propagation is what makes this slow).
    `max_frames` independently caps the number of slider/animation frames, since a small
    `downsample_rate` on a long trajectory would otherwise generate one frame per plotted point.
    `body` can be a single CentralBody or a list (e.g. `[EARTH, MOON]`) to plot several at once --
    each body's `center` sets where it's drawn, so position them in the same frame as `r`."""
    trajectories = r if isinstance(r, list) else [r]
    n = len(trajectories[0])
    if any(len(traj) != n for traj in trajectories):
        raise ValueError("all trajectories must have the same number of rows")

    if downsample_rate is None:
        downsample_rate = max(1, n // 200)

    multi = len(trajectories) > 1
    fig = go.Figure()
    current_position_traces = []

    for idx, traj in enumerate(trajectories):
        suffix = f" {idx}" if multi else ""
        traj_plot = traj[::downsample_rate]

        fig.add_scatter3d(
            x=traj_plot[:, 0],
            y=traj_plot[:, 1],
            z=traj_plot[:, 2],
            mode="markers",
            marker=dict(
                size=1, color=_TRAJECTORY_COLORS[idx % len(_TRAJECTORY_COLORS)]
            ),
            name=f"Trajectory{suffix}",
        )

        # Final Position
        fig.add_scatter3d(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            z=[traj[-1, 2]],
            mode="markers",
            marker=dict(size=10, color="green"),
            name=f"Final Position{suffix}",
        )

        # Initial Position
        fig.add_scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode="markers",
            marker=dict(
                size=10,
                color="red",
            ),
            name=f"Initial Position{suffix}",
        )

        if slider:
            fig.add_scatter3d(
                x=[traj[0, 0]],
                y=[traj[0, 1]],
                z=[traj[0, 2]],
                mode="markers",
                marker=dict(size=6, color="orange", symbol="diamond"),
                name=f"Current Position{suffix}",
            )
            current_position_traces.append(len(fig.data) - 1)

    bodies = body if isinstance(body, list) else [body]
    add_central_bodies(fig, bodies, method=method)

    if slider:
        if t is None:
            t = np.arange(n)

        step_indices = np.arange(0, n, downsample_rate)
        if len(step_indices) > max_frames:
            step_indices = step_indices[:: max(1, len(step_indices) // max_frames)]
        if step_indices[-1] != n - 1:
            step_indices = np.append(step_indices, n - 1)

        fig.frames = [
            go.Frame(
                data=[
                    go.Scatter3d(x=[traj[i, 0]], y=[traj[i, 1]], z=[traj[i, 2]])
                    for traj in trajectories
                ],
                traces=current_position_traces,
                name=str(i),
            )
            for i in step_indices
        ]

        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "y": 0,
                    "xanchor": "left",
                    "x": 0.1,
                    "len": 0.9,
                    "currentvalue": {
                        "prefix": "Time: ",
                        "suffix": " s",
                        "visible": True,
                    },
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "method": "animate",
                            "label": f"{t[i]:.0f}",
                        }
                        for i in step_indices
                    ],
                }
            ],
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {"frame": {"duration": 0}, "mode": "immediate"},
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

    fig.update_layout(
        width=width,
        height=height,
        autosize=width is None or height is None,
        title_font=dict(size=24, family="Garamond"),
        title_text=title,
        title_x=0.5,
        title_y=0.9,
        scene=dict(aspectmode="data"),
    )
    fig.show()


def plot_orbit(
    r: np.ndarray | list[np.ndarray],
    body: CentralBody | list[CentralBody] = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
    max_frames: int = 300,
):
    _plot_trajectory(
        r,
        "Satellite Trajectory",
        body=body,
        method=method,
        width=width,
        height=height,
        t=t,
        slider=slider,
        downsample_rate=downsample_rate,
        max_frames=max_frames,
    )


def plot_rocket(
    r: np.ndarray | list[np.ndarray],
    body: CentralBody | list[CentralBody] = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
    max_frames: int = 300,
):
    _plot_trajectory(
        r,
        "Rocket Trajectory",
        body=body,
        method=method,
        width=width,
        height=height,
        t=t,
        slider=slider,
        downsample_rate=downsample_rate,
        max_frames=max_frames,
    )
