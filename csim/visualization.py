import numpy as np
import plotly.graph_objects as go

from .central_bodies import EARTH, CentralBody, central_body_trace


def _plot_trajectory(
    r: np.ndarray,
    title: str,
    body: CentralBody = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
):
    """`width`/`height` in px; pass None for either to let Plotly fill its container instead of
    using a fixed size. `slider` adds a time slider (+ play/pause) that scrubs a marker along the
    trajectory; `downsample_rate` controls how many of `r`'s rows become slider steps/frames
    (defaults to roughly 200 evenly-spaced steps)."""
    fig = go.Figure()

    # Plot trajectory
    fig.add_scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
        name="Trajectory",
    )

    # Final Position
    fig.add_scatter3d(
        x=[r[-1, 0]],
        y=[r[-1, 1]],
        z=[r[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Final Position",
    )

    # Initial Position
    fig.add_scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode="markers",
        marker=dict(
            size=10,
            color="red",
        ),
        name="Initial Position",
    )

    if slider:
        fig.add_scatter3d(
            x=[r[0, 0]],
            y=[r[0, 1]],
            z=[r[0, 2]],
            mode="markers",
            marker=dict(size=6, color="orange", symbol="diamond"),
            name="Current Position",
        )
        current_position_trace = len(fig.data) - 1

    fig.add_trace(central_body_trace(body, method=method))

    if slider:
        if t is None:
            t = np.arange(len(r))
        if downsample_rate is None:
            downsample_rate = max(1, len(r) // 200)

        step_indices = np.arange(0, len(r), downsample_rate)
        if step_indices[-1] != len(r) - 1:
            step_indices = np.append(step_indices, len(r) - 1)

        fig.frames = [
            go.Frame(
                data=[go.Scatter3d(x=[r[i, 0]], y=[r[i, 1]], z=[r[i, 2]])],
                traces=[current_position_trace],
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
    r: np.ndarray,
    body: CentralBody = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
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
    )


def plot_rocket(
    r: np.ndarray,
    body: CentralBody = EARTH,
    method: str = "surface",
    width: int | None = 1600,
    height: int | None = 900,
    t: np.ndarray = None,
    slider: bool = True,
    downsample_rate: int | None = None,
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
    )
