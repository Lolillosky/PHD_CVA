import plotly.express as px
import plotly.graph_objects as go


import plotly.graph_objects as go

def plot_3d_scatter(
    dataframe,
    x_label="x",
    y_label="y",
    z_label="z",
    fig=None,
    color="blue"
):
    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=800,
            height=800
        )

    fig.add_trace(
        go.Scatter3d(
            x=dataframe[x_label],
            y=dataframe[y_label],
            z=dataframe[z_label],
            mode="markers",
            marker=dict(
                color=color,
                size=1
            )
        )
    )

    return fig


import numpy as np


def plot_surface_with_contours(func, x_min, x_max, y_min, y_max, n_points, color,fig = None):
    """
    Plots a 3D surface of z = func(x,y) with two sets of contours.

    Parameters
    ----------
    func : callable
        Function of two variables, f(x,y).
    x_min, x_max : float
        Range for x axis.
    y_min, y_max : float
        Range for y axis.
    n_points : int
        Resolution of the grid.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure with the surface and contours.
    """
    # Grid
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y).reshape((n_points,n_points))

    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=800,
            height=800
        )


    # Add lines along x direction
    for i in range(len(y)):
        fig.add_trace(go.Scatter3d(
        x=X[i, :],
        y=Y[i, :],
        z=Z[i, :],
        mode="lines",
        line=dict(color=color, width=1.0),
        showlegend=False
    ))
        
        # Add lines along y direction
    for j in range(len(x)):
        fig.add_trace(go.Scatter3d(
        x=X[:, j],
        y=Y[:, j],
        z=Z[:, j],
        mode="lines",
        line=dict(color=color, width=1.0),
        showlegend=False
    ))


    

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

