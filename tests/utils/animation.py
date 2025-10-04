"""Animation utilities for iLQR test trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_trajectory(states, draw_fn, filename, fps=10, **draw_kwargs):
    """
    Create and save an animated .gif of a state trajectory.

    Args:
        states: Array of shape (T+1, n) containing state trajectory
        draw_fn: Callable(ax, state, **kwargs) that draws the system at given state
        filename: Output .gif filename
        fps: Frames per second for animation
        **draw_kwargs: Additional keyword arguments passed to draw_fn

    Example:
        def draw_bicycle(ax, state, goal=(0, 0)):
            x, y, theta = state
            # Draw bicycle at (x, y) with heading theta
            ...

        animate_trajectory(states, draw_bicycle, "bicycle.gif", fps=20, goal=(0, 0))
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        draw_fn(ax, states[frame], **draw_kwargs)

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000/fps)

    # Use tight layout to minimize whitespace
    plt.tight_layout()
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close()
