"""
test_ilqr_bicycle.py

Test iLQR on a NONLINEAR unicycle (bicycle) model.

System:
    State:   x = [x_pos, y_pos, θ]  (position and heading)
    Control: u = [v, ω]              (linear and angular velocity)
    Dynamics: Nonlinear kinematic unicycle model
    Cost:     Quadratic in state and control (drive to origin)

This test validates iLQR on a nonlinear system where linearization
is essential for convergence.
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR
from utils.animation import animate_trajectory


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.1  # [s] integration step
T = 30  # horizon length
n, m = 3, 2  # state & control dimensions


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Unicycle discrete-time dynamics (NONLINEAR).

    state   = [x, y, θ]   position and heading
    control = [v, ω]      linear & angular velocity commands
    """
    x, y, th = state
    v, w = control

    x_next = x + dt * v * jnp.cos(th)
    y_next = y + dt * v * jnp.sin(th)
    th_next = th + dt * w
    return jnp.array([x_next, y_next, th_next])


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
def build_unicycle_cost(
    w_pos=1.0, w_th=0.1, w_u=1e-2, term_weight=100.0
):
    """
    Quadratic cost: drive state to origin, minimize control effort.

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u)"""
        pos = w_pos * (x[0] ** 2 + x[1] ** 2)
        heading = w_th * (x[2] ** 2)
        control = w_u * jnp.sum(u**2)
        return pos + heading + control

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T)"""
        return term_weight * (
            x_T[0] ** 2 + x_T[1] ** 2 + 0.1 * x_T[2] ** 2
        )

    @jax.jit
    def traj_cost(xs, us):
        """Full trajectory cost"""
        step_costs = jax.vmap(stage_cost)(xs[:-1], us)
        return jnp.sum(step_costs) + terminal_cost(xs[-1])

    return {
        "stage": stage_cost,
        "terminal": terminal_cost,
        "traj": traj_cost,
    }


cost = build_unicycle_cost(term_weight=100.0)


# ========================================================================== #
# 3. Problem setup
# ========================================================================== #
x0 = jnp.array([-6.0, 10.0, 0.9])  # initial state
u_init = jnp.zeros((T, m))  # zero-controls initial guess

dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims)


# ========================================================================== #
# 4. Solve
# ========================================================================== #
print("=" * 70)
print("TEST: Nonlinear Unicycle (Bicycle) Model")
print("=" * 70)

start_time = time.time()
(states, controls), (success, stats) = ilqr.solve(x0, u_init)

ilqr.print_stats(stats.block_until_ready())
print(f"\niLQR solve time: {time.time() - start_time:.2f} s")

states_np = jax.device_get(states)
controls_np = jax.device_get(controls)

print(f"Success flag   : {success}")
print(f"Final state    : {states_np[-1]}")
print(f"Distance to 0  : {jnp.linalg.norm(states_np[-1][:2]):.4f} m")
print("=" * 70)


# ========================================================================== #
# 5. Visualization
# ========================================================================== #
t = jnp.arange(T + 1) * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# (a) XY trajectory
axs[0].plot(states_np[:, 0], states_np[:, 1], marker="o", ms=2)
axs[0].plot(states_np[0, 0], states_np[0, 1], "go", ms=8, label="Start")
axs[0].plot(states_np[-1, 0], states_np[-1, 1], "ro", ms=8, label="End")
axs[0].plot(0, 0, "k*", ms=12, label="Target")
axs[0].set_title("Nonlinear Unicycle: XY Trajectory")
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
axs[0].axis("equal")
axs[0].grid(alpha=0.3)
axs[0].legend()

# (b) State vs time
axs[1].plot(t, states_np[:, 0], label="x")
axs[1].plot(t, states_np[:, 1], label="y")
axs[1].plot(t, states_np[:, 2], label="θ", linestyle="--")
axs[1].set_title("State Trajectories")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("state")
axs[1].grid(alpha=0.3)
axs[1].legend(ncol=3)

# (c) Control inputs
axs[2].step(t[:-1], controls_np[:, 0], where="post", label="v (linear)")
axs[2].step(t[:-1], controls_np[:, 1], where="post", label="ω (angular)")
axs[2].set_title("Control Inputs")
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("control")
axs[2].grid(alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig("figures/test_ilqr.png", dpi=300)
plt.close()

print("Plot saved to: figurestest_ilqr.png")


# ========================================================================== #
# 6. Animation
# ========================================================================== #
def draw_bicycle(ax, state, trajectory=None, goal=(0, 0)):
    """Draw bicycle at given state with trajectory trail and goal."""
    x, y, theta = state

    # Draw trajectory trail
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3,
                linewidth=1)

    # Draw goal
    ax.plot(goal[0], goal[1], 'k*', ms=20, label='Goal')

    # Draw bicycle as arrow showing position and heading
    arrow_length = 0.8
    dx = arrow_length * jnp.cos(theta)
    dy = arrow_length * jnp.sin(theta)
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.42,
                            head_width=0.2, head_length=0.2,
                            fc='red', ec='darkred', linewidth=2))

    # Set axis properties
    ax.set_xlim(states_np[:, 0].min() - 2, states_np[:, 0].max() + 2)
    ax.set_ylim(states_np[:, 1].min() - 2, states_np[:, 1].max() + 2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend()


print("Creating animation...")
animate_trajectory(states_np, draw_bicycle, "figures/test_ilqr.gif",
                   fps=10, trajectory=states_np, goal=(0, 0))
print("Animation saved to: figures/test_ilqr.gif\n")
