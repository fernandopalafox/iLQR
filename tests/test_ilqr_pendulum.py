"""
test_ilqr_pendulum.py

Test iLQR on the NONLINEAR inverted pendulum swing-up task.

System:
    State:   x = [θ, θ_dot]  (angle and angular velocity)
    Control: u = [τ]         (torque)
    Dynamics: Nonlinear inverted pendulum dynamics
    Cost:     Quadratic in control, and penalizes distance from
              the upright position (θ=0, θ_dot=0).

This test validates iLQR on a classic nonlinear control problem
that requires a significant change in state (swing-up).
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR
from utils.animation import animate_trajectory


# ========================================================================== #
# 1. System definition
# ========================================================================== #
# System parameters
g = 9.81   # gravity [m/s^2]
mass = 1.0   # mass [kg]
l = 1.0    # length [m]
b = 0.1    # damping [Nms/rad]
I = mass * l**2  # moment of inertia [kg*m^2]

# Simulation parameters
dt = 0.05  # [s] integration step
T = 100    # horizon length
n, m = 2, 1  # state & control dimensions


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Inverted pendulum discrete-time dynamics (NONLINEAR).

    state   = [θ, θ_dot]  angle & angular velocity
    control = [τ]         torque
    """
    theta, theta_dot = state
    tau = control[0]

    # Continuous-time dynamics: I * θ_ddot = τ - mgl*sin(θ) - b*θ_dot
    theta_ddot = (tau - mass * g * l * jnp.sin(theta) - b * theta_dot) / I

    # Euler integration
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * theta_ddot
    return jnp.array([theta_next, theta_dot_next])


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
def build_pendulum_cost(
    w_angle=1.0, w_vel=0.1, w_u=1e-3, term_weight=100.0
):
    """
    Cost: drive state to upright, minimize control effort.

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """
    x_target = jnp.array([0.0, 0.0]) # Target state (upright position)

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u)"""
        # Penalize distance from upright position (θ=0)
        # Using 1 - cos(θ) is a smooth way to represent angle error
        angle_err = w_angle * (1.0 - jnp.cos(x[0] - x_target[0]))
        vel_err = w_vel * (x[1] - x_target[1]) ** 2
        control_cost = w_u * jnp.sum(u**2)
        return angle_err + vel_err + control_cost

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T)"""
        angle_err = w_angle * (1.0 - jnp.cos(x_T[0] - x_target[0]))
        vel_err = w_vel * (x_T[1] - x_target[1]) ** 2
        return term_weight * (angle_err + vel_err)

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


cost = build_pendulum_cost(w_u=1e-4, term_weight=100.0)


# ========================================================================== #
# 3. Problem setup
# ========================================================================== #
x0 = jnp.array([jnp.pi, 0.0])  # initial state (hanging down at rest)
u_init = jnp.zeros((T, m))  # zero-controls initial guess

dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims)


# ========================================================================== #
# 4. Solve
# ========================================================================== #
print("=" * 70)
print("TEST: Nonlinear Inverted Pendulum Swing-up")
print("=" * 70)

start_time = time.time()
(states, controls), (success, stats) = ilqr.solve(x0, u_init)

ilqr.print_stats(stats.block_until_ready())
print(f"\niLQR solve time: {time.time() - start_time:.2f} s")

states_np = jax.device_get(states)
controls_np = jax.device_get(controls)

# Angle wrapping for nice display
final_angle_wrapped = (states_np[-1, 0] + jnp.pi) % (2 * jnp.pi) - jnp.pi

print(f"Success flag   : {success}")
print(f"Final state    : [{final_angle_wrapped:.4f}, {states_np[-1, 1]:.4f}]")
print(f"Final angle err: {jnp.abs(final_angle_wrapped):.4f} rad")
print("=" * 70)


# ========================================================================== #
# 5. Visualization
# ========================================================================== #
t = jnp.arange(T + 1) * dt
theta_wrapped = (states_np[:, 0] + jnp.pi) % (2 * jnp.pi) - jnp.pi

fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
fig.suptitle("Inverted Pendulum Swing-up", fontsize=16)

# (a) State vs time
axs[0].plot(t, theta_wrapped, label="θ [rad]")
axs[0].plot(t, states_np[:, 1], label="θ_dot [rad/s]", linestyle="--")
axs[0].set_title("State Trajectories")
axs[0].set_xlabel("time [s]")
axs[0].set_ylabel("state")
axs[0].grid(alpha=0.3)
axs[0].legend()

# (b) Control inputs
axs[1].step(t[:-1], controls_np[:, 0], where="post", label="τ (torque)")
axs[1].set_title("Control Input")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("Torque [Nm]")
axs[1].grid(alpha=0.3)
axs[1].legend()

plt.savefig("figures/test_ilqr_pendulum.png", dpi=300)
plt.close()

print("Plot saved to: figures/test_ilqr_pendulum.png")


# ========================================================================== #
# 6. Animation
# ========================================================================== #
def draw_pendulum(ax, state, length):
    """Draw pendulum at a given state."""
    theta = state[0]

    # Define points
    pivot = (0, 0)
    mass_pos = (pivot[0] + length * jnp.sin(theta),
                pivot[1] - length * jnp.cos(theta))

    # Clear axis and set properties
    ax.clear()
    ax.set_xlim(-length * 1.5, length * 1.5)
    ax.set_ylim(-length * 1.5, length * 1.5)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # Draw elements
    ax.plot([pivot[0], mass_pos[0]], [pivot[1], mass_pos[1]], 'k-', lw=3) # Rod
    ax.plot(pivot[0], pivot[1], 's', color='black', markersize=10)     # Pivot
    ax.plot(0, length, 'g*', ms=20, label='Target')                       # Target
    ax.plot(mass_pos[0], mass_pos[1], 'o', color='crimson', markersize=25,
            markeredgecolor='black')                                  # Mass
    ax.legend(loc='upper right')

# Use partial to pass the pendulum length to the drawing function
draw_fn = partial(draw_pendulum, length=l)

print("Creating animation...")
animate_trajectory(states_np, draw_fn, "figures/test_ilqr_pendulum.gif",
                   fps=int(1/dt))
print("Animation saved to: figures/test_ilqr_pendulum.gif\n")