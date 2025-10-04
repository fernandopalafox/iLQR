"""
test_ilqr_rocket.py

Test iLQR on a NONLINEAR planar rocket landing problem.

System:
    State:   x = [px, py, vx, vy, θ, ω]
             (pos, vel, angle from horizontal, angular vel)
    Control: u = [F, τ]
             (main thruster force, torque)
    Dynamics: Nonlinear 2D rocket dynamics under gravity
    Cost:     Quadratic cost to land at origin (0,0) with zero
              velocity and upright angle (π/2).

This test validates iLQR on a classic optimal control problem, finding a
fuel-efficient trajectory to a soft landing.
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR
from utils.animation import animate_trajectory


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.02  # [s] integration step (smaller for more accuracy)
T = 250    # horizon length (250 steps * 0.02s = 5s)
n, m = 6, 2  # state & control dimensions

# --- Physical constants --- #
g = 9.81         # gravity [m/s^2]
mass = 1.0       # rocket mass [kg]
inertia = 0.1    # rocket moment of inertia [kg*m^2]
ROCKET_H = 0.5   ## MODIFIED ##: Rocket height, for goal state and drawing
ROCKET_W = 0.25  ## MODIFIED ##: Rocket width, for drawing


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Planar rocket discrete-time dynamics (NONLINEAR).

    state   = [px, py, vx, vy, θ, ω]
    control = [F_main, τ]
    θ is the angle from the positive x-axis (horizontal).
    """
    px, py, vx, vy, theta, omega = state
    F_main, tau = control

    # --- Equations of Motion --- #
    # Acceleration in world frame
    ax = (F_main * jnp.cos(theta)) / mass
    ay = (F_main * jnp.sin(theta)) / mass - g
    alpha = tau / inertia

    # --- Euler integration --- #
    px_next = px + dt * vx
    py_next = py + dt * vy
    vx_next = vx + dt * ax
    vy_next = vy + dt * ay
    theta_next = theta + dt * omega
    omega_next = omega + dt * alpha

    return jnp.array([px_next, py_next, vx_next, vy_next, theta_next, omega_next])


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
## MODIFIED ##: The goal for py is now ROCKET_H / 2 so the base lands at y=0.
# The state (px, py) represents the center of mass.
x_goal = jnp.array([0., ROCKET_H / 2.0, 0., 0., jnp.pi / 2.0, 0.])

def build_rocket_cost(
    w_pos=1.0, w_vel=1.0, w_ang=5.0, w_F=1e-4, w_tau=1e-4, term_weight=500.0
):
    """
    Quadratic cost for rocket landing.

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """
    # Define state and control weighting matrices
    Q = jnp.diag(jnp.array([
        w_pos, w_pos,  # penalize x, y position error
        w_vel, w_vel,  # penalize vx, vy velocity error
        w_ang, w_ang   # penalize θ, ω angular error
    ]))
    R = jnp.diag(jnp.array([w_F, w_tau]))  # penalize control effort

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u)"""
        x_err = x - x_goal
        # Wrap angle error to [-pi, pi] for correctness
        x_err = x_err.at[4].set((x_err[4] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return x_err.T @ Q @ x_err + u.T @ R @ u

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T)"""
        x_err = x_T - x_goal
        x_err = x_err.at[4].set((x_err[4] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return term_weight * (x_err.T @ Q @ x_err)

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

cost = build_rocket_cost()


# ========================================================================== #
# 3. Problem setup
# ========================================================================== #
# Initial state: rocket is at (3,2), moving down-left, and tilted
x0 = jnp.array([3.0, 2.0, -1.0, -1.0, jnp.pi / 2.0 + 0.5, 0.1])
u_init = jnp.zeros((T, m))  # zero-controls initial guess

dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims)


# ========================================================================== #
# 4. Solve
# ========================================================================== #
print("=" * 70)
print("TEST: Nonlinear Planar Rocket Landing")
print("=" * 70)

start_time = time.time()
(states, controls), (success, stats) = ilqr.solve(x0, u_init)

ilqr.print_stats(stats.block_until_ready())
print(f"\niLQR solve time: {time.time() - start_time:.2f} s")

states_np = jax.device_get(states)
controls_np = jax.device_get(controls)
final_error = states_np[-1] - x_goal

print(f"Success flag   : {success}")
print(f"Final state    : {states_np[-1]}")
print(f"Final error    : {jnp.linalg.norm(final_error):.4f}")
print("=" * 70)


# ========================================================================== #
# 5. Visualization
# ========================================================================== #
t = jnp.arange(T + 1) * dt

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# (a) State vs time
axs[0].plot(t, states_np[:, 0], label="x (pos)")
axs[0].plot(t, states_np[:, 1], label="y (pos)")
axs[0].plot(t, states_np[:, 4], label="θ (angle)", linestyle="-.")
axs[0].axhline(y=x_goal[1], color='green', linestyle='--', label='y goal') ## MODIFIED ##
axs[0].axhline(y=jnp.pi/2, color='gray', linestyle='--', label='θ goal')
axs[0].set_title("State Trajectories")
axs[0].set_ylabel("Position / Angle")
axs[0].grid(alpha=0.5)
axs[0].legend()

# (b) Control inputs
axs[1].step(t[:-1], controls_np[:, 0], where="post", label="F_main (thrust)")
axs[1].step(t[:-1], controls_np[:, 1], where="post", label="τ (torque)")
axs[1].set_title("Control Inputs")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("Force / Torque")
axs[1].grid(alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.savefig("figures/test_ilqr_rocket.png", dpi=300)
plt.close()

print("Plot saved to: figures/test_ilqr_rocket.png")


# ========================================================================== #
# 6. Animation
# ========================================================================== #
def draw_rocket(ax, state, world_size=4.0):
    """Draw the rocket for a given state."""
    px, py, _, _, theta, _ = state

    # --- Rocket Body (a cute triangle) --- #
    ## MODIFIED ##: Use global constants for rocket geometry
    h, w = ROCKET_H, ROCKET_W
    # Define vertices in rocket's body frame (origin is center of mass)
    nose = jnp.array([h / 2, 0])
    l_base = jnp.array([-h / 2, -w / 2])
    r_base = jnp.array([-h / 2, w / 2])
    vertices = jnp.array([nose, l_base, r_base])

    # Standard 2D rotation matrix
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                   [jnp.sin(theta),  jnp.cos(theta)]])

    # Rotate and translate vertices to world frame
    verts_rot = jax.vmap(lambda v: R @ v)(vertices)
    verts_world = verts_rot + jnp.array([px, py])
    ax.add_patch(Polygon(verts_world, closed=True,
                         facecolor='skyblue', edgecolor='black'))

    # --- Cute Flame --- #
    flame_h, flame_w = 0.3, 0.15
    flame_v1 = jnp.array([-h / 2, 0])
    flame_v2 = jnp.array([-h / 2 - flame_h, -flame_w / 2])
    flame_v3 = jnp.array([-h / 2 - flame_h, flame_w / 2])
    flame_verts = jnp.array([flame_v1, flame_v2, flame_v3])
    flame_rot = jax.vmap(lambda v: R @ v)(flame_verts)
    flame_world = flame_rot + jnp.array([px, py])
    ax.add_patch(Polygon(flame_world, closed=True,
                         facecolor='orange', edgecolor='red'))
    
    # --- Landing Pad --- #
    pad_w = 1.0
    ax.plot([-pad_w / 2, pad_w / 2], [0, 0], 'k-', lw=3, zorder=0)

    # --- Set axis properties --- #
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-0.5, world_size)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)


print("Creating animation...")
animate_trajectory(states_np, draw_rocket, "figures/test_ilqr_rocket.gif",
                   fps=20, world_size=4.0)
print("Animation saved to: figures/test_ilqr_rocket.gif\n")