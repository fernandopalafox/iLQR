"""
test_ilqr_parking.py

Test iLQR on a NONLINEAR car parking problem with realistic steering dynamics.

System:
    State:   x = [px, py, θ, v, δ]
             (pos_x, pos_y, heading, velocity, steering angle)
    Control: u = [a, ω_δ]
             (acceleration, steering rate)
    Dynamics: Nonlinear kinematic bicycle model where steering angle is a state.
    Cost:     Quadratic cost to park at the origin (0,0) with zero
              heading, velocity, and steering angle.

This test validates iLQR on a classic motion planning problem, finding a
smooth trajectory to autonomously park a car. The updated dynamics prevent
instantaneous steering changes, leading to a more realistic path.
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR
from utils.animation import animate_trajectory


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.1   # [s] integration step
T = 50     # horizon length (50 steps * 0.1s = 5s)
# --- MODIFIED: State dimension is now 5 to include steering angle δ --- #
n, m = 5, 2  # state & control dimensions

# --- Physical constants --- #
WHEELBASE = 0.5  # vehicle wheelbase [m]
CAR_L = 0.6      # car length for drawing
CAR_W = 0.3      # car width for drawing


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Kinematic bicycle model discrete-time dynamics (NONLINEAR).

    state   = [px, py, θ, v, δ]  (x, y, heading, velocity, steering angle)
    control = [a, ω_δ]           (acceleration, steering *rate*)
    θ is the angle from the positive x-axis.
    """
    # --- MODIFIED: Unpack new state and control vectors --- #
    px, py, theta, v, delta = state
    a, steer_rate = control

    # --- Equations of Motion (Kinematic Bicycle Model) --- #
    px_next = px + dt * v * jnp.cos(theta)
    py_next = py + dt * v * jnp.sin(theta)
    # Steering angle `delta` is now from the state
    theta_next = theta + dt * v / WHEELBASE * jnp.tan(delta)
    v_next = v + dt * a
    # --- NEW: Dynamics for the steering angle itself --- #
    delta_next = delta + dt * steer_rate

    return jnp.array([px_next, py_next, theta_next, v_next, delta_next])


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
# --- MODIFIED: Goal state is now 5D, including zero steering angle --- #
x_goal = jnp.array([0., 0., 0., 0., 0.])

def build_car_cost(
    w_pos=2.0, w_head=1.0, w_vel=1.0, w_steer_pos=0.1,
    w_accel=1e-2, w_steer_rate=1e-1, term_weight=500.0
):
    """
    Quadratic cost for car parking.

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """
    # --- MODIFIED: Q is 5x5, R now penalizes steer *rate* --- #
    Q = jnp.diag(jnp.array([
        w_pos,        # penalize x position error
        w_pos,        # penalize y position error
        w_head,       # penalize heading error
        w_vel,        # penalize velocity error
        w_steer_pos,  # NEW: penalize non-zero steering angle
    ]))
    R = jnp.diag(jnp.array([w_accel, w_steer_rate]))  # penalize control effort

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u)"""
        x_err = x - x_goal
        # Wrap angle error to [-pi, pi] for correctness
        x_err = x_err.at[2].set((x_err[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return x_err.T @ Q @ x_err + u.T @ R @ u

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T)"""
        x_err = x_T - x_goal
        x_err = x_err.at[2].set((x_err[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
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

cost = build_car_cost()


# ========================================================================== #
# 3. Problem setup
# ========================================================================== #
# --- MODIFIED: Initial state is now 5D, starting with zero steering --- #
x0 = jnp.array([-2.0, 1.0, 1.0, 0.0, 0.0])
u_init = jnp.zeros((T, m))  # zero-controls initial guess

# --- MODIFIED: Pass the new dimensions to the solver --- #
dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims, max_iterations=100)


# ========================================================================== #
# 4. Solve
# ========================================================================== #
print("=" * 70)
print("TEST: Nonlinear Car Parking (with Realistic Steering)")
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
axs[0].plot(t, states_np[:, 0], label="px (pos x)")
axs[0].plot(t, states_np[:, 1], label="py (pos y)")
axs[0].plot(t, states_np[:, 2], label="θ (heading)", linestyle="-.")
axs[0].plot(t, states_np[:, 3], label="v (velocity)", linestyle=":")
# --- NEW: Plot the steering angle state --- #
axs[0].plot(t, states_np[:, 4], label="δ (steering)", linestyle="--")
axs[0].axhline(y=x_goal[0], color='gray', linestyle='--', label='Goal')
axs[0].set_title("State Trajectories")
axs[0].set_ylabel("Position / Angle / Vel")
axs[0].grid(alpha=0.5)
axs[0].legend()

# (b) Control inputs
axs[1].step(t[:-1], controls_np[:, 0], where="post", label="a (acceleration)")
# --- MODIFIED: The second control is now steering *rate* --- #
axs[1].step(t[:-1], controls_np[:, 1], where="post", label="ω_δ (steer rate)")
axs[1].set_title("Control Inputs")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("Accel / Angle Rate")
axs[1].grid(alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.savefig("figures/test_ilqr_parking.png", dpi=300)
plt.close()

print("Plot saved to: figures/test_ilqr_parking.png")


# ========================================================================== #
# 6. Animation
# ========================================================================== #
# --- NEW: Wheel constants for drawing a more realistic car --- #
WHEEL_L = 0.18   # [m] wheel length
WHEEL_W = 0.07   # [m] wheel width

def draw_car(ax, state, world_size=3.0):
    """
    Draw a more realistic car with steerable wheels for a given state.

    The car's state (px, py) is the center of the rear axle, which is
    consistent with the kinematic model. The drawing is constructed
    relative to the car's geometric center.
    """
    # --- MODIFIED: Unpack full 5D state, including steering angle --- #
    px_rear_axle, py_rear_axle, theta, _, delta = state

    # --- 1. Find car's geometric center from its rear axle state --- #
    # The geometric center is `WHEELBASE/2` ahead of the rear axle.
    R_body = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta),  jnp.cos(theta)]])
    
    offset_from_rear_axle = jnp.array([WHEELBASE / 2, 0.])
    px_center, py_center = jnp.array(
        [px_rear_axle, py_rear_axle]
    ) + R_body @ offset_from_rear_axle

    # --- 2. Define component vertices in the car's local frame --- #
    # (Centered at the car's geometric origin)
    
    # Car body vertices
    l, w = CAR_L, CAR_W
    body_verts_local = jnp.array([
        [-l/2, w/2], [l/2, w/2], [l/2, -w/2], [-l/2, -w/2]
    ])

    # Wheel vertices (centered at their own origin)
    wl, ww = WHEEL_L, WHEEL_W
    wheel_verts_origin = jnp.array([
        [-wl/2, ww/2], [wl/2, ww/2], [wl/2, -ww/2], [-wl/2, -ww/2]
    ])

    # Wheel center positions relative to the car's geometric center
    y_pos = w / 2 * 1.1  # Place wheels slightly outside the body width
    front_axle_x = WHEELBASE / 2
    rear_axle_x = -WHEELBASE / 2
    
    wheel_centers_local = {
        "fr": jnp.array([front_axle_x, -y_pos]), # front-right
        "fl": jnp.array([front_axle_x,  y_pos]), # front-left
        "rr": jnp.array([rear_axle_x,  -y_pos]), # rear-right
        "rl": jnp.array([rear_axle_x,   y_pos]), # rear-left
    }

    # --- 3. Transform and draw each component --- #
    @jax.vmap
    def to_world(v_local):
        """Helper to rotate and translate local vertices to world frame."""
        return R_body @ v_local + jnp.array([px_center, py_center])

    # Draw body
    body_verts_world = to_world(body_verts_local)
    ax.add_patch(Polygon(
        body_verts_world, closed=True, facecolor='cornflowerblue', edgecolor='black'
    ))

    # Draw wheels
    R_steer = jnp.array([[jnp.cos(delta), -jnp.sin(delta)],
                         [jnp.sin(delta),  jnp.cos(delta)]])
    
    # Front wheels (apply steering rotation)
    fr_verts_steered = jax.vmap(lambda v: R_steer @ v)(wheel_verts_origin)
    fr_verts_local = fr_verts_steered + wheel_centers_local["fr"]
    ax.add_patch(Polygon(
        to_world(fr_verts_local), closed=True, facecolor='dimgray', edgecolor='black'
    ))

    fl_verts_steered = jax.vmap(lambda v: R_steer @ v)(wheel_verts_origin)
    fl_verts_local = fl_verts_steered + wheel_centers_local["fl"]
    ax.add_patch(Polygon(
        to_world(fl_verts_local), closed=True, facecolor='dimgray', edgecolor='black'
    ))
    
    # Rear wheels (no steering rotation)
    rr_verts_local = wheel_verts_origin + wheel_centers_local["rr"]
    rl_verts_local = wheel_verts_origin + wheel_centers_local["rl"]
    ax.add_patch(Polygon(
        to_world(rr_verts_local), closed=True, facecolor='dimgray', edgecolor='black'
    ))
    ax.add_patch(Polygon(
        to_world(rl_verts_local), closed=True, facecolor='dimgray', edgecolor='black'
    ))
    
    # --- 4. Draw the parking spot (goal) --- #
    # The goal state has the rear axle at (0,0), so the spot is centered
    # `WHEELBASE/2` ahead of the origin.
    goal_center = jnp.array([WHEELBASE / 2, 0.])
    spot_verts = body_verts_local + goal_center
    ax.add_patch(Polygon(spot_verts, closed=True, facecolor='lightgray',
                         edgecolor='dimgray', linestyle='--', zorder=0))

    # --- 5. Set plot style --- #
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)


print("Creating animation...")
animate_trajectory(states_np, draw_car, "figures/test_ilqr_parking.gif",
                   fps=20, world_size=3.0)
print("Animation saved to: figures/test_ilqr_parking.gif\n")