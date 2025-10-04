"""
test_ilqr_cartpole.py

Test iLQR on a NONLINEAR cart-pole system.

System:
    State:   x = [x, x_dot, θ, θ_dot]  (cart pos/vel, pole ang/vel)
    Control: u = [F]                   (horizontal force on cart)
    Dynamics: Nonlinear cart-pole dynamics
    Cost:     Quadratic in state and control (balance pole at origin)

This test validates iLQR on an unstable nonlinear system where the goal is
to find a stabilizing control policy.
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR
from utils.animation import animate_trajectory


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.01  # [s] integration step
T = 200    # horizon length (200 steps * 0.05s = 10s)
n, m = 4, 1  # state & control dimensions

# --- Physical constants --- #
g = 9.81           # gravity [m/s^2]
m_c = 1.0          # cart mass [kg]
m_p = 0.1          # pole mass [kg]
pole_len = 1.0     # pole length [m]
l = pole_len / 2.  # distance from pivot to pole CoM [m]


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Cart-pole discrete-time dynamics (NONLINEAR).

    state   = [x, x_dot, θ, θ_dot]
    control = [F]
    θ is the angle from the upward vertical.
    """
    x, x_dot, th, th_dot = state
    F = control[0]

    # --- Equations of Motion --- #
    sin_th = jnp.sin(th)
    cos_th = jnp.cos(th)
    
    # Angular acceleration of the pole (th_ddot)
    temp = (F + m_p * l * th_dot**2 * sin_th) / (m_c + m_p)
    th_ddot = (g * sin_th - cos_th * temp) / \
              (l * (4.0/3.0 - m_p * cos_th**2 / (m_c + m_p)))

    # Linear acceleration of the cart (x_ddot)
    x_ddot = temp - m_p * l * th_ddot * cos_th / (m_c + m_p)

    # --- Euler integration --- #
    x_next = x + dt * x_dot
    x_dot_next = x_dot + dt * x_ddot
    th_next = th + dt * th_dot
    th_dot_next = th_dot + dt * th_ddot
    
    return jnp.array([x_next, x_dot_next, th_next, th_dot_next])


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
def build_cartpole_cost(
    w_x=1.0, w_th=10.0, w_vel=0.1, w_u=1e-3, term_weight=100.0
):
    """
    Quadratic cost: balance pole upright (θ=0) at the origin (x=0).

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """
    # Define state and control weighting matrices
    Q = jnp.diag(jnp.array([w_x, w_vel, w_th, w_vel]))
    R = w_u * jnp.eye(m)

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u) = x'Qx + u'Ru"""
        return x.T @ Q @ x + u.T @ R @ u

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T) = x_T' Q_f x_T"""
        return term_weight * (x_T.T @ Q @ x_T)

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

cost = build_cartpole_cost(term_weight=100.0)


# ========================================================================== #
# 3. Problem setup
# ========================================================================== #
# Initial state: pole deviated 0.1 rad (5.7 deg) from vertical
x0 = jnp.array([0.0, -0.1, 0.3, 0.0])
u_init = jnp.zeros((T, m))  # zero-controls initial guess

dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims, reg_param_max=1e1)


# ========================================================================== #
# 4. Solve
# ========================================================================== #
print("=" * 70)
print("TEST: Nonlinear Cart-Pole Balancing")
print("=" * 70)

start_time = time.time()
# Set regularization lower than default to allow for more aggressive steps
(states, controls), (success, stats) = ilqr.solve(x0, u_init)

ilqr.print_stats(stats.block_until_ready())
print(f"\niLQR solve time: {time.time() - start_time:.2f} s")

states_np = jax.device_get(states)
controls_np = jax.device_get(controls)

print(f"Success flag   : {success}")
print(f"Final state    : {states_np[-1]}")
print(f"Final error    : {jnp.linalg.norm(states_np[-1]):.4f}")
print("=" * 70)


# ========================================================================== #
# 5. Visualization
# ========================================================================== #
t = jnp.arange(T + 1) * dt

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# (a) State vs time
axs[0].plot(t, states_np[:, 0], label="x (pos)")
axs[0].plot(t, states_np[:, 2], label="θ (angle)", linestyle="-.")
axs[0].set_title("State Trajectories")
axs[0].set_ylabel("Position / Angle")
axs[0].grid(alpha=0.5)
axs[0].legend()

# (b) Control inputs
axs[1].step(t[:-1], controls_np[:, 0], where="post", label="F (force)")
axs[1].set_title("Control Input")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("Force [N]")
axs[1].grid(alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.savefig("figures/test_ilqr_cartpole.png", dpi=300)
plt.close()

print("Plot saved to: figures/test_ilqr_cartpole.png")


# ========================================================================== #
# 6. Animation
# ========================================================================== #
def draw_cartpole(ax, state, world_size=5.0):
    """Draw the cart-pole system for a given state."""
    cart_w, cart_h = 0.5, 0.25
    x_cart, _, theta, _ = state

    # --- Cart --- #
    cart_y = 0  # Cart is on a track at y=0
    cart_x = x_cart - cart_w / 2
    ax.add_patch(Rectangle((cart_x, cart_y - cart_h/2), cart_w, cart_h,
                            facecolor='royalblue', edgecolor='black'))

    # --- Pole --- #
    pivot_x, pivot_y = x_cart, cart_y
    pole_tip_x = pivot_x + pole_len * jnp.sin(theta)
    pole_tip_y = pivot_y + pole_len * jnp.cos(theta)
    ax.plot([pivot_x, pole_tip_x], [pivot_y, pole_tip_y], 'darkorange', lw=4)
    ax.add_patch(Circle((pivot_x, pivot_y), 0.05, color='black')) # Pivot

    # --- Track --- #
    ax.plot([-world_size, world_size], [0, 0], 'k--', lw=1)

    # --- Set axis properties --- #
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-0.5, pole_len * 1.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)


print("Creating animation...")
animate_trajectory(states_np, draw_cartpole, "figures/test_ilqr_cartpole.gif",
                   fps=int(1/dt), world_size=1.0)
print("Animation saved to: figures/test_ilqr_cartpole.gif\n")