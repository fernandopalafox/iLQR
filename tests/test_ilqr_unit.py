"""
test_ilqr_unit.py

Test iLQR on a LINEAR-QUADRATIC system.

System:
    State:   x = [x_pos, y_pos, θ]  (position and heading)
    Control: u = [v, ω]              (velocity commands)
    Dynamics: Linear time-invariant (x+ = Ax + Bu)
    Cost:     Quadratic (drive to origin)

This is a unit test where iLQR should match LQR solution since the
problem is exactly linear-quadratic (no nonlinearity).
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ========================================================================== #
# Import solver
# ========================================================================== #
from ilqr import iLQR


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.1  # [s] integration step
T = 30  # horizon length
n, m = 3, 2  # state & control dimensions

# Linear dynamics matrices
A = jnp.eye(n)  # Identity (position integrator)
B = jnp.array(
    [
        [dt, 0.0],  # x  ← v
        [0.0, dt],  # y  ← ω (treat as y-velocity)
        [0.0, dt],  # θ  ← ω
    ]
)


@jax.jit
def dynamics(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Linear discrete-time dynamics.

    x+ = A*x + B*u
    """
    return A @ state + B @ control


# ========================================================================== #
# 2. Cost definition
# ========================================================================== #
# Quadratic cost matrices
Q = jnp.diag(jnp.array([1.0, 1.0, 0.01]))  # state cost
R = 1e-2 * jnp.eye(m)  # control cost
Qf = 100.0 * Q  # terminal cost


def build_lq_cost():
    """
    Quadratic cost: x'Qx + u'Ru (drive to origin).

    Returns dict with 'stage', 'terminal', 'traj' cost functions.
    """

    @jax.jit
    def stage_cost(x, u):
        """Stage cost ℓ(x, u) = x'Qx + u'Ru"""
        return x @ Q @ x + u @ R @ u

    @jax.jit
    def terminal_cost(x_T):
        """Terminal cost Φ(x_T) = x_T' Qf x_T"""
        return x_T @ Qf @ x_T

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


cost = build_lq_cost()


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
print("TEST: Linear-Quadratic System")
print("=" * 70)

# Warmup run to compile JAX functions
(states, controls), (success, stats) = ilqr.solve(x0, u_init)
jax.block_until_ready((states, controls, success, stats))

# Timed run
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
axs[0].set_title("Linear-Quadratic: XY Trajectory")
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
plt.savefig("figures/ex_5_ilqr_unit.png", dpi=300)
plt.close()

print("Plot saved to: figures/ex_5_ilqr_unit.png\n")
