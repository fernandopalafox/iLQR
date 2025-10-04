"""
test_ilqr_unit.py

Test iLQR on a LINEAR-QUADRATIC system.

System:
    State:   x = [x_pos, y_pos, θ]  (position and heading)
    Control: u = [v, ω]              (velocity commands)
    Dynamics: Linear time-invariant (x+ = Ax + Bu)
    Cost:     Quadratic (drive to origin)

This is a unit test where iLQR should match the LQR solution since the
problem is exactly linear-quadratic (no nonlinearity). This script
computes both solutions and verifies that they are equivalent.
"""

import time
from collections import namedtuple
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ========================================================================== #
# Import solvers
# ========================================================================== #
# NOTE: Assuming the LQR solver is available alongside iLQR
from ilqr import iLQR, LQR


# ========================================================================== #
# 1. System definition
# ========================================================================== #
dt = 0.1  # [s] integration step
T = 30  # horizon length
n, m = 3, 2  # state & control dimensions

# Linear dynamics matrices (time-invariant)
A = jnp.eye(n)
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
        return x @ Q @ x + u @ R @ u

    @jax.jit
    def terminal_cost(x_T):
        return x_T @ Qf @ x_T

    @jax.jit
    def traj_cost(xs, us):
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


# ========================================================================== #
# 4. Solve with both iLQR and LQR
# ========================================================================== #

# (a) Solve with iLQR
# -------------------------------------------------------------------------- #
print("=" * 70)
print("SOLVING WITH iLQR")
print("=" * 70)
dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, T, dims)

# Warmup run to compile JAX functions
(states_ilqr, controls_ilqr), _ = ilqr.solve(x0, u_init)
jax.block_until_ready((states_ilqr, controls_ilqr))

# Timed run
start_time = time.time()
(states_ilqr, controls_ilqr), (success_ilqr, stats_ilqr) = ilqr.solve(x0,
                                                                      u_init)
ilqr.print_stats(stats_ilqr.block_until_ready())
print(f"\niLQR solve time: {time.time() - start_time:.4f} s")

states_ilqr_np = jax.device_get(states_ilqr)
controls_ilqr_np = jax.device_get(controls_ilqr)

print(f"Success flag   : {success_ilqr}")
print(f"Final state    : {states_ilqr_np[-1]}")


# (b) Solve with LQR
# -------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("SOLVING WITH LQR (for comparison)")
print("=" * 70)

# The LQR solver expects time-varying matrices. For this LTI problem, we
# stack the constant matrices along a new time dimension.
A_lqr = jnp.array([A] * T)
B_lqr = jnp.array([B] * T)
Q_lqr = jnp.array([Q] * T + [Qf])
R_lqr = jnp.array([R] * T)

# Use namedtuples to create simple objects with the expected attributes
DynamicsLQR = namedtuple("DynamicsLQR", ["A", "B"])
CostLQR = namedtuple("CostLQR", ["Q", "R"])
dynamics_lqr = DynamicsLQR(A=A_lqr, B=B_lqr)
cost_lqr = CostLQR(Q=Q_lqr, R=R_lqr)

lqr_solver = LQR(cost_lqr, dynamics_lqr)

start_time_lqr = time.time()
(states_lqr, controls_lqr), _ = lqr_solver.solve(x0, u_init)
jax.block_until_ready((states_lqr, controls_lqr))
print(f"LQR solve time: {time.time() - start_time_lqr:.4f} s")

states_lqr_np = jax.device_get(states_lqr)
controls_lqr_np = jax.device_get(controls_lqr)
print(f"Final state    : {states_lqr_np[-1]}")


# ========================================================================== #
# 5. Compare iLQR and LQR solutions
# ========================================================================== #
state_diff = jnp.linalg.norm(states_ilqr_np - states_lqr_np)
control_diff = jnp.linalg.norm(controls_ilqr_np - controls_lqr_np)

print("\n" + "=" * 70)
print("COMPARISON: iLQR vs. LQR")
print("=" * 70)
print(f"Difference in state trajectories (L2 norm) : {state_diff:.6e}")
print(f"Difference in control trajectories (L2 norm): {control_diff:.6e}")
print("=" * 70)


# ========================================================================== #
# 6. Visualization
# ========================================================================== #
t = jnp.arange(T + 1) * dt
fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
fig.suptitle("iLQR vs. LQR Solution for Linear-Quadratic Problem", fontsize=16)

# (a) XY trajectory
axs[0].plot(states_ilqr_np[:, 0], states_ilqr_np[:, 1], "b-", lw=2, 
            label="iLQR")
axs[0].plot(states_lqr_np[:, 0], states_lqr_np[:, 1], "r--", lw=2, label="LQR")
axs[0].plot(x0[0], x0[1], "go", ms=10, label="Start")
axs[0].plot(0, 0, "k*", ms=15, label="Target")
axs[0].set_title("XY Trajectory")
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
axs[0].axis("equal")
axs[0].grid(alpha=0.3)
axs[0].legend()

# (b) State vs time
line_styles = ["-", "-", "--"]
colors = ["C0", "C1", "C2"]
state_labels = ["x", "y", "θ"]
for i in range(n):
    axs[1].plot(
        t,
        states_ilqr_np[:, i],
        color=colors[i],
        linestyle="-",
        label=f"{state_labels[i]} (iLQR)",
    )
    axs[1].plot(
        t,
        states_lqr_np[:, i],
        color=colors[i],
        linestyle="--",
        label=f"{state_labels[i]} (LQR)",
    )
axs[1].set_title("State Trajectories")
axs[1].set_xlabel("time [s]")
axs[1].set_ylabel("state")
axs[1].grid(alpha=0.3)
axs[1].legend(ncol=3)

# (c) Control inputs
control_labels = ["v (linear)", "ω (angular)"]
for i in range(m):
    axs[2].step(
        t[:-1],
        controls_ilqr_np[:, i],
        where="post",
        color=colors[i],
        linestyle="-",
        label=f"{control_labels[i]} (iLQR)",
    )
    axs[2].step(
        t[:-1],
        controls_lqr_np[:, i],
        where="post",
        color=colors[i],
        linestyle="--",
        label=f"{control_labels[i]} (LQR)",
    )
axs[2].set_title("Control Inputs")
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("control")
axs[2].grid(alpha=0.3)
axs[2].legend()

# Save the figure
plt.savefig("figures/test_ilqr_vs_lqr.png", dpi=300)
plt.close()

print("\nPlot saved to: figures/test_ilqr_vs_lqr.png\n")