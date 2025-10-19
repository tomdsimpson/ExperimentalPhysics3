# Tom Simpson
# Joint Linear Fit on y-data
# 12/10/25

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# --- File Management --- #
SCRIPT_DIR = Path(__file__).parent.resolve()

def read_data(dimension="y", n=1):
    f = SCRIPT_DIR / "BallBearingOnlyY" / "raw_csv" / f"{dimension}_motion_{n}.csv"
    raw_data = np.genfromtxt(f, delimiter=",")
    time = raw_data.T[0, 1:]  # trim column header
    pos = raw_data.T[1, 1:]
    return time, pos


# --- Linear Drag Model --- #
def linear_y_model(g, v_i, n, t, y_i=0):
    """
    Linear drag fall model (downward positive):
        v(t) = (v_i - g*n) * exp(-t/n) + g*n
        y(t) = (n*v_i - g*n**2) * (1 - exp(-t/n)) + g*n*t + y_i
    """
    return (n * v_i - g * n**2) * (1 - np.exp(-t / n)) + g * n * t + y_i


# --- Loss Function --- #
def rmse_loss_ly_joint(params, times, true_y):
    """
    Joint RMSE loss for [g, v_i, n]
    """
    g, v_i, n = params
    pred_y = linear_y_model(g, v_i, n, times)
    return np.mean((true_y - pred_y) ** 2)


# --- Fitting Routine --- #
def fit_linear_y(times, y_positions, bounds):
    """
    Two-stage optimization: differential evolution + local polish
    """
    # Global search
    result_global = differential_evolution(
        lambda p: rmse_loss_ly_joint(p, times, y_positions),
        bounds,
        polish=False,
    )

    # Local refinement
    result_local = minimize(
        lambda p: rmse_loss_ly_joint(p, times, y_positions),
        result_global.x,
        bounds=bounds,
        method="L-BFGS-B",
    )

    return result_local.x, result_local.fun


# --- Analysis Across Trials --- #
def determine_g_joint_linear(y_bounds):
    gs, ns, vis = [], [], []

    for i in range(5):
        times, y_positions = read_data("y", n=i + 1)

        # Fit [g, v_i, n]
        params, loss = fit_linear_y(times, y_positions, y_bounds)
        opt_g, opt_v_i, opt_n = params
        gs.append(opt_g)
        ns.append(opt_n)
        vis.append(opt_v_i)

        print(f"Trial {i+1} | g={opt_g:.4f}, v_i={opt_v_i:.4f}, n={opt_n:.4f}, loss={loss:.4e}")

        # Create side-by-side plots
        y_pred = linear_y_model(opt_g, opt_v_i, opt_n, times)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Fit plot 
        axes[0].plot(times, y_pred, label="Fit", color="red")
        axes[0].plot(times, y_positions, ".", label="Data", color="black")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Position y (m)")
        axes[0].legend()
        axes[0].set_title(f"Trial {i+1} Linear Fit")

        # Residual plot
        axes[1].plot(times, y_pred - y_positions, "x", color="black")
        axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Residual (m)")
        axes[1].set_title(f"Trial {i+1} Residual Plot")

        plt.tight_layout()
        plt.show()

    # Stats
    opt_g_mean = np.mean(gs)
    opt_g_std = np.std(gs)
    opt_g_stderr = opt_g_std / np.sqrt(len(gs))

    print("\n--- Linear Joint Fit Summary ---")
    print(f"g = {opt_g_mean:.4f} ± {opt_g_stderr:.3f}")
    print(f"Mean n = {np.mean(ns):.4f}, Mean v_i = {np.mean(vis):.4f}")
    print("-----------------------------------")


# --- Main --- #
def main():
    # Reasonable physical bounds
    y_bounds = [
        (5, 15),      # g (m/s²)
        (0, 5),       # v_i (m/s)
        (0.001, 10),  # n (s)
    ]

    determine_g_joint_linear(y_bounds)


if __name__ == "__main__":
    main()
