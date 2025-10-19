# Tom Simpson
# Parameter Sweep Model Fitting
# 09/10/25

# --- Module Imports --- #
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# Get the absolute path of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()


# Read the CSV
def read_data(dimension="x", n=1):

    f = SCRIPT_DIR / "PingPong1" / "raw_csv" / f"{dimension}_motion_{n}.csv" 
    raw_data = np.genfromtxt(f, delimiter=",")
    time = raw_data.T[0, 1:] # Trim column header
    pos = raw_data.T[1, 1:]
    return time, pos

# --- Linear --- #
def linear_x_model(n, v_i, t, x_i=0):
    return n*v_i*(1-np.exp(-t/n)) + x_i
def linear_y_model(g, v_i, n, t, y_i=0):
    return (n*v_i-(g*n**2))*(1-np.exp(-t/n)) + g*n*t + y_i

def rmse_loss_x(params, times, true_x):
    n, v_i = params
    pred_x = linear_x_model(n, v_i, times)
    return np.mean((true_x - pred_x)**2)
def rmse_loss_y(params, times, true_y, n):
    g, v_i = params
    pred_y = linear_y_model(g, v_i, n, times)
    return np.mean((true_y - pred_y)**2)

# --- Quadratic --- #
def quadratic_x_model(n, v_i, t, x_i=0):
    return n * np.log(1 + (t*v_i)/n) + x_i

def quadratic_y_model(g, v_i, n, t, y_i=0):
    eps = 1e-10  # Small value to avoid division by zero
    v_t = np.sqrt(max(g * n, eps))
    # Clip arctanh argument to avoid domain errors
    arg = np.clip(v_i / v_t, -1 + eps, 1 - eps)

    theta0 = np.arctanh(arg)
    theta_t = (g * t) / v_t + theta0
    num = np.cosh(theta_t)
    denom = np.cosh(theta0)
    # Avoid division by zero in log
    denom = np.where(denom == 0, eps, denom)
    result = y_i + n * np.log(num / denom)
    return result

def rmse_loss_qx(params, times, true_x):
    n, v_i = params
    pred_x = quadratic_x_model(n, v_i, times)
    return np.mean((true_x - pred_x)**2)
def rmse_loss_qy(params, times, true_y, n):
    g, v_i = params
    pred_y = quadratic_y_model(g, v_i, n, times)
    return np.mean((true_y - pred_y)**2)

def determine_g(x_bounds, y_bounds, x0, y0, mode="linear"):

    if mode == "linear":
        x_func = rmse_loss_x
        y_func = rmse_loss_y
    elif mode == "quadratic":
        x_func = rmse_loss_qx
        y_func = rmse_loss_qy

    gs = []

    for i in range(5):

        # Load Trial Data
        times_x, x_positions = read_data(dimension="x", n=i+1)
        times_y, y_positions = read_data(dimension="y", n=i+1)

        # Find N = k/m using minimize
        result_x = minimize(
            x_func, x0, args=(times_x, x_positions), bounds=x_bounds
        )
        opt_n, opt_v_i = result_x.x

        # Find g using minimize
        result_y = minimize(
            y_func, y0, args=(times_y, y_positions, opt_n), bounds=y_bounds
        )
        opt_g, opt_v_i = result_y.x
        print(f"Optimal g: {opt_g:.5}")
        gs.append(opt_g)

        if mode == "linear":
            plt.plot(times_y, y_positions, ".")
            plt.plot(times_y, linear_y_model(opt_g, opt_v_i, opt_n, times_y))
            plt.show()
        elif mode == "quadratic":
            plt.plot(times_y, y_positions, ".")
            plt.plot(times_y, quadratic_y_model(opt_g, opt_v_i, opt_n, times_y))
            plt.show()

    # Cleaning Value and finding STDE
    opt_g = np.mean(gs)
    stdev_g = np.std(gs)
    stder_g = stdev_g / np.sqrt(5)

    print("---")
    print(mode)
    print(f"Optimal g: {opt_g:.5} \u00B1 {stder_g:.2}")
    print("---")

def main():

    x_bounds = [(0, 1), (0, 2)]   
    y_bounds = [(0, 20), (0, 2)]
    x0 = (0.5, 0.5)
    y0 = (9, 0.5)

    determine_g(x_bounds, y_bounds, x0, y0, mode="linear")
    determine_g(x_bounds, y_bounds, x0, y0, mode="quadratic")

main()