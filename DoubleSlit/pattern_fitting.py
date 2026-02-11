import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pathlib
path = pathlib.Path(__file__).parent.resolve()

sns.set_theme(context='notebook', style="white", palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

# Constants
D = 0.5 #m
RED = 660e-9 #m Red light
SEP = 0.000347

# Smooth X
xs = np.linspace(-0.005,0.005,1000)



# --- Functions --- #

# Load data and convert to SI
def read_data(filepath):

    data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
    # Rescale to m
    data[:,0] = data[:,0] * 0.001
    return data

# Fit func for slit width with
def generate_theory(xs, x_shift, slit_width, wavelength = RED):

    # Constants
    xs = xs - x_shift
    sep = SEP
    distance = D

    alpha = (np.pi * slit_width * xs) / (wavelength * distance)
    beta  = (np.pi * sep * xs) / (wavelength * distance)
    alpha = np.where(alpha==0,1e-10, alpha)
    return (1 * ((np.sin(alpha) * np.cos(beta)) / (alpha))**2)

# Fit func for asymmetric slit intensity
def generate_theory_asym(xs, x_shift, slit_width, A1, A2, wavelength=RED):
    xs = xs - x_shift
    alpha = (np.pi * slit_width * xs) / (wavelength * D)
    beta  = (np.pi * SEP * xs) / (wavelength * D)
    alpha = np.where(alpha==0,1e-10, alpha)
    return ((np.sin(alpha)/alpha)**2) * (A1**2 + A2**2 + 2*A1*A2*np.cos(2*beta))

# Fit func for the wavelength with known geometry.
def generate_theory_wav(xs, wavelength, slit_width = 0.00009):

    # Constants
    sep = SEP
    distance = D
    a = 1 # Normalised data

    alpha = (np.pi * slit_width * xs) / (wavelength * distance)
    beta  = (np.pi * sep * xs) / (wavelength * distance)
    alpha = np.where(alpha==0,1e-10, alpha)
    return (a * ((np.sin(alpha) * np.cos(beta)) / (alpha))**2)

# Fitting for slit width and x-shift
def width_fit(data):
    
    max_pos = data[np.argmax(data[:,1]), 0]
    popt, cov = curve_fit(generate_theory, data[:,0], data[:,1], [max_pos, 0.00009, 660e-9], ) # Measured sp. microscope was not good....
    shift, slit_width, wavelength = popt
    dshift, dw, dlambda = np.sqrt(np.diag(cov))
    print(popt)
    return shift, slit_width, wavelength

# Fitting for wavelength
def find_wavelength(norm_data, width):
    popt, cov = curve_fit(lambda xs, x: generate_theory_wav(xs, x, slit_width=width), norm_data[:,0], norm_data[:,1], [550e-9]) # Rough green sp.
    return popt, np.sqrt(np.diag(cov))

# Asym fit
def asym_fit(data):

    max_pos = data[np.argmax(data[:,1]), 0]
    popt, cov = curve_fit(generate_theory_asym, data[:,0], data[:,1], [max_pos, 0.00009,1,1, 660e-9], ) # Measured sp. microscope was not good....
    shift, slit_width, a1, a2, wavelength = popt
    dshift, dw, da1, da2, dlambda = np.sqrt(np.diag(cov))
    print(dw)
    return shift, slit_width, a1, a2, wavelength



# --- Main --- #
def main():

    # Load RED Laser Data
    laser1_dat = read_data(path / "Session2" / "data_session2.csv")
    laser2_dat = read_data(path / "Session2" / "data_session2.2.csv")
    laser2_singleL_dat = read_data(path / "Session2" / "singleslit1.csv")
    laser2_singleR_dat = read_data(path / "Session2" / "singleslit2.csv")

    # Fit (normalised)
    laser1_dat[:,1] /= np.max(laser1_dat[:,1])
    laser2_dat[:,1] /= np.max(laser2_dat[:,1])

    shift1, width1, wavelength1 = width_fit(laser1_dat)
    shift2, width2, wavelength2 = width_fit(laser2_dat)
    WIDTH = np.average([width1, width2])
    wavelength = np.average([wavelength1, wavelength2])

    # Center data
    laser1_dat[:,0] -= shift1
    laser2_dat[:,0] -= shift2

    # --- Laser Data Plots ---
    # Data only (no fits)
    fig_data, (axd1, axd2) = plt.subplots(1,2, sharey=True, sharex=True, figsize=(16,6))
    sns.lineplot(x=laser1_dat[:,0], y=laser1_dat[:,1], ax=axd1, color="red")
    sns.scatterplot(x=laser1_dat[:,0], y=laser1_dat[:,1], ax=axd1, color="black", s=20, label="Measured Point")
    axd1.set_title("Laser 1 Data (No Fit)")
    axd1.set_ylabel("Normalised Intensity")
    axd1.set_xlabel("Position (m)")
    axd1.legend()

    sns.lineplot(x=laser2_dat[:,0], y=laser2_dat[:,1], ax=axd2, color="red")
    sns.scatterplot(x=laser2_dat[:,0], y=laser2_dat[:,1], ax=axd2, color="black", s=20, label="Measured Point")
    axd2.set_title("Laser 2 Data (No Fit)")
    axd2.set_ylabel("Normalised Intensity")
    axd2.set_xlabel("Position (m)")
    axd2.legend()
    fig_data.tight_layout()

    # --- Laser Fit Plots with Residuals ---
    fig_fit, axs = plt.subplots(2,2, sharex=True, figsize=(16,10), gridspec_kw={'height_ratios': [3, 1]})
    # Fit plots
    sns.scatterplot(x=laser1_dat[:,0], y=laser1_dat[:,1], ax=axs[0,0], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory(xs, 0, WIDTH, wavelength), ax=axs[0,0], color="red", label="Slit Width Fit")
    axs[0,0].set_title("Laser 1 Fit")
    axs[0,0].set_ylabel("Normalised Intensity")
    axs[0,0].legend()
    # Residuals
    fit1 = generate_theory(laser1_dat[:,0], 0, WIDTH, wavelength)
    residuals1 = laser1_dat[:,1] - fit1
    axs[1,0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[1,0].scatter(laser1_dat[:,0], residuals1, color='blue', s=10)
    axs[1,0].set_ylabel("Residuals")
    axs[1,0].set_xlabel("Position (m)")

    sns.scatterplot(x=laser2_dat[:,0], y=laser2_dat[:,1], ax=axs[0,1], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory(xs, 0, WIDTH, wavelength), ax=axs[0,1], color="red", label="Slit Width Fit")
    axs[0,1].set_title("Laser 2 Fit")
    axs[0,1].set_ylabel("Normalised Intensity")
    axs[0,1].legend()
    fit2 = generate_theory(laser2_dat[:,0], 0, WIDTH, wavelength)
    residuals2 = laser2_dat[:,1] - fit2
    axs[1,1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[1,1].scatter(laser2_dat[:,0], residuals2, color='blue', s=10)
    axs[1,1].set_ylabel("Residuals")
    axs[1,1].set_xlabel("Position (m)")
    fig_fit.tight_layout()


    # --- Asym Laser Fit Plots
    fig_fit_asym, axs = plt.subplots(2,2, sharex=True, figsize=(16,10), gridspec_kw={'height_ratios': [3, 1]})

    ashift1, awidth1, a1, a2, awavelength1 = asym_fit(laser1_dat)
    ashift2, awidth2, b1, b2, awavelength2 = asym_fit(laser2_dat)
    print(f"Slit Width: {(awidth1 + awidth2) / 2}")

    # Fit plots
    sns.scatterplot(x=laser1_dat[:,0], y=laser1_dat[:,1], ax=axs[0,0], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory_asym(xs, ashift1, awidth1, a1, a2, awavelength1), ax=axs[0,0], color="red", label="Slit Width Fit")
    axs[0,0].set_title("Laser 1 Fit")
    axs[0,0].set_ylabel("Normalised Intensity")
    axs[0,0].legend()
    # Residuals
    fit1 = generate_theory_asym(laser1_dat[:,0], ashift1, awidth1, a1, a2, awavelength1)
    residuals1 = laser1_dat[:,1] - fit1
    axs[1,0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[1,0].scatter(laser1_dat[:,0], residuals1, color='blue', s=10)
    axs[1,0].set_ylabel("Residuals")
    axs[1,0].set_xlabel("Position (m)")

    # Fit plots
    sns.scatterplot(x=laser2_dat[:,0], y=laser2_dat[:,1], ax=axs[0,1], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory_asym(xs, ashift2, awidth2, b1, b2, awavelength2), ax=axs[0,1], color="red", label="Slit Width Fit")
    axs[0,1].set_title("Laser 2 Fit")
    axs[0,1].set_ylabel("Normalised Intensity")
    axs[0,1].legend()
    # Residuals
    fit2 = generate_theory_asym(laser2_dat[:,0], ashift2, awidth2, b1, b2, awavelength1)
    residuals2 = laser2_dat[:,1] - fit2
    axs[1,1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[1,1].scatter(laser2_dat[:,0], residuals2, color='blue', s=10)
    axs[1,1].set_ylabel("Residuals")
    axs[1,1].set_xlabel("Position (m)")

    fig_fit_asym.tight_layout()


    # --- Low Light using found slit width --- #
    # Load data

    bulb1_dat = read_data(path / "Session3" / "interferencePattern.csv")
    bulb2_dat = read_data(path / "Session4" / "interferencePattern.csv")
    bulb2L_dat = read_data(path / "Session4" / "singleLeft.csv")
    bulb2R_dat = read_data(path / "Session4" / "singleRight.csv")


    # Fit (normalised) order important
    bulb2L_dat[:,1] /= np.max(bulb2_dat[:,1])
    bulb2R_dat[:,1] /= np.max(bulb2_dat[:,1])
    bulb1_dat[:,2] /= np.max(bulb1_dat[:,1])
    bulb1_dat[:,1] /= np.max(bulb1_dat[:,1])
    bulb2_dat[:,2] /= np.max(bulb2_dat[:,1])
    bulb2_dat[:,1] /= np.max(bulb2_dat[:,1])


    # Approx fit to get shift
    shift1, width1, wavelength1 = width_fit(bulb1_dat)
    shift2, width2, wavelength2 = width_fit(bulb2_dat)
    bulb1_dat[:,0] -= shift1
    bulb2_dat[:,0] -= shift2
    bulb2L_dat[:,0] -= shift2
    bulb2R_dat[:,0] -= shift2

    # Fit for wavelength
    green1, dg1 = find_wavelength(bulb1_dat, WIDTH)
    green2, dg2 = find_wavelength(bulb2_dat, WIDTH)
    GREEN = np.average([green1, green2])
    print(f"Green wavelength: {GREEN}")
    print(f"Error: {np.sqrt(dg1**2 + dg2**2)}")

    # --- Bulb Data Plots ---
    fig_bulb_data, (axbd1, axbd2) = plt.subplots(1,2, sharey=True, sharex=True, figsize=(16,6))
    sns.lineplot(x=bulb1_dat[:,0], y=bulb1_dat[:,1], ax=axbd1, color="green")
    sns.scatterplot(x=bulb1_dat[:,0], y=bulb1_dat[:,1], ax=axbd1, color="black", s=20, label="Measured Point")
    axbd1.set_title("Bulb 1 Data (No Fit)")
    axbd1.set_ylabel("Normalised Intensity")
    axbd1.set_xlabel("Position (m)")
    axbd1.legend()

    sns.lineplot(x=bulb2_dat[:,0], y=bulb2_dat[:,1], ax=axbd2, color="green")
    sns.scatterplot(x=bulb2_dat[:,0], y=bulb2_dat[:,1], ax=axbd2, color="black", s=20, label="Measured Point")
    axbd2.set_title("Bulb 2 Data (No Fit)")
    axbd2.set_ylabel("Normalised Intensity")
    axbd2.set_xlabel("Position (m)")
    axbd2.legend()
    fig_bulb_data.tight_layout()

    fig_single, ax = plt.subplots(1,1)
    sns.lineplot(x=bulb2_dat[:,0], y=bulb2_dat[:,1], ax=ax, color="black")
    sns.lineplot(x=bulb2R_dat[:,0], y=bulb2R_dat[:,1], ax=ax, color="red")
    sns.lineplot(x=bulb2L_dat[:,0], y=bulb2L_dat[:,1], ax=ax, color="blue")
    ax.set_title("Single Slits and Double Slit")
    ax.set_ylabel("Normalised Intensity")
    ax.set_xlabel("Position (m)")
    fig_single.tight_layout()

    # --- Bulb Fit Plots with Residuals ---
    fig_bulb_fit, axs_bulb = plt.subplots(2,2, sharex=True, figsize=(16,10), gridspec_kw={'height_ratios': [3, 1]})
    # Bulb 1 fit
    sns.scatterplot(x=bulb1_dat[:,0], y=bulb1_dat[:,1], ax=axs_bulb[0,0], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory(xs, 0, WIDTH, GREEN), ax=axs_bulb[0,0], color="green", label="Wavelength Fit")
    axs_bulb[0,0].set_title("Bulb 1 Fit")
    axs_bulb[0,0].set_ylabel("Normalised Intensity")
    axs_bulb[0,0].legend()
    fitb1 = generate_theory(bulb1_dat[:,0], 0, WIDTH, GREEN)
    residualsb1 = bulb1_dat[:,1] - fitb1
    # Error bars for bulb1 residuals
    axs_bulb[1,0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs_bulb[1,0].errorbar(bulb1_dat[:,0], residualsb1, yerr=bulb1_dat[:,2], fmt='o', color='blue', markersize=4, label='Residuals')
    axs_bulb[1,0].set_ylabel("Residuals")
    axs_bulb[1,0].set_xlabel("Position (m)")
    axs_bulb[1,0].legend()
    # Bulb 2 fit
    sns.scatterplot(x=bulb2_dat[:,0], y=bulb2_dat[:,1], ax=axs_bulb[0,1], color="black", s=20, label="Measured Point")
    sns.lineplot(x=xs, y=generate_theory(xs, 0, WIDTH, GREEN), ax=axs_bulb[0,1], color="green", label="Wavelength Fit")
    axs_bulb[0,1].set_title("Bulb 2 Fit")
    axs_bulb[0,1].set_ylabel("Normalised Intensity")
    axs_bulb[0,1].legend()
    fitb2 = generate_theory(bulb2_dat[:,0], 0, WIDTH, GREEN)
    residualsb2 = bulb2_dat[:,1] - fitb2
    # Error bars for bulb2 residuals
    axs_bulb[1,1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs_bulb[1,1].errorbar(bulb2_dat[:,0], residualsb2, yerr=bulb2_dat[:,2], fmt='o', color='blue', markersize=4, label='Residuals')
    axs_bulb[1,1].set_ylabel("Residuals")
    axs_bulb[1,1].set_xlabel("Position (m)")
    axs_bulb[1,1].legend()
    fig_bulb_fit.tight_layout()

    plt.show()


main()