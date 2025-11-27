import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import pandas as pd
import dataframe_image as dfi

# --- Constants
HC = 1.23984193e-6   #eVm
W_LENGTH = 6.44e-7 #m
N = 1.46 
F = 150 # mm
CON_FAC = 28 / 2048 # Length of detector / num pixels mm/pixel
RNG = np.random.default_rng(seed=2528135)
# ---

def read_data(filepath):

    centre_pos = 0
    uncertainty = 0
    x1 = []
    x2 = []
    b_field = []

    with open(filepath) as f:

        f.readline()
        line = f.readline().split(",")
        centre_pos = float(line[0])
        uncertainty = float(line[0])
        f.readline()

        for line in f:
            line = line.strip().split(",")
            b_field.append(line[0])
            x1.append(line[1])
            x2.append(line[2])

        x1 = np.array(x1, dtype=float)
        x2 = np.array(x2, dtype=float)
        b_field = np.array(b_field,dtype=float)

    return centre_pos, uncertainty, x1, x2, b_field


def calculate_delta_energy(centre_pos, x1, x2, f, table_out=False):

    # Convert Pixels to mm relative to spectrum centre
    pos_1 = (x1 - centre_pos) * CON_FAC
    pos_2 = (x2 - centre_pos) * CON_FAC

    # Convert to angles (radians)
    alpha_1 = np.arctan((pos_1 / f))
    alpha_2 = np.arctan((pos_2 / f))

    # Find beta, intermediate angle
    beta_1 = np.arcsin(np.sin(alpha_1) / N)
    beta_2 = np.arcsin(np.sin(alpha_2) / N)

    # Wave shift %
    waveshift = np.cos(beta_2) / np.cos(beta_1) - 1
    # Energy shift eV
    delta_E   = (-HC * waveshift) / W_LENGTH

    if table_out:

        data_array = np.array(
            [b_field, pos_1,pos_2,alpha_1,alpha_2,
             beta_1,beta_2,waveshift,delta_E]
            ).T
        df = pd.DataFrame(
            data_array, 
            columns=["B (T)", "pos1 (mm)", "pos2 (mm)", "\u03B1 1 (rads)",
                     "\u03B1 2 (rads)", "\u03B2 1 (rads)", "\u03B2 2 (rads)",
                     "Waveshift (%)", "\u0394E (eV)"]
            )
        
        return delta_E, df

    return delta_E


# --- Monte Carlo Error Estimate
def find_error(centre_pos, uncertainty, x1, x2):

    es = np.array([0.5]*len(x1))
    n = 10000 # Sample size
    x1s = RNG.normal(x1[:,np.newaxis], es[:,np.newaxis], size=(len(x1), n))
    x2s = RNG.normal(x2[:,np.newaxis], es[:,np.newaxis], size=(len(x2), n))

    fs = RNG.normal(F, 1, size=n)

    delta_Es = calculate_delta_energy(centre_pos, x1s, x2s, fs)
    #print(delta_Es[:3])
    E_errors = np.std(delta_Es, axis=1)
    return E_errors
    
# --- Weighted Linear Regression
def model(x, m, b):
    return m*x+b
    
def data_fit(x, y, y_err):
    
    popt, pcov = scipy.optimize.curve_fit(model, x, y, sigma=y_err, absolute_sigma=True)
    m, b = popt
    dm, db = np.sqrt(np.diag(pcov))

    return m, b, dm, db



# ---- Main Code and Plotting --- #

fig, (row1, row2, row3) = plt.subplots(3,2,sharex=True,gridspec_kw={"height_ratios": [1, 2, 1]})
fig.set_figwidth(15)
fig.set_figheight(8)
fig.set_tight_layout(True)

axes = [row2[0], row2[0], row2[1], row2[1]]
res_axes = [row1[0],row3[0],row1[1],row3[1]]

# --- Labels and Lims
row2[0].set_ylabel("\u0394E (\u03BCeV)")
row3[0].set_xlabel("Magnetic Field (T)")
row3[1].set_xlabel("Magnetic Field (T)")
row1[0].set_title("Transverse Configuration")
row1[1].set_title("Longitudinal Configuration")
row1[0].set_ylabel("Residual Value (\u03BCeV)")
row3[0].set_ylabel("Residual Value (\u03BCeV)")
row1[0].set_ylim(-4.25,4.25)
row3[0].set_ylim(-4.25,4.25)

row1[1].set_ylabel("Residual Value (\u03BCeV)")
row2[1].set_ylabel("\u0394E (\u03BCeV)")
row3[1].set_ylabel("Residual Value (\u03BCeV)")
row1[1].set_ylim(-4.25,4.25)
row3[1].set_ylim(-4.25,4.25)

# Loop Setup
files = [
    "./CSV_Data/Python_Data/transverse_plus.csv",
    "./CSV_Data/Python_Data/transverse_minus.csv",
    "./CSV_Data/Python_Data/longitudinal_plus.csv",
    "./CSV_Data/Python_Data/longitudinal_minus.csv"
]
colors = [
    "red",
    "blue",
    "red",
    "blue"
]
labels = [
    "\u03c3 +",
    "\u03c3 -",
    "\u03c3 +",
    "\u03c3 -"

]

mu_bs = np.zeros(4)
mu_es = np.zeros(4)

for i, filename in enumerate(files):
    
    # Read Data
    centre_pos, uncertainty, x1, x2, b_field = read_data(filename)

    # Find delta E and generate table
    table_name = str(filename.split("/")[-1])[:-4]
    delta_E, df = calculate_delta_energy(centre_pos, x1, x2, F, table_out=True)

    # Find E errors and add to table
    E_errors = find_error(centre_pos, uncertainty, x1, x2)
    df["\u0394E error"] = E_errors


    
    # Fit Data
    m, b, me, be = data_fit(b_field, delta_E, E_errors)
    # Generate Predicted Values
    y_pred = model(b_field, m, b)
    #Find residuals and propagate error
    residuals = y_pred-delta_E
    residual_error = np.sqrt((b_field*me)**2+E_errors**2 + be**2)

    df["Pred y (eV)"] = y_pred
    df["Residual (eV)"] = residuals
    df["Res Error (eV)"] = residual_error

    # Export Table image
    df_styled = df.style.format({"\u0394E error": "{:.3}", "Residual (eV)":"{:.3}", "Res Error (eV)":"{:.3}"}).hide(axis="index")
    dfi.export(df_styled, f"IMG/Tables/{table_name}.png")


    # Plot residuals
    res_axes[i].axhline(0,color="black",linestyle="--", linewidth="0.5")
    res_axes[i].errorbar(b_field, residuals*1e6, residual_error*1e6, fmt="None", capsize=3, ecolor="black", elinewidth=0.75)
    res_axes[i].plot(b_field, residuals*1e6, ".", color=colors[i])
    # Plot data
    axes[i].plot(b_field, y_pred*1e6, linewidth=0.5, linestyle="--", color=colors[i])
    axes[i].errorbar(b_field, delta_E*1e6, E_errors*1e6, fmt="None", capsize=3, ecolor="black", elinewidth=0.75)
    axes[i].plot(b_field, delta_E*1e6, ".", color=colors[i], label=labels[i])
    axes[i].axhline(0,color="black",linestyle="--", linewidth="0.5")
    # Save Bohr magneton value and error
    mu_bs[i] = np.abs(m)
    mu_es[i] = np.abs(me)

    print(np.abs(m))
    print(np.abs(me))


# Weighted Average
avg = np.sum(mu_bs / mu_es**2) / np.sum(1/mu_es**2)
error = np.sqrt(1 / np.sum(1/mu_es**2))

print(avg)
print(error)

row2[0].legend()
row2[1].legend()
plt.show()
