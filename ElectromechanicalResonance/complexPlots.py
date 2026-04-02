# Tom Simpson
# 12/03/26
# Plotting the locii in the complex plane

# --- Load Packages --- #
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pathlib
import scipy.optimize as scopt
import scipy.interpolate as scint

# --- Setup --- #
RNG = np.random.default_rng()
sns.set_theme(style="ticks")
path = pathlib.Path(__file__).parent.resolve()

# --- Read CSVs --- #
df1  = pd.read_csv(path / "CSV" / "detailedResonance.csv", delimiter=",")
dfm  = pd.read_csv(path / "CSV"  / "small_mass.csv", delimiter=",")


# --- Functions --- #

# Standardise in radians and angular frequency
def rescale_angles_decompose(df):

    # Correct frequency Scaling
    df["Frequency (Hz)"] = df["Frequency (Hz)"] * 2*np.pi

    # Switch to Radianss
    df["Phase (Degrees)"] = df["Phase (Degrees)"].apply(np.radians)
    df["Error (Degrees)"] = df["Error (Degrees)"].apply(np.radians)
    df = df.rename(columns={"Phase (Degrees)":"Phase (Radians)","Error (Degrees)":"Error (Radians)", "Frequency (Hz)":"Frequency (Rads/s)"})
    
    # Impedance Decomposition
    df["Resistance (Ohm)"] = df["Impedance (Ohm)"] * np.abs(np.cos(df["Phase (Radians)"]))
    df["Reactance (Ohm)"] = df["Impedance (Ohm)"] * np.sin(df["Phase (Radians)"])

    return df


# Circle Fit
def fit_scalar(data):

    # Radial residuals function for least_squares
    def circle_residuals(pos, data):
        x, y = pos
        r_xs = data["Resistance (Ohm)"] - x
        r_ys = data["Reactance (Ohm)"] - y
        rs = np.sqrt(r_xs**2 + r_ys**2)
        return rs - np.mean(rs)

    left  = np.min(data["Resistance (Ohm)"])
    right = np.max(data["Resistance (Ohm)"])
    width = right-left
    est_mid = left+width/2 # Initial Guess for minimisation

    result = scopt.least_squares(circle_residuals, [est_mid, 0], args=(data,))
    centre_pos = result.x
    x, y = centre_pos

    # Radius and error
    r_xs = data["Resistance (Ohm)"] - x
    r_ys = data["Reactance (Ohm)"] - y
    rs = np.sqrt(r_xs**2 + r_ys**2)
    radius = np.mean(rs)
    var = np.var(rs)
    dr = (np.sqrt(var) / np.sqrt(len(data)))

    # Construct covariance matrix
    J = result.jac
    n_obs = len(result.fun)
    n_params = len(result.x)
    residual_variance = np.sum(result.fun**2) / (n_obs - n_params)
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * residual_variance
    perr = np.sqrt(np.diag(cov))

    return centre_pos, radius, dr, perr


# Finding the resonance frequency
def find_resonance(df, df_vals):

    # centre, radius, dr, dc
    centre_pos, radius, dr, perr = df_vals
    
    # Smooth data
    spline = scint.CubicSpline(df["Frequency (Rads/s)"], df["Phase (Radians)"], extrapolate=None)

    def calc_frequency(df, centre_pos, radius):
       
       # Find Rightmost Point Angle
        x = centre_pos[0]+radius
        y = centre_pos[1]
        theta_res = np.arctan((y/x))
        
        # Find Roots
        def f(x):
            return spline(x) - theta_res
        res_freq = scopt.brentq(f,80*2*np.pi,140*2*np.pi)
        
        return theta_res, res_freq

    # Setup MC Values
    iterations = 100
    xs = RNG.normal(centre_pos[0], perr[0], size=iterations)
    ys = RNG.normal(centre_pos[1], perr[1], size=iterations)
    radii = RNG.normal(radius, dr, size=iterations)
    centre_positions = np.column_stack((xs,ys))

    resonant_thetas = []
    resonant_frequencies = []

    # Iterate through MC values
    for i in range(iterations):
        result = calc_frequency(df, centre_positions[i], radii[i])
        resonant_thetas.append(result[0])
        resonant_frequencies.append(result[1])

    # Find mean value and error estimation
    theta_res = np.mean(resonant_thetas)
    dtheta = np.sqrt(np.var(resonant_thetas)) 
    res_freq = np.mean(resonant_frequencies)
    dres_freq = np.sqrt(np.var(resonant_frequencies))

    return theta_res, dtheta, res_freq, dres_freq


# Find top and bottom point frequencies
def find_equivalence_frequecies(df, df_vals):

    # Smooth data
    spline = scint.CubicSpline(df["Frequency (Rads/s)"], df["Phase (Radians)"], extrapolate=None)

    def calc_frequency(df, centre_pos, radius, bounds):
        
        # Find top/bottom angle
        x = centre_pos[0]
        y = centre_pos[1] + radius
        theta = np.arctan2(y,x)
                          
        # Find Roots
        def f(x):
            return spline(x) - theta
        freq = scopt.brentq(f,bounds[0], bounds[1]) # Tight Bound as there are two solutions in each case
        
        return freq

    # Setup MC values
    iterations = 100
    xs = RNG.normal(centre_pos[0], perr[0], size=iterations)
    ys = RNG.normal(centre_pos[1], perr[1], size=iterations)
    radii = RNG.normal(radius, dr, size=iterations)
    centre_positions = np.column_stack((xs,ys))

    freq_plus  = []
    freq_minus = []
    bounds_p = [650,760]
    bounds_m = [760,900]

    # Iterate through MC values, top and bottom
    for i in range(iterations):
        freq_plus.append(calc_frequency(df, centre_positions[i], radii[i], bounds_p))
        freq_minus.append(calc_frequency(df, centre_positions[i], -1*radii[i], bounds_m))

    # Final values and error estimate
    freq_p = np.mean(freq_plus)
    dfreq_p = np.sqrt(np.var(freq_plus))
    freq_m = np.mean(freq_minus)
    dfreq_m = np.sqrt(np.var(freq_minus))

    theta_p = np.arctan2(centre_pos[1] + radius,centre_pos[0])
    theta_m = np.arctan2(centre_pos[1] - radius,centre_pos[0])


    return freq_p, dfreq_p, freq_m, dfreq_m, theta_p, theta_m


def mechanical_properties(res_freq, dres_freq, res_freq_m, dres_freq_m):

    # Define measured mass
    m = 0.000231    
    dm = 0.000001
    ms = RNG.normal(m, dm,100)

    def capacitance(res_freq, res_freq_m, m):
        return (1/m * (1/res_freq_m**2 - 1/res_freq**2))

    # Setup MC values
    res_freqs  = RNG.normal(res_freq, dres_freq, 100)
    res_freqs_m = RNG.normal(res_freq_m, dres_freq_m, 100)
    capacitances = capacitance(res_freqs, res_freqs_m, ms)

    # Final capactiance value and error
    c_m  = np.mean(capacitances)
    dc_m = np.sqrt(np.var(capacitances))


    def inductance(res_freq, c_m):
        return (1 / (c_m * res_freq**2))

    # Setup MC values
    res_freqs  = RNG.normal(res_freq, dres_freq, 100)
    c_ms = RNG.normal(c_m, dc_m, 100)
    inductances = inductance(res_freqs, c_ms)

    # Final inductance and error
    l_m = np.mean(inductances)
    dl_m = np.sqrt(np.var(inductances))

    return c_m, dc_m, l_m, dl_m

def mechanical_properties_two(c_m, dc_m, l_m, dl_m, freq_p, dfreq_p, freq_m, dfreq_m, radius, dr):

    def mechanical_resistance(c_m, l_m, freq):
        return (1 / (c_m * freq) - freq*l_m) 

    # Setup MC values 
    c_ms = RNG.normal(c_m,dc_m,100)
    l_ms = RNG.normal(l_m,dl_m,100)
    freq_ps = RNG.normal(freq_p, dfreq_p, 100)
    freq_ms = RNG.normal(freq_m, dfreq_m, 100)

    mres_ps = mechanical_resistance(c_ms, l_ms, freq_ps)
    mres_ms = mechanical_resistance(c_ms, l_ms, freq_ms)
    
    # Value for both points, then average
    mres_p = np.mean(mres_ps)
    dmres_p = np.sqrt(np.var(mres_ps))
    mres_m = -1 * np.mean(mres_ms)
    dmres_m = np.sqrt(np.var(mres_ms))
    mres = np.mean([mres_p, mres_m])
    dmres = np.mean([dmres_p, dmres_m])

    def transduction_coefficient(mres, radius):
        return np.sqrt(2* radius * mres)

    # Setup MC values
    rs = RNG.normal(radius, dr, 100)
    mress = RNG.normal(mres, dmres, 100)

    transduction_coefs = transduction_coefficient(mress, rs)

    # Final transduction value
    t = np.mean(transduction_coefs) 
    dt = np.sqrt(np.var(transduction_coefs))

    return mres, dmres, t, dt

# Calculate radial residuals for plotting
def calc_residuals(df, centre_pos, radius):

    x, y = centre_pos
    r_xs = df["Resistance (Ohm)"] - x
    r_ys = df["Reactance (Ohm)"] - y
    rs = np.sqrt(r_xs**2 + r_ys**2)
    residuals = rs - radius
    return residuals

# Generate smooth spline data for plot
def gen_spline_plot(df):
    
    x_0 = np.min(df["Frequency (Rads/s)"])
    x_f = np.max(df["Frequency (Rads/s)"])
    spline = scint.CubicSpline(df["Frequency (Rads/s)"], df["Phase (Radians)"], extrapolate=None)
    xs = np.linspace(x_0,x_f,1000)
    ys = spline(xs)

    return xs, ys

# Process read CSVs
df1 = rescale_angles_decompose(df1)
dfm = rescale_angles_decompose(dfm)

# Fit processed datasets
df1_vals = fit_scalar(df1)
dfm_vals = fit_scalar(dfm)
centre_pos, radius, dr, perr = df1_vals
centre_pos_m, radius_m, dr_m, perr_m = dfm_vals

# Find resonances
theta_res, dtheta, res_freq, dres_freq = find_resonance(df1, df1_vals)
theta_res_m, dtheta_m, res_freq_m, dres_freq_m = find_resonance(dfm, dfm_vals)

# Find mechanical properties
c_m, dc_m, l_m, dl_m = mechanical_properties(res_freq, dres_freq, res_freq_m, dres_freq_m)
freq_p, dfreq_p, freq_m, dfreq_m, theta_p, theta_m = find_equivalence_frequecies(df1, df1_vals)
mres, dmres, t, dt = mechanical_properties_two(c_m, dc_m, l_m, dl_m, freq_p, dfreq_p, freq_m, dfreq_m, radius, dr)

# Output all key values
print("Initial Setup")
print(f"Radius: {radius}  |  {dr}")
print(f"Centre Pos: {centre_pos}  |  {perr}")
print(f"Resonant Phase: {theta_res}  |  {dtheta}")
print(f"Resonant Frequency: {res_freq}  |  {dres_freq}")
print("")
print("---")
print("")
print("Mass Displaced Setup")
print(f"Radius: {radius_m}  |  {dr_m}")
print(f"Centre Pos: {centre_pos_m}  |  {perr_m}")
print(f"Resonant Phase: {theta_res_m}  |  {dtheta_m}")
print(f"Resonant Frequency: {res_freq_m}  |  {dres_freq_m}")
print("")
print("---")
print("")
print(f"Capacitance: {c_m}  |  {dc_m}")
print(f"Inductance: {l_m}  |  {dl_m}")
print("")
print("---")
print("")
print(f"Positive Equi Freq: {freq_p}  |  {dfreq_p}")
print(f"Negative Equi Freq: {freq_m}  |  {dfreq_m}")
print(f"Mechanical Resistance: {mres}  | {dmres}")
print(f"Transduction Coef: {t}  | {dt}")

# Setup Plots
fig_1, (r1,r2)   =    plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios':[2,1]})
fig_m, (rm1,rm2) =    plt.subplots(2, 1, figsize=(6,8), gridspec_kw={'height_ratios':[2,1]})
phase_1, rp      =    plt.subplots(1, 1, figsize=(6,6))
phase_2, rp2     =    plt.subplots(1, 1, figsize=(6,6))

# Standard circle fit + residuals
sns.scatterplot(df1, x=f"Resistance (Ohm)", y=f"Reactance (Ohm)", ax=r1, color="blue")
r1.scatter(centre_pos[0],centre_pos[1], color="blue", marker="x", label="Centre Point")
circ = plt.Circle(centre_pos, radius, fill=False, color="blue", alpha=0.8, linestyle="--", label="Circle Fit")
r1.add_artist(circ)
r1.legend()

res   = calc_residuals(df1, centre_pos, radius)
sns.scatterplot(df1, x="Frequency (Rads/s)", y=res, ax=r2, color="blue", s=15)


# Mass perturbed circle fit + residuals
sns.scatterplot(dfm, x=f"Resistance (Ohm)", y=f"Reactance (Ohm)", ax=rm1, color="red")
rm1.scatter(centre_pos_m[0],centre_pos_m[1], color="red", marker="x", label="Centre Point")
circ = plt.Circle(centre_pos_m, radius_m, fill=False, color="Red", alpha=0.8, linestyle="--", label="Circle Fit")
rm1.add_artist(circ)
rm1.legend()

res_m = calc_residuals(dfm, centre_pos_m, radius_m)
sns.scatterplot(dfm, x="Frequency (Rads/s)", y=res_m, ax=rm2, color="red",s=15)


# Scales and Labels
r2.hlines(0,0,1600, linestyles="--", colors="black", linewidth=1)
r2.set_xlim(0,1600)
r2.set_ylabel(f"Residual ($\Omega$)")
r2.set_xlabel(r"Frequency ($rads^{-1}$)")

rm2.hlines(0,0,1750, linestyles="--", colors="black", linewidth=1)
rm2.set_xlim(0,1750)
rm2.set_ylabel(f"Residual ($\Omega$)")
rm2.set_xlabel(f"Frequency ($rads^{{-1}}$)")

r1.set_title("Loudspeaker Impedance")
r1.hlines(0, 0, 45, linestyles="--", colors="black", linewidth=1)
r1.set_aspect('equal')
r1.set_xlim((0,45))
r1.set_xlabel(f"Resistance ($\Omega$)")
r1.set_ylabel(f"Reactance ($\Omega$)")

rm1.set_title("Loudspeaker Impedance, with Added Mass ")
rm1.hlines(0, 0, 35, linestyles="--", colors="black", linewidth=1)
rm1.set_aspect('equal')
rm1.set_xlim((0,35))
rm1.set_xlabel(f"Resistance ($\Omega$)")
rm1.set_ylabel(f"Reactance ($\Omega$)")


# Frequency Phase Plots
# Smooth interpolation
x1s,y1s = gen_spline_plot(df1)
xms,yms = gen_spline_plot(dfm)

sns.lineplot(x=x1s,y=y1s,ax=rp,color="blue", linestyle="--", label="Cubic Interpolation")
sns.scatterplot(df1, x="Frequency (Rads/s)",y="Phase (Radians)",ax=rp, color="blue")
rp.hlines(theta_res, 0, np.max(df1["Frequency (Rads/s)"]), colors="blue", linewidth=1)
rp.vlines(res_freq, -1,1, colors="blue", linewidth=1)
rp.annotate(f"$\omega_r$", (660,-0.95), fontsize=16)
rp.annotate(f'$\\theta_r$', (50,-0.13), fontsize=16)
rp.set_ylabel(f"Phase ($rad$)")
rp.set_xlabel(f"Angular Frequency ($rads^{{-1}}$)")
rp.set_xlim(0, np.max(df1["Frequency (Rads/s)"]))
rp.set_ylim(-1,1)
rp.set_title("Phase-Frequency Relation")
rp.legend()

sns.lineplot(x=xms,y=yms,ax=rp2,color="red", linestyle="--", label="Cubic Interpolation")
sns.scatterplot(dfm, x="Frequency (Rads/s)",y="Phase (Radians)",ax=rp2, color="red")
rp2.hlines(theta_res_m, 0, np.max(df1["Frequency (Rads/s)"]), colors="red", linewidth=1)
rp2.vlines(res_freq_m, -1,1, colors="red", linewidth=1)
rp2.annotate(f"$\omega_r'$", (600,-0.95), fontsize=16)
rp2.annotate(f"$\\theta_r'$", (50,-0.13), fontsize=16)
rp2.set_xlim(0, np.max(dfm["Frequency (Rads/s)"]))
rp2.set_ylim(-1,1)
rp2.set_ylabel(f"Phase ($rad$)")
rp2.set_xlabel(f"Angular Frequency ($rads^{{-1}}$)")
rp2.legend()
rp2.set_title("Phase-Frequency Relation, with Added Mass")

# Equivalence Point Plot
fig_eq, ax_eq = plt.subplots(figsize=(9,5))
sns.lineplot(x=x1s,y=y1s,ax=ax_eq,color="blue", linestyle="--", label="Cubic Interpolation")
sns.scatterplot(df1, x="Frequency (Rads/s)",y="Phase (Radians)",ax=ax_eq, color="blue")

ax_eq.hlines([theta_p, theta_m], 0, np.max(df1["Frequency (Rads/s)"]), colors=["blue","red"], linewidth=1)
ax_eq.vlines([freq_p, freq_m], -1,1, colors=["blue","red"], linewidth=1)

ax_eq.annotate(f"$\\theta_{{top}}$", (20, 0.4), fontsize=16)
ax_eq.annotate(f"$\\theta_{{bottom}}$", (20, -0.72), fontsize=16)
ax_eq.annotate(f"$\\omega_{{top}}$", (710, -0.9), fontsize=16)
ax_eq.annotate(f"$\\omega_{{bottom}}$", (850, -0.9), fontsize=16)

ax_eq.set_ylabel(f"Phase ($rad$)")
ax_eq.set_xlabel(f"Angular Frequency ($rads^{{-1}}$)")
ax_eq.set_xlim(0, np.max(df1["Frequency (Rads/s)"]))
ax_eq.set_ylim(-1,1)
ax_eq.set_title("Mechanical Impedance - Real and Imaginary Equivalence Points")
ax_eq.legend()

plt.tight_layout(pad=1.5)
plt.show()