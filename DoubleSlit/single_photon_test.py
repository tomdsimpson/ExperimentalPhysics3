import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pathlib
path = pathlib.Path(__file__).parent.resolve()


# Load data and convert to SI
def read_data(filepath):

    data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
    # Rescale to m
    data[:,0] = data[:,0] * 0.001
    return data

def generate_theory_wav(xs, x_shift, amplitude):

    # Constants
    sep = 0.000347
    distance = 0.5
    wavelength = 551e-9
    slit_width = 5.34e-5
    xs = xs - x_shift

    alpha = (np.pi * slit_width * xs) / (wavelength * distance)
    beta  = (np.pi * sep * xs) / (wavelength * distance)
    alpha = np.where(alpha==0,1e-10, alpha)
    
    return (amplitude * ((np.sin(alpha) * np.cos(beta)) / (alpha))**2)

def fit(data):

    max_val = np.max(data[:,1])
    max_pos = data[np.argmax(data[:,1]),0]
    print(max_pos)
    popt, pcov = curve_fit(generate_theory_wav, data[:,0], data[:,1], [max_pos, max_val])
    x_shift, amplitude = popt
    
    return x_shift, amplitude




data = read_data(path / "Session3" / "interferencePattern.csv")
x_shift, amplitude = fit(data)
centred_data = data
centred_data[:,0] -= x_shift

xs = np.linspace(-0.005, 0.005, 1000)
ys = generate_theory_wav(xs, 0, amplitude)


xs /= 0.00009 # Sample slit width to give dimensionless plot
total_count = np.trapezoid(y=ys, x=xs)
print(total_count)
efficiency = 0.028 # For 551nm light
true_count = (total_count / efficiency) / 10
print(true_count)
