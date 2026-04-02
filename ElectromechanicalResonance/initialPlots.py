# Initial Sweep Plots
# 28/03/26

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pathlib

sns.set_theme(style="ticks")
path = pathlib.Path(__file__).parent.resolve()

df_sweep   = pd.read_csv(path / "CSV" / "initialSweep.csv", delimiter=",")
df_low     = pd.read_csv(path / "CSV" / "125hz.csv", delimiter=",")
df_med     = pd.read_csv(path / "CSV" / "1500hz.csv", delimiter=",")
df_sweep_m = pd.read_csv(path / "CSV" / "initialSweepM.csv", delimiter=",")

fig1, ax1 = plt.subplots(figsize=(9,5))
sns.lineplot(data=df_sweep, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax1)
sns.scatterplot(data=df_sweep, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax1)
ax1.set_ylabel(f"Impedance ($\\Omega$)")
ax1.set_title("Initial Frequency Sampling")

fig2, ax2 = plt.subplots(figsize=(9,5))
sns.lineplot(data=df_low, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax2)
sns.scatterplot(data=df_low, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax2)
ax2.set_ylabel(f"Impedance ($\\Omega$)")
ax2.set_title("Resonance Point")

fig3, ax3 = plt.subplots(figsize=(9,5))
sns.lineplot(data=df_med, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax3)
sns.scatterplot(data=df_med, x="Frequency (Hz)", y="Impedance (Ohm)", color="blue", ax=ax3)
ax3.set_ylabel(f"Impedance ($\\Omega$)")


fig4, ax4 = plt.subplots(figsize=(9,5))
sns.lineplot(data=df_sweep_m, x="Frequency (Hz)", y="Impedance (Ohm)", color="red", ax=ax4)
sns.scatterplot(data=df_sweep_m, x="Frequency (Hz)", y="Impedance (Ohm)", color="red", ax=ax4)
ax4.set_ylabel(f"Impedance ($\\Omega$)")
ax4.set_title("Mass Perturbed Resonance Point")

plt.tight_layout()  
plt.show()