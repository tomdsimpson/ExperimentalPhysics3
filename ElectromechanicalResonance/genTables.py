# Appendix Tables

import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

path = pathlib.Path(__file__).parent.resolve()


# ---
def decompose(df):

    # Correct frequency Scaling
    #df["Frequency (Hz)"] = df["Frequency (Hz)"] * 2*np.pi

    # Switch to Radianss
    df["Phase (Degrees)"] = df["Phase (Degrees)"].apply(np.radians)
    df["Error (Degrees)"] = df["Error (Degrees)"].apply(np.radians)
    df = df.rename(columns={"Phase (Degrees)":"Phase (rad)","Error (Degrees)":"Error (rad)", "Impedance (Ohm)":"Impedance (\u2126)"})
    
    # Electrical Inductance Correction
    df["Resistance (\u2126)"] = df["Impedance (\u2126)"] * np.abs(np.cos(df["Phase (rad)"]))
    df["Reactance (\u2126)"] = df["Impedance (\u2126)"] * np.sin(df["Phase (rad)"])
    df = df.rename(columns={})

    return df


# Read CSVs
potential_dat_2 = pd.read_csv(path / "CSV/potential_2.csv")
potential_dat_2 = potential_dat_2.rename(columns={"Impedance (Ohm)" : "Impedace (\u2126)"})
raw_dat_1       = pd.read_csv(path / "CSV/detailedResonance.csv")
raw_dat_1       = decompose(raw_dat_1)
raw_dat_m       = pd.read_csv(path / "CSV/small_mass.csv")
raw_dat_m       = decompose(raw_dat_m)

sweep_dat       = pd.read_csv(path / "CSV/initialSweep.csv")
potential_dat_1 = pd.read_csv(path / "CSV/potential_1.csv")


styled = potential_dat_2.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}])
styled = styled.hide(axis="index")
styled = styled.set_properties(**{'text-align': 'center'})  # Center all cells
styled = styled.format("{:.4f}", subset=potential_dat_2.columns[[1]])       # Col 1
styled = styled.format("{:.4f}", subset=potential_dat_2.columns[[2]])       # Col 2
styled = styled.format("{:.4f}", subset=potential_dat_2.columns[[3]])       # Col 3
styled = styled.format("{:.3}", subset=potential_dat_2.columns[[4]])        # Col 4+
dfi.export(styled, path / "Tables/potential_res_2.png")

styled = raw_dat_1.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}])
styled = styled.hide(axis="index")
styled = styled.set_properties(**{'text-align': 'center'})  # Center all cells
styled = styled.format("{:.4f}", subset=raw_dat_1.columns[[1]])       # Col 1
styled = styled.format("{:.4f}", subset=raw_dat_1.columns[[2]])       # Col 2
styled = styled.format("{:.4f}", subset=raw_dat_1.columns[[3]])       # Col 3
styled = styled.format("{:.3}", subset=raw_dat_1.columns[[4]])        # Col 4
styled = styled.format("{:.3f}", subset=raw_dat_1.columns[[5]])        # Col 5
styled = styled.format("{:.4f}", subset=raw_dat_1.columns[[6]])        # Col 6
dfi.export(styled, path / "Tables/detailedResonance.png")

styled = raw_dat_m.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}])
styled = styled.hide(axis="index")
styled = styled.set_properties(**{'text-align': 'center'})  # Center all cells
styled = styled.format("{:.4f}", subset=raw_dat_m.columns[[1]])       # Col 1
styled = styled.format("{:.4f}", subset=raw_dat_m.columns[[2]])       # Col 2
styled = styled.format("{:.4f}", subset=raw_dat_m.columns[[3]])       # Col 3
styled = styled.format("{:.3}", subset=raw_dat_m.columns[[4]])        # Col 4
styled = styled.format("{:.3f}", subset=raw_dat_m.columns[[5]])        # Col 5
styled = styled.format("{:.4f}", subset=raw_dat_m.columns[[6]])        # Col 6
dfi.export(styled, path / "Tables/massResonance.png")

# ---
styled = sweep_dat.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}])
styled = styled.hide(axis="index")
styled = styled.set_properties(**{'text-align': 'center'})  # Center all cells
styled = styled.format("{:.4f}", subset=sweep_dat.columns[[1]])       # Col 1
styled = styled.format("{:.4f}", subset=sweep_dat.columns[[2]])       # Col 2
styled = styled.format("{:.4f}", subset=sweep_dat.columns[[3]])       # Col 3
styled = styled.format("{:.3}", subset=sweep_dat.columns[[4]])        # Col 4+
dfi.export(styled, path / "Tables/sweep.png")

styled = potential_dat_1.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}])
styled = styled.hide(axis="index")
styled = styled.set_properties(**{'text-align': 'center'})  # Center all cells
styled = styled.format("{:.4f}", subset=potential_dat_1.columns[[1]])       # Col 1
styled = styled.format("{:.4f}", subset=potential_dat_1.columns[[2]])       # Col 2
styled = styled.format("{:.4f}", subset=potential_dat_1.columns[[3]])       # Col 3
styled = styled.format("{:.3}", subset=potential_dat_1.columns[[4]])        # Col 4+
dfi.export(styled, path / "Tables/potential_res_1.png")


sns.scatterplot(data=potential_dat_2, x="Frequency (Hz)", y="Impedace (\u2126)", color="blue")
sns.lineplot(data=potential_dat_2, x="Frequency (Hz)", y="Impedace (\u2126)", color="blue")
plt.title("Potential Resonance Point")
plt.show()