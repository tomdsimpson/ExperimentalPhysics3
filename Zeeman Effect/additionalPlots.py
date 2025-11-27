import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {}
sns.set_theme(style="ticks", rc=custom_params)



filename = "./CSV_Data/Plot_Data/transverse2.csv"
df  = pd.read_csv(filename)
df.columns = [
    "B (T)",
    "\u03B1 1 (rads)",
    "\u03B1 2 (rads)",
    "Waveshift (%)",
    "\u0394E (\u03BCeV)",
    "Pred y (\u03BCeV)",
    "Residual (\u03BCeV)",
    "Res Error (\u03BCeV)"
]

split_idx = df.index[df.isna().all(axis=1)][0]
df_plus = df.iloc[:split_idx]
df_minus = df.iloc[split_idx+1:]

fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
    3, 1, figsize=(8, 10),
    sharex=True,                      # <-- shared x-axis
    gridspec_kw={"height_ratios": [1, 2, 1],
                 "hspace": 0.2})

# Top Res
sns.scatterplot(data=df_plus, x="B (T)", y="Residual (\u03BCeV)", ax=ax_top,
                color="red")
ax_top.errorbar(df_plus["B (T)"], df_plus["Residual (\u03BCeV)"], df_plus["Res Error (\u03BCeV)"], fmt="None", capsize=3, ecolor="black", elinewidth=0.75)
ax_top.axhline(0, color="black", linewidth=0.5)
ax_top.set_ylabel("Residual (\u03BCeV)")
ax_top.set_ylim(-5,5)
fig.suptitle("Transverse Configuration")


sns.scatterplot(data=df_plus, x="B (T)", y="\u0394E (\u03BCeV)", color="red", 
                ax=ax_mid, label="\u03c3 +" )
sns.lineplot(data=df_plus, x="B (T)", y="Pred y (\u03BCeV)", color="red", linewidth=0.75, 
             linestyle="--", ax=ax_mid)
sns.scatterplot(data=df_minus, x="B (T)", y="\u0394E (\u03BCeV)", color="blue",
                ax=ax_mid, label="\u03c3 -")
sns.lineplot(data=df_minus, x="B (T)", y="Pred y (\u03BCeV)", linewidth="0.75",
             linestyle="--", ax=ax_mid)
ax_mid.axhline(0, color="black", linewidth=0.5)
ax_mid.set_ylabel("\u0394E (\u03BCeV)")
ax_mid.legend()


sns.scatterplot(data=df_minus, x="B (T)", y="Residual (\u03BCeV)", ax=ax_bottom,
                color="blue")
ax_bottom.errorbar(df_minus["B (T)"], df_minus["Residual (\u03BCeV)"], df_minus["Res Error (\u03BCeV)"], fmt="None", capsize=3, ecolor="black", elinewidth=0.75)
ax_bottom.axhline(0, color="black", linewidth=0.5)
ax_bottom.set_xlabel("Magnetic Field Strength (T)")
ax_bottom.set_ylabel("Residual (\u03BCeV)")

ax_bottom.set_ylim(-5,5)

# Export Table image
df_styled = df_minus.style.hide(axis="index")
dfi.export(df_styled, f"IMG/Tables/VCom_transverse_minus2.png")
df_styled = df_plus.style.hide(axis="index")
dfi.export(df_styled, f"IMG/Tables/VCom_transverse_plus2.png")

plt.show()
