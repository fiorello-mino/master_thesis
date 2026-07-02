import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# carica il CSV esportato da ParaView
df = pd.read_csv("contour_phi_05.csv")

# nomi colonne nel tuo file
time_col = "Time"
y_col = "Points:1"   # coordinata verticale della superficie

amp_list = []
for t, group in df.groupby(time_col):
    y = group[y_col].values
    ymax = y.max()
    ymin = y.min()
    A = 0.5 * (ymax - ymin)
    amp_list.append((t, A))

amp_df = pd.DataFrame(amp_list, columns=["Time", "Amplitude"])

# salva su file
amp_df.to_csv("amplitude_vs_time.csv", index=False)

# plot veloce
plt.figure()
plt.plot(amp_df["Time"], amp_df["Amplitude"], "-o")
plt.xlabel("Time")
plt.ylabel("Amplitude (from contour phi=0.5)")
plt.grid(True)
plt.tight_layout()
plt.show()