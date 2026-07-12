import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
CSV_FILE = "population_sensitivity_summary.csv"
OUTPUT = "population_sensitivity_combined.png"

# ==========================
# Load data
# ==========================
df = pd.read_csv(CSV_FILE)
df = df.sort_values("population")

metrics = [
    "hv_median",
    "igd_plus_median",
    "spacing_median",
    "runtime_median"
]

labels = {
    "hv_median": "nHV",
    "igd_plus_median": "IGD+",
    "spacing_median": "Spacing",
    "runtime_median": "Runtime"
}

# ==========================
# Normalize to [0,1]
# ==========================
norm = df.copy()

for m in metrics:
    mn = df[m].min()
    mx = df[m].max()

    if mx == mn:
        norm[m] = 1.0
    else:
        norm[m] = (df[m] - mn) / (mx - mn)

# ==========================
# Plot
# ==========================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.2,
    "lines.markersize": 6
})

plt.figure(figsize=(7,5))

for m in metrics:
    plt.plot(
        norm["population"],
        norm[m],
        marker='o',
        label=labels[m]
    )

plt.xlabel("Population")
plt.ylabel("Normalized Metric Value")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=700, bbox_inches="tight")
plt.show()