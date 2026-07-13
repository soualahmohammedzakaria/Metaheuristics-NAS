import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

ROOT = "."      # Folder containing method folders

METHODS = [
    ("mosho", "MOSHO"),
    ("nsga2", "NSGA-II"),
    ("abc", "ABC\n(HiveNAS)"),
    ("firefly", "Firefly"),
    ("pso", "PSO\n(MOPSO)"),
    ("mowso", "MOWSO"),
]

# =====================================================
# Read results
# =====================================================

rows = []

for folder, display in METHODS:

    csvs = glob.glob(
        os.path.join(ROOT, folder, "*_context_metrics.csv")
    )

    if len(csvs) == 0:
        print(f"Skipping {folder}")
        continue

    df = pd.read_csv(csvs[0])

    rows.append({
        "Method": display,
        "nHV": df["hv_median"].median(),
        "IGD+": df["igd_plus_median"].median(),
        "Spacing": df["spacing_median"].median(),
        "Runtime": df["runtime_median"].median()
    })

summary = pd.DataFrame(rows)

# =====================================================
# Plot style
# =====================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

colors = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD"
]

# =====================================================
# Helper function
# =====================================================

def draw_subplot(ax, metric, ylabel, log=False):

    values = summary[metric]

    bars = ax.bar(
        summary["Method"],
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_ylabel(ylabel)

    # Cleaner journal-style appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log:
        ax.set_yscale("log")
    else:
        ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):

        if log:
            ypos = val * 1.08
        else:
            ypos = val + values.max() * 0.02

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:.4f}" if val < 1 else f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
        

# =====================================================
# Figure 1 : nHV + IGD+
# =====================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11,4.5)
)

draw_subplot(
    axes[0],
    "nHV",
    "Median nHV"
)

draw_subplot(
    axes[1],
    "IGD+",
    "Median IGD$^{+}$",
    log=True
)

plt.tight_layout()

plt.savefig(
    "fig_comparative_nhv_igd.png",
    dpi=700,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Figure 2 : Runtime + Spacing
# =====================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11,4.5)
)

draw_subplot(
    axes[0],
    "Runtime",
    "Median Runtime (s)"
)

draw_subplot(
    axes[1],
    "Spacing",
    "Median Spacing"
)

plt.tight_layout()

plt.savefig(
    "fig_comparative_time_spacing.png",
    dpi=700,
    bbox_inches="tight"
)

plt.close()

print("Figures generated successfully.")