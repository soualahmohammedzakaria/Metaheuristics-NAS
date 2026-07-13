import matplotlib.pyplot as plt

# ==========================================================
# Data
# ==========================================================

methods = ["MOSHO", "NSGA-II", "PSO"]

duration = [0.64, 0.82, 0.71]

energy = [0.00052, 0.00048, 0.00045]

co2 = [0.000311, 0.000306, 0.000280]

colors = ["#0B4F8A", "#4CAF50", "#F39C12"]

# ==========================================================
# Style
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.5, 3.6)
)

# ==========================================================
# Helper
# ==========================================================

def plot_bar(ax, values, title, ylabel, ylim=None, decimals=3):

    bars = ax.bar(
        methods,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.6
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ymax = ax.get_ylim()[1]

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + ymax*0.015,
            f"{val:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

# ==========================================================
# Duration
# ==========================================================

plot_bar(
    axes[0],
    duration,
    "Duration",
    "Time (s)",
    ylim=(0.55, 0.90),
    decimals=2
)

# ==========================================================
# Energy
# ==========================================================

plot_bar(
    axes[1],
    energy,
    "Energy Consumed",
    "Energy (kWh)",
    ylim=(0.0, 0.00065),
    decimals=5
)

# ==========================================================
# CO2
# ==========================================================

plot_bar(
    axes[2],
    co2,
    "CO$_2$ Emissions",
    "kg CO$_2$",
    ylim=(0.0, 0.00035),
    decimals=6
)

plt.tight_layout()

plt.savefig(
    "fig_green_metrics.png",
    dpi=700,
    bbox_inches="tight"
)

plt.show()