import matplotlib.pyplot as plt

# ==========================================
# Publication style
# ==========================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

# ==========================================
# Updated values
# ==========================================

metrics = {
    "nHV": {
        "values": [1, 0.9987, 0.9969],   # Best, Median, Worst
        "ylabel": "nHV ↑",
        "ylim": (0.995, 1.0005),
        "fmt": "{:.4f}"
    },
    "IGD$^{+}$": {
        "values": [0.0038, 0.0064, 0.0067],
        "ylabel": "IGD$^{+}$ ↓",
        "ylim": (0.003, 0.013),
        "fmt": "{:.4f}"
    },
    "Spacing": {
        "values": [2.1, 2.74, 3.55],
        "ylabel": "Spacing ↓",
        "ylim": (2.0, 3.8),
        "fmt": "{:.2f}"
    }
}

colors = ["#2ca25f", "#3b6fb6", "#c0392b"]
labels = ["Best", "Median", "Worst"]

# ==========================================
# Plot
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

for ax, (title, info) in zip(axes, metrics.items()):

    bars = ax.bar(
        labels,
        info["values"],
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title(title)
    ax.set_ylabel(info["ylabel"])
    ax.set_ylim(info["ylim"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    rng = info["ylim"][1] - info["ylim"][0]

    for b, v in zip(bars, info["values"]):
        ax.text(
            b.get_x() + b.get_width()/2,
            v + rng*0.02,
            info["fmt"].format(v),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

plt.tight_layout()
plt.savefig("fig_performance_assessment.png", dpi=700, bbox_inches="tight")
plt.show()