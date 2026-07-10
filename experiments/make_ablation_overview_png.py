from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results_final" / "ablation_suite_summary.csv"
OUTPUT_PNG = ROOT / "results_final" / "ablation_statistics" / "ablation_overview_all_variants.png"


TABLE_ORDER = [
    "mosho_enhanced",
    "abl_u01",
    "abl_u02",
    "abl_u03",
    "abl_u04",
    "abl_u05",
    "abl_u06",
    "abl_u07",
    "abl_u08",
    "abl_u10",
    "abl_u11",
    "g_search",
    "g_adapt",
    "g_archive",
    "g_noadv",
    "g_nobase",
    "g_core",
]


DISPLAY_LABELS = {
    "mosho_enhanced": "MOSHO",
    "abl_u01": "U01",
    "abl_u02": "U02",
    "abl_u03": "U03",
    "abl_u04": "U04",
    "abl_u05": "U05",
    "abl_u06": "U06",
    "abl_u07": "U07",
    "abl_u08": "U08",
    "abl_u10": "U09",
    "abl_u11": "U10",
    "g_search": "G-SEARCH",
    "g_adapt": "G-ADAPT",
    "g_archive": "G-ARCHIVE",
    "g_noadv": "G-NOADV",
    "g_nobase": "G-NOBASE",
    "g_core": "G-CORE",
}


METRICS = [
    ("hv_median", "nHV", "nHV ↑", True, 3),
    ("igd_plus_median", "IGD+", "IGD+ ↓", False, 3),
    ("spacing_median", "Spacing", "Spacing ↓", False, 2),
    ("runtime_sec_median", "Runtime", "Runtime (s) ↓", False, 2),
]


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    frame = pd.read_csv(SUMMARY_CSV)
    required = {"variant", "tex_label", "hv_median", "igd_plus_median", "spacing_median", "runtime_sec_median"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{SUMMARY_CSV} missing columns: {sorted(missing)}")

    frame = frame[frame["variant"].isin(TABLE_ORDER)].copy()
    frame["variant"] = pd.Categorical(frame["variant"], categories=TABLE_ORDER, ordered=True)
    frame = frame.sort_values("variant")
    frame["display_label"] = frame["variant"].map(DISPLAY_LABELS)
    return frame


def _build_color_map() -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors)
    return {label: palette[index] for index, label in enumerate(frame_labels())}


def frame_labels() -> list[str]:
    return [DISPLAY_LABELS[variant] for variant in TABLE_ORDER]


def _plot_metric(ax, frame: pd.DataFrame, metric_key: str, title: str, y_label: str, higher_is_better: bool, decimals: int, color_map: dict[str, str]) -> None:
    ordered = frame.sort_values(metric_key, ascending=not higher_is_better)
    colors = [color_map[label] for label in ordered["display_label"]]
    bars = ax.bar(ordered["display_label"], ordered[metric_key], color=colors, width=0.68, edgecolor="#2f2f2f", linewidth=0.4)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params(axis="x", rotation=32, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, ordered[metric_key].to_list()):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#4d5a73",
            rotation=0,
            clip_on=False,
        )


def main() -> int:
    frame = _load_summary()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    rcParams.update({
        "font.family": ["Georgia", "Times New Roman", "DejaVu Serif", "serif"],
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(24, 13), constrained_layout=True)
    color_map = _build_color_map()

    for ax, (metric_key, title, y_label, higher_is_better, decimals) in zip(axes.flat, METRICS):
        _plot_metric(ax, frame, metric_key, title, y_label, higher_is_better, decimals, color_map)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote: {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())