from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results_final"
SUMMARY_CSV = RESULTS_ROOT / "ablation_suite_summary.csv"
OUTPUT_DIR = RESULTS_ROOT / "ablation_statistics"
CONTROL_VARIANT = "mosho_enhanced"


METRICS = [
    {
        "key": "hv_median",
        "series": "hv",
        "label": "nHV",
        "higher_is_better": True,
        "filename": "ablation_hv.png",
    },
    {
        "key": "igd_plus_median",
        "series": "igd_plus",
        "label": "IGD$^+$",
        "higher_is_better": False,
        "filename": "ablation_igd_plus.png",
    },
    {
        "key": "spacing_median",
        "series": "spacing",
        "label": "Spacing",
        "higher_is_better": False,
        "filename": "ablation_spacing.png",
    },
    {
        "key": "runtime_sec_median",
        "series": "runtime_sec",
        "label": "Runtime (s)",
        "higher_is_better": False,
        "filename": "ablation_runtime.png",
    },
]


def _load_per_run_series(variant: str, base_method: str, series: str) -> pd.Series:
    csv_path = RESULTS_ROOT / variant / f"{base_method}_metrics_by_run.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"run_id", series}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    return df.groupby("run_id", sort=True)[series].mean().sort_index()


def _align_runs(control: pd.Series, variant: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    common = control.index.intersection(variant.index)
    if common.empty:
        raise ValueError("No overlapping run_id values found for paired test")
    return control.loc[common].to_numpy(dtype=float), variant.loc[common].to_numpy(dtype=float)


def _oriented(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    return values if higher_is_better else -values


def _wilcoxon_pvalue(control: np.ndarray, variant: np.ndarray) -> float:
    try:
        return float(wilcoxon(control, variant, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _a12(control: np.ndarray, variant: np.ndarray) -> float:
    if control.size == 0 or variant.size == 0:
        return float("nan")
    greater = 0.0
    for value in control:
        greater += np.sum(value > variant)
        greater += 0.5 * np.sum(value == variant)
    return float(greater / (control.size * variant.size))


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def _plot_metric(summary: pd.DataFrame, metric: dict[str, object]) -> None:
    ordered = summary.sort_values(metric["key"], ascending=not metric["higher_is_better"])
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(ordered["tex_label"], ordered[metric["key"]], color="#2a9d8f" if metric["higher_is_better"] else "#e76f51")
    ax.set_title(f"{metric['label']} across all tested ablation variants")
    ax.set_ylabel(metric["label"])
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ordered[metric["key"]].to_list()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / str(metric["filename"]), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_a12(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        ordered = summary[summary["metric_key"] == metric["key"]].sort_values("a12_oriented", ascending=False)
        colors = ["#2a9d8f" if value >= 0.5 else "#264653" for value in ordered["a12_oriented"]]
        ax.bar(ordered["tex_label"], ordered["a12_oriented"], color=colors)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Oriented A12 for {metric['label']}")
        ax.set_ylabel("A12")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Vargha-Delaney A12 effect sizes against MOSHO", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "ablation_a12_effect.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    required_summary = {"variant", "tex_label", "hv_median", "igd_plus_median", "spacing_median", "runtime_sec_median"}
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"{SUMMARY_CSV} missing columns: {sorted(missing)}")

    control_summary = summary.loc[summary["variant"] == CONTROL_VARIANT]
    if control_summary.empty:
        raise ValueError(f"Control variant {CONTROL_VARIANT!r} not found in summary CSV")

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        control_base_method = str(control_summary.iloc[0]["base_method"])
        control_series = _load_per_run_series(CONTROL_VARIANT, control_base_method, str(metric["series"]))
        metric_rows: list[dict[str, object]] = []
        p_values: list[float] = []

        for _, variant_row in summary.iterrows():
            variant = str(variant_row["variant"])
            if variant == CONTROL_VARIANT:
                continue

            variant_series = _load_per_run_series(variant, str(variant_row["base_method"]), str(metric["series"]))
            control_values, variant_values = _align_runs(control_series, variant_series)
            oriented_control = _oriented(control_values, bool(metric["higher_is_better"]))
            oriented_variant = _oriented(variant_values, bool(metric["higher_is_better"]))

            p_value = _wilcoxon_pvalue(oriented_control, oriented_variant)
            p_values.append(p_value)
            metric_rows.append(
                {
                    "metric_key": metric["key"],
                    "metric_label": metric["label"],
                    "variant": variant,
                    "tex_label": str(variant_row["tex_label"]),
                    "p_value": p_value,
                    "a12_oriented": _a12(oriented_control, oriented_variant),
                    "control_median": float(np.median(oriented_control)),
                    "variant_median": float(np.median(oriented_variant)),
                }
            )

        adjusted = _holm_bonferroni(p_values)
        for row, p_holm in zip(metric_rows, adjusted):
            row["p_holm"] = p_holm
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(["metric_key", "p_holm", "tex_label"])
    result.to_csv(OUTPUT_DIR / "ablation_statistics_summary.csv", index=False)

    for metric in METRICS:
        _plot_metric(summary, metric)
    _plot_a12(result)

    print(result.to_string(index=False))
    print(f"Wrote plots and statistics to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())