from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pareto_scatter(
    points: list[tuple[float, float]],
    title: str,
    output_path: Path,
    reference_points: list[tuple[float, float]] | None = None,
) -> None:
    _ensure_parent(output_path)
    plt.figure(figsize=(7, 5))

    if points:
        xs = [p[1] for p in points]  # latency on x
        ys = [p[0] for p in points]  # accuracy on y
        plt.scatter(xs, ys, s=14, alpha=0.45, label="runs fronts")

    if reference_points:
        rxs = [p[1] for p in reference_points]
        rys = [p[0] for p in reference_points]
        plt.scatter(rxs, rys, s=26, alpha=0.9, label="reference front")

    plt.title(title)
    plt.xlabel("Latency")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_hv_boxplot(
    hv_by_context: dict[int, list[float]],
    output_path: Path,
    title: str = "Hypervolume Distribution by Context",
) -> None:
    _ensure_parent(output_path)
    keys = sorted(hv_by_context.keys())
    data = [hv_by_context[k] for k in keys]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=[str(k) for k in keys], showfliers=False)
    plt.title(title)
    plt.xlabel("Context ID")
    plt.ylabel("Hypervolume")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_runtime_boxplot(
    runtime_by_context: dict[int, list[float]],
    output_path: Path,
    title: str = "Runtime Distribution by Context",
) -> None:
    _ensure_parent(output_path)
    keys = sorted(runtime_by_context.keys())
    data = [runtime_by_context[k] for k in keys]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=[str(k) for k in keys], showfliers=False)
    plt.title(title)
    plt.xlabel("Context ID")
    plt.ylabel("Runtime (s)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_context_metric_heatmap(
    values_by_context: dict[int, float],
    output_path: Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    _ensure_parent(output_path)
    keys = sorted(values_by_context.keys())
    vals = [values_by_context[k] for k in keys]

    plt.figure(figsize=(12, 2.2))
    plt.imshow([vals], aspect="auto", cmap=cmap)
    plt.colorbar(label="value")
    plt.xticks(range(len(keys)), [str(k) for k in keys])
    plt.yticks([0], ["metric"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
