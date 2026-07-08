from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_method_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_DEVICES,
    _resolve_path,
    _write_csv,
    run_analysis,
)

BUDGETS = (1000, 2000, 5000, 7000, 10000, 15000)
POPULATIONS = (20, 30, 50, 80, 100)


def _parse_list_arg(raw: str) -> tuple[str, ...]:
    raw = (raw or "").strip()
    if not raw:
        return ()
    if "," in raw:
        items = [x.strip() for x in raw.split(",")]
    else:
        items = [x.strip() for x in raw.split()]
    return tuple(x for x in items if x)


def _parse_float(raw: str) -> float:
    if raw == "inf":
        return float("inf")
    return float(raw)


def _collect_from_per_run_csv(path: Path) -> dict[str, float]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    def _mean(key: str) -> float:
        if not rows:
            return 0.0
        vals = [_parse_float(r[key]) for r in rows]
        finite = [v for v in vals if v != float("inf")]
        if not finite:
            return float("inf")
        return sum(finite) / len(finite)

    return {
        "best_accuracy_mean": _mean("best_accuracy"),
        "best_latency_mean": _mean("best_latency"),
        "hv_mean": _mean("hv"),
        "igd_plus_mean": _mean("igd_plus"),
        "spacing_mean": _mean("spacing"),
        "c_metric_mean": _mean("c_metric"),
        "runtime_sec_mean": _mean("runtime_sec"),
        "evaluations_mean": _mean("evaluations"),
    }


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.10g}"


def run_sensitivity(
    csv_path: Path,
    runs: int,
    seed: int | None,
    seed_step: int,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    reference_fronts_csv: Path | None,
    results_root: Path,
) -> None:
    output_root = results_root / "_sensitivity results"
    budget_dir = output_root / "budget_variation"
    pop_dir = output_root / "population_variation"
    budget_dir.mkdir(parents=True, exist_ok=True)
    pop_dir.mkdir(parents=True, exist_ok=True)

    budget_summary_rows: list[list[str]] = []
    for budget in BUDGETS:
        output_name = f"budget_{budget}_pop_50"
        per_run_csv = run_analysis(
            method="mosho",
            csv_path=csv_path,
            runs=runs,
            pop_size=50,
            budget=budget,
            seed=seed,
            seed_step=seed_step,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=reference_fronts_csv,
            results_root=budget_dir,
            output_name=output_name,
        )
        agg = _collect_from_per_run_csv(per_run_csv)
        budget_summary_rows.append(
            [
                str(budget),
                "50",
                _fmt(agg["best_accuracy_mean"]),
                _fmt(agg["best_latency_mean"]),
                _fmt(agg["hv_mean"]),
                _fmt(agg["igd_plus_mean"]),
                _fmt(agg["spacing_mean"]),
                _fmt(agg["c_metric_mean"]),
                _fmt(agg["runtime_sec_mean"]),
                _fmt(agg["evaluations_mean"]),
            ]
        )

    _write_csv(
        budget_dir / "budget_variation_summary.csv",
        [
            "budget",
            "pop_size",
            "best_accuracy_mean",
            "best_latency_mean",
            "hv_mean",
            "igd_plus_mean",
            "spacing_mean",
            "c_metric_mean",
            "runtime_sec_mean",
            "evaluations_mean",
        ],
        budget_summary_rows,
    )

    pop_summary_rows: list[list[str]] = []
    for pop_size in POPULATIONS:
        output_name = f"budget_15000_pop_{pop_size}"
        per_run_csv = run_analysis(
            method="mosho",
            csv_path=csv_path,
            runs=runs,
            pop_size=pop_size,
            budget=15000,
            seed=seed,
            seed_step=seed_step,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=reference_fronts_csv,
            results_root=pop_dir,
            output_name=output_name,
        )
        agg = _collect_from_per_run_csv(per_run_csv)
        pop_summary_rows.append(
            [
                "15000",
                str(pop_size),
                _fmt(agg["best_accuracy_mean"]),
                _fmt(agg["best_latency_mean"]),
                _fmt(agg["hv_mean"]),
                _fmt(agg["igd_plus_mean"]),
                _fmt(agg["spacing_mean"]),
                _fmt(agg["c_metric_mean"]),
                _fmt(agg["runtime_sec_mean"]),
                _fmt(agg["evaluations_mean"]),
            ]
        )

    _write_csv(
        pop_dir / "population_variation_summary.csv",
        [
            "budget",
            "pop_size",
            "best_accuracy_mean",
            "best_latency_mean",
            "hv_mean",
            "igd_plus_mean",
            "spacing_mean",
            "c_metric_mean",
            "runtime_sec_mean",
            "evaluations_mean",
        ],
        pop_summary_rows,
    )

    print("Sensitivity analysis completed.")
    print(f"Output root: {output_root}")
    print(f"Budget variation folder: {budget_dir}")
    print(f"Population variation folder: {pop_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for MOSHO using run_method_analysis.")
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input benchmark CSV.",
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per setting.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help="Seed increment between runs.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Datasets to run (comma or space separated).",
    )
    parser.add_argument(
        "--devices",
        default=",".join(DEFAULT_DEVICES),
        help="Devices to run (comma or space separated).",
    )
    parser.add_argument(
        "--reference-fronts",
        default="",
        help="Optional fixed reference Pareto fronts CSV.",
    )
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root output directory.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    datasets = _parse_list_arg(args.datasets) or DEFAULT_DATASETS
    devices = _parse_list_arg(args.devices) or DEFAULT_DEVICES
    reference = _resolve_path(args.reference_fronts) if (args.reference_fronts or "").strip() else None

    run_sensitivity(
        csv_path=csv_path,
        runs=args.runs,
        seed=args.seed,
        seed_step=args.seed_step,
        datasets=datasets,
        devices=devices,
        reference_fronts_csv=reference,
        results_root=_resolve_path(args.results_root),
    )
