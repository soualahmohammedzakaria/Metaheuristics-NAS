from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

# Ensure imports work when running: python experiments/run_method_analysis.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import SinglePointMutation, GaussianMutation, ABCNeighborSampler
from nas_framework.population import Individual, Population, PSOPopulation, ABCPopulation
from nas_framework.replacement import ElitistReplacement, CrowdingReplacement, RankBasedReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import ABCFireflyStrategy,BruteForceParetoSearch, RandomSearch, SkylineSearch, MOWSOSearch, MOSHOSearch, PSOSearchStrategy, ABCSearchStrategy, FireflySearchStrategy, HybridMBOStrategy,  APSOESearch, NSGA2SearchStrategy
from nas_framework.selection import TournamentSelection, RouletteWheelSelection
from utilities.metrics import c_metric, igd_plus, non_dominated, normalized_hypervolume_to_reference_2d
from utilities.plotting import (
    save_context_metric_heatmap,
    save_hv_boxplot,
    save_pareto_scatter,
    save_runtime_boxplot,
)

DEFAULT_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
DEFAULT_DEVICES = ("edgegpu", "edgetpu", "eyeriss", "fpga", "pixel3", "raspi4")
OBJECTIVE_DIRECTIONS = (1, -1)


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    alt = ROOT / path_like
    if alt.exists():
        return alt
    return p


def _extract_front(strategy: Any, result: Any) -> list[Individual]:
    if isinstance(result, Population):
        return result.pareto_front()
    if isinstance(result, list) and (not result or isinstance(result[0], Individual)):
        return result

    history = getattr(strategy, "history", None)
    if history and getattr(history, "pareto_archive", None):
        archive = history.pareto_archive
        if archive:
            return archive[-1]

    raise TypeError(
        "Unable to get Pareto front from strategy result. "
        "Expected Population, list[Individual], or strategy.history.pareto_archive."
    )


def _build_strategy(
    method: str,
    search_space: CSVSearchSpace,
    evaluator: Evaluator,
    pop_size: int,
    budget: int,
):
    if method == "skyline":
        return SkylineSearch(search_space=search_space, evaluator=evaluator)
    if method == "bruteforce":
        return BruteForceParetoSearch(search_space=search_space, evaluator=evaluator)
    if method == "random":
        population = Population(search_space, evaluator, size=pop_size)
        return RandomSearch(
            population=population,
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=ElitistReplacement(),
            evaluator=evaluator,
            budget=budget,
        )
    if method == "mowso":
        return MOWSOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
        )
    if method == "mosho":
        return MOSHOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
        )
    if method == "pso":
        population = PSOPopulation(search_space, evaluator, size=pop_size, w=0.4)
        return PSOSearchStrategy(
            population=population,
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(search_space),
            replacement=CrowdingReplacement(),
            evaluator=evaluator,
            budget=budget,
            w=0.4,
        )
    if method == "abc_firefly":
        return ABCFireflyStrategy(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            budget=budget,
            fa_prob=0.5,
            archive_size=pop_size,
            exh_fraction=4.0,
            dmt_fraction=0.67,
        )
    if method == "abc":
        limit = max(5, budget // 25)
        population = ABCPopulation(search_space, evaluator, size=pop_size, abandonment_limit=limit)
        return ABCSearchStrategy(
            population=population,
            neighbor_sampler=ABCNeighborSampler(search_space),
            selection=RouletteWheelSelection(),
            evaluator=evaluator,
            budget=budget,
        )
    if method == "firefly":
        return FireflySearchStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
            w_perf=0.6,
            gamma=1.0,
            beta0=1.0,
            max_chances=5,
            use_fap=True,
            fa_prob=0.5,
        )
    if method == "hybrid_mbo":
        return HybridMBOStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
            w_perf=0.6,
            rank_fraction=0.5,
            fa_prob=0.5,
        )
    

    if method == "apso_e":
        return APSOESearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
            w=0.4,
            c1=1.5,
            c2=1.5,
            w_decay=0.99,
            max_hop=3,
            div_threshold=0.4,
            n_inject=4,
        )
    if method == "nsga2":
        return NSGA2SearchStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=2),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
        )
 
    raise ValueError(f"Unknown method: {method}")


def _to_points(front: list[Individual]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for ind in front:
        if ind.fitness is None:
            continue
        acc, lat = ind.fitness
        if acc != acc or lat != lat:
            continue
        points.append((float(acc), float(lat)))
    return points


def _contexts(datasets: tuple[str, ...], devices: tuple[str, ...]):
    context_id = 0
    for dataset in datasets:
        for device in devices:
            context_id += 1
            yield context_id, dataset, device


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.10g}"


def run_analysis(
    method: str,
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    results_root: Path,
) -> None:
    out_dir = results_root / method
    out_dir.mkdir(parents=True, exist_ok=True)

    pareto_rows: list[list[str]] = []
    run_context_metrics: list[dict[str, Any]] = []
    fronts_by_context: dict[int, list[list[tuple[float, float]]]] = defaultdict(list)

    for run_id in range(1, runs + 1):
        if seed is not None:
            random.seed(seed + run_id)

        for context_id, dataset, device in _contexts(datasets, devices):
            search_space = CSVSearchSpace(str(csv_path))
            benchmark = CSVBenchmarkAPI(str(csv_path))
            evaluator = Evaluator(benchmark, dataset=dataset, device=device)
            strategy = _build_strategy(method, search_space, evaluator, pop_size, budget)

            t0 = time.perf_counter()
            result = strategy.run()
            runtime_s = time.perf_counter() - t0
            front = _extract_front(strategy, result)
            points = _to_points(front)
            fronts_by_context[context_id].append(points)

            best_acc = max((p[0] for p in points), default=float("nan"))
            best_lat = min((p[1] for p in points), default=float("nan"))
            evals = getattr(strategy, "evaluations", 0)

            for ind in front:
                if ind.fitness is None:
                    continue
                acc, lat = ind.fitness
                if acc != acc or lat != lat:
                    continue
                arch_id = ind.metadata.get("arch_id") if ind.metadata else None
                pareto_rows.append([
                    str(run_id),
                    str(context_id),
                    str(arch_id),
                    device,
                    dataset,
                    f"{acc:.6f}",
                    f"{lat:.6f}",
                    f"{runtime_s:.6f}",
                    str(evals),
                ])

            run_context_metrics.append(
                {
                    "run_id": run_id,
                    "context_id": context_id,
                    "dataset": dataset,
                    "device": device,
                    "best_accuracy": best_acc,
                    "best_latency": best_lat,
                    "runtime": runtime_s,
                    "points": points,
                }
            )

    # Build reference fronts and reference points per context from all runs.
    reference_front_by_context: dict[int, list[tuple[float, float]]] = {}
    ref_point_by_context: dict[int, tuple[float, float]] = {}
    for context_id, runs_fronts in fronts_by_context.items():
        union_points = [p for front in runs_fronts for p in front]
        reference_front = non_dominated(union_points, OBJECTIVE_DIRECTIONS)
        reference_front_by_context[context_id] = reference_front

        if union_points:
            min_acc = min(p[0] for p in union_points)
            max_lat = max(p[1] for p in union_points)
            # Slight margin to include all dominated rectangles for HV.
            ref_point_by_context[context_id] = (min_acc - 1e-9, max_lat + 1e-9)
        else:
            ref_point_by_context[context_id] = (0.0, 0.0)

    # Compute metrics per run/context.
    metrics_rows: list[list[str]] = []
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    hv_by_context: dict[int, list[float]] = defaultdict(list)
    runtime_by_context: dict[int, list[float]] = defaultdict(list)

    for rec in run_context_metrics:
        cid = rec["context_id"]
        points = rec["points"]
        reference_front = reference_front_by_context[cid]
        ref_point = ref_point_by_context[cid]

        hv = normalized_hypervolume_to_reference_2d(
            points,
            reference_front,
            OBJECTIVE_DIRECTIONS,
            ref_point,
        )
        igd_p = igd_plus(points, reference_front, OBJECTIVE_DIRECTIONS)
        c_val = c_metric(points, reference_front, OBJECTIVE_DIRECTIONS)
        runtime_s = rec["runtime"]

        grouped[cid].append(
            {
                "best_accuracy": rec["best_accuracy"],
                "best_latency": rec["best_latency"],
                "hv": hv,
                "igd_plus": igd_p,
                "c_metric": c_val,
                "runtime": runtime_s,
            }
        )
        hv_by_context[cid].append(hv)
        runtime_by_context[cid].append(runtime_s)

        metrics_rows.append(
            [
                str(rec["run_id"]),
                str(cid),
                rec["device"],
                rec["dataset"],
                _fmt(rec["best_accuracy"]),
                _fmt(rec["best_latency"]),
                _fmt(hv),
                _fmt(igd_p),
                _fmt(c_val),
                _fmt(runtime_s),
            ]
        )

    # Aggregate mean/std per context.
    summary_rows: list[list[str]] = []
    context_meta = {cid: (ds, dv) for cid, ds, dv in _contexts(datasets, devices)}

    for cid in sorted(grouped.keys()):
        vals = grouped[cid]
        dataset, device = context_meta[cid]

        def _m(key: str) -> float:
            return mean(v[key] for v in vals)

        def _s(key: str) -> float:
            if len(vals) <= 1:
                return 0.0
            return pstdev(v[key] for v in vals)

        summary_rows.append(
            [
                str(cid),
                device,
                dataset,
                _fmt(_m("best_accuracy")),
                _fmt(_s("best_accuracy")),
                _fmt(_m("best_latency")),
                _fmt(_s("best_latency")),
                _fmt(_m("hv")),
                _fmt(_s("hv")),
                _fmt(_m("igd_plus")),
                _fmt(_s("igd_plus")),
                _fmt(_m("c_metric")),
                _fmt(_s("c_metric")),
                _fmt(_m("runtime")),
                _fmt(_s("runtime")),
            ]
        )

    pareto_csv = out_dir / f"{method}_pareto_front.csv"
    metrics_csv = out_dir / f"{method}_context_metrics.csv"
    per_run_csv = out_dir / f"{method}_metrics_by_run.csv"

    _write_csv(
        pareto_csv,
        [
            "run_id",
            "context_id",
            "archid",
            "device",
            "dataset",
            "accuracy",
            "latency",
            "runtime_sec",
            "evaluations",
        ],
        pareto_rows,
    )

    _write_csv(
        metrics_csv,
        [
            "context_id",
            "device",
            "dataset",
            "best_accuracy_mean",
            "best_accuracy_std",
            "best_latency_mean",
            "best_latency_std",
            "hv_mean",
            "hv_std",
            "igd_plus_mean",
            "igd_plus_std",
            "c_metric_mean",
            "c_metric_std",
            "runtime_mean",
            "runtime_std",
        ],
        summary_rows,
    )

    _write_csv(
        per_run_csv,
        [
            "run_id",
            "context_id",
            "device",
            "dataset",
            "best_accuracy",
            "best_latency",
            "hv",
            "igd_plus",
            "c_metric",
            "runtime_sec",
        ],
        metrics_rows,
    )

    # Plots
    save_hv_boxplot(hv_by_context, out_dir / "hv_boxplot.png")
    save_runtime_boxplot(runtime_by_context, out_dir / "runtime_boxplot.png")

    hv_means = {cid: mean(vals) for cid, vals in hv_by_context.items() if vals}
    save_context_metric_heatmap(
        hv_means,
        out_dir / "hv_mean_heatmap.png",
        title=f"HV Mean by Context ({method})",
    )

    # Pareto scatter per context from all runs + reference front.
    for cid, fronts in fronts_by_context.items():
        merged = [p for front in fronts for p in front]
        reference = reference_front_by_context.get(cid, [])
        dataset, device = context_meta[cid]
        save_pareto_scatter(
            points=merged,
            reference_points=reference,
            title=f"Pareto Scatter context={cid} ({dataset}, {device})",
            output_path=out_dir / f"pareto_scatter_context_{cid}.png",
        )

    print("Analysis completed successfully.")
    print(f"Method: {method}")
    print(f"Runs: {runs}")
    print(f"Output folder: {out_dir}")
    print(f"Pareto CSV: {pareto_csv}")
    print(f"Context summary CSV: {metrics_csv}")
    print(f"Per-run metrics CSV: {per_run_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one method on all contexts multiple times and export metrics + plots."
    )
    parser.add_argument(
        "--method",
        choices=["random", "bruteforce", "skyline", "mowso", "mosho", "abc_firefly", "pso", "abc", "firefly", "hybrid_mbo", "apso_e", "nsga2"],
        default="random",
        help="Search strategy method to analyze.",
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input benchmark CSV.",
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of repetitions.")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (random method).")
    parser.add_argument("--budget", type=int, default=60, help="Evaluation budget (random method).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root output directory where method folder will be created.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    run_analysis(
        method=args.method,
        csv_path=csv_path,
        runs=args.runs,
        pop_size=args.pop_size,
        budget=args.budget,
        seed=args.seed,
        datasets=DEFAULT_DATASETS,
        devices=DEFAULT_DEVICES,
        results_root=_resolve_path(args.results_root),
    )
