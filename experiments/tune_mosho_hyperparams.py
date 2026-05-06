"""experiments/tune_mosho_hyperparams.py

Hyperparameter tuning for `nas_framework.search_strategy.MOSHOSearch`.

This script runs a lightweight successive-halving random search that maximizes
normalized hypervolume (HV) against a *fixed* reference Pareto front per context.

Why fixed reference?
- `experiments/run_method_analysis.py` builds the reference front from the union
  of runs, which makes scores change between trials.
- For tuning, you want a stable target. By default we use the precomputed
  optimal fronts in `experiments/results/optimal_pareto_fronts.csv`.

Usage (recommended)
-------------------
    python experiments/tune_mosho_hyperparams.py --budget 5000 --device edgegpu

You can narrow contexts for faster iteration:
    python experiments/tune_mosho_hyperparams.py --datasets cifar100 --device edgegpu

Output
------
Prints the best hyperparameters found (e0, e_min, e_max, delta) and a compact
score summary.

Notes
-----
- MOSHO currently stores `eta` but does not use it in the algorithm; this tuner
  therefore does not search over `eta`.
- The scoring objective is mean HV across all selected contexts and seeds.

"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure imports work when running: python experiments/tune_mosho_hyperparams.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.evaluator import Evaluator
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import MOSHOSearch
from utilities.metrics import igd_plus, normalized_hypervolume_to_reference_2d

DIRECTIONS: tuple[int, int] = (1, -1)  # maximize accuracy, minimize latency

DEFAULT_CSV = ROOT / "nas_benchmarks" / "datasets" / "nas_hw_search_space_bench.csv"
DEFAULT_REFERENCE = ROOT / "experiments" / "results" / "optimal_pareto_fronts.csv"
DEFAULT_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")


@dataclass(frozen=True)
class MoshoHyperparams:
    e0: float
    e_min: float
    e_max: float
    delta: float

    def to_kwargs(self) -> dict:
        return {
            "e0": float(self.e0),
            "e_min": float(self.e_min),
            "e_max": float(self.e_max),
            "delta": float(self.delta),
        }


@dataclass
class TrialResult:
    params: MoshoHyperparams
    score: float
    hv_mean: float
    hv_std: float
    igd_mean: float
    igd_std: float
    runtime_s: float


def _parse_list_arg(raw: str) -> tuple[str, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    if "," in raw:
        items = [x.strip() for x in raw.split(",")]
    else:
        items = [x.strip() for x in raw.split()]  # allow space-separated
    return tuple(x for x in items if x)


def _to_points(front: Iterable) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for ind in front:
        fit = getattr(ind, "fitness", None)
        if not fit:
            continue
        acc, lat = fit
        # filter NaNs
        if acc != acc or lat != lat:
            continue
        points.append((float(acc), float(lat)))
    return points


def _load_reference_fronts(path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference fronts file not found: {path}. "
            "Generate it first (e.g. via brute-force/skyline) or pass --reference."
        )

    fronts: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"device", "dataset", "accuracy", "latency"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV missing required columns: {sorted(missing)}")

        for row in reader:
            device = (row.get("device") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            if not device or not dataset:
                continue

            try:
                acc = float(row.get("accuracy") or "nan")
                lat = float(row.get("latency") or "nan")
            except ValueError:
                continue

            if acc != acc or lat != lat:
                continue

            fronts.setdefault((dataset, device), []).append((acc, lat))

    return fronts


def _reference_point(reference_front: list[tuple[float, float]]) -> tuple[float, float]:
    if not reference_front:
        return (0.0, 0.0)
    min_acc = min(p[0] for p in reference_front)
    max_lat = max(p[1] for p in reference_front)
    return (min_acc - 1e-9, max_lat + 1e-9)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("log-uniform bounds must be > 0")
    lo = math.log(low)
    hi = math.log(high)
    return math.exp(rng.uniform(lo, hi))


def _sample_hyperparams(rng: random.Random) -> MoshoHyperparams:
    # Conservative ranges centered around current defaults.
    e0 = rng.uniform(0.6, 2.2)

    # Ensure e_min < e0.
    e_min_hi = min(0.8, max(0.06, 0.9 * e0))
    e_min = rng.uniform(0.05, e_min_hi)

    # Ensure e_max >= e0.
    e_max_lo = max(1.0, e0)
    e_max = rng.uniform(e_max_lo, 3.0)

    # Energy decay per failed move.
    delta = _log_uniform(rng, 0.005, 0.15)

    # Round for readability/stable diffs when updating defaults.
    return MoshoHyperparams(
        e0=round(e0, 3),
        e_min=round(e_min, 3),
        e_max=round(e_max, 3),
        delta=round(delta, 4),
    )


def _evaluate_trial(
    *,
    search_space: CSVSearchSpace,
    benchmark: CSVBenchmarkAPI,
    reference_fronts: dict[tuple[str, str], list[tuple[float, float]]],
    datasets: tuple[str, ...],
    device: str,
    pop_size: int,
    budget: int,
    seeds: tuple[int, ...],
    params: MoshoHyperparams,
) -> TrialResult:
    hv_values: list[float] = []
    igd_values: list[float] = []

    t0 = time.perf_counter()
    for dataset in datasets:
        ref_front = reference_fronts.get((dataset, device), [])
        if not ref_front:
            raise KeyError(
                f"No reference Pareto front found for dataset={dataset}, device={device}. "
                "Check --reference or reduce contexts."
            )
        ref_point = _reference_point(ref_front)

        for seed in seeds:
            random.seed(seed)
            evaluator = Evaluator(benchmark, dataset=dataset, device=device)
            max_iterations = max(1, int(budget) // int(pop_size))

            strategy = MOSHOSearch(
                search_space=search_space,
                evaluator=evaluator,
                pop_size=pop_size,
                max_iterations=max_iterations,
                archive_size=pop_size,
                **params.to_kwargs(),
            )

            front = strategy.run()
            points = _to_points(front)
            hv = normalized_hypervolume_to_reference_2d(points, ref_front, DIRECTIONS, ref_point)
            igd = igd_plus(points, ref_front, DIRECTIONS)
            hv_values.append(float(hv))
            igd_values.append(float(igd))

    runtime_s = time.perf_counter() - t0

    hv_mean = statistics.mean(hv_values) if hv_values else 0.0
    hv_std = statistics.pstdev(hv_values) if len(hv_values) > 1 else 0.0
    igd_mean = statistics.mean(igd_values) if igd_values else float("inf")
    igd_std = statistics.pstdev(igd_values) if len(igd_values) > 1 else 0.0

    # Primary objective: maximize mean HV; small penalty for instability.
    score = hv_mean - 0.10 * hv_std

    return TrialResult(
        params=params,
        score=score,
        hv_mean=hv_mean,
        hv_std=hv_std,
        igd_mean=igd_mean,
        igd_std=igd_std,
        runtime_s=runtime_s,
    )


def _successive_halving(
    *,
    search_space: CSVSearchSpace,
    benchmark: CSVBenchmarkAPI,
    reference_fronts: dict[tuple[str, str], list[tuple[float, float]]],
    datasets: tuple[str, ...],
    device: str,
    pop_size: int,
    n_trials: int,
    seed: int,
    budgets: tuple[int, ...],
    stage_seeds: tuple[tuple[int, ...], ...],
    keep_fraction: float,
) -> TrialResult:
    rng = random.Random(seed)

    candidates = [_sample_hyperparams(rng) for _ in range(n_trials)]

    best: TrialResult | None = None

    for stage_idx, budget in enumerate(budgets):
        seeds = stage_seeds[min(stage_idx, len(stage_seeds) - 1)]

        print(f"\nStage {stage_idx + 1}/{len(budgets)}  budget={budget}  seeds={list(seeds)}")
        scored: list[TrialResult] = []

        for i, params in enumerate(candidates, start=1):
            tr = _evaluate_trial(
                search_space=search_space,
                benchmark=benchmark,
                reference_fronts=reference_fronts,
                datasets=datasets,
                device=device,
                pop_size=pop_size,
                budget=budget,
                seeds=seeds,
                params=params,
            )
            scored.append(tr)

            if best is None or tr.score > best.score:
                best = tr

            if i % max(1, len(candidates) // 10) == 0 or i == len(candidates):
                print(
                    f"  {i:>3}/{len(candidates)}  "
                    f"best_score={best.score:.4f}  "
                    f"best_hv={best.hv_mean:.4f}±{best.hv_std:.4f}"
                )

        scored.sort(key=lambda t: (t.score, t.hv_mean, -t.igd_mean), reverse=True)

        # Keep the top fraction for the next stage.
        keep_n = max(1, int(round(len(scored) * keep_fraction)))
        top = scored[:keep_n]

        print("Top candidates this stage:")
        for rank, tr in enumerate(top[: min(5, len(top))], start=1):
            p = tr.params
            print(
                f"  #{rank}: score={tr.score:.4f}  hv={tr.hv_mean:.4f}±{tr.hv_std:.4f}  "
                f"igd={tr.igd_mean:.4f}±{tr.igd_std:.4f}  "
                f"e0={p.e0} e_min={p.e_min} e_max={p.e_max} delta={p.delta}"
            )

        candidates = [tr.params for tr in top]

    if best is None:
        raise RuntimeError("No trials executed")
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune MOSHO hyperparameters via successive halving.")
    p.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to NAS CSV benchmark/search-space file.",
    )
    p.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="CSV containing the fixed reference Pareto fronts.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Datasets to tune on (comma or space separated).",
    )
    p.add_argument(
        "--device",
        default="edgegpu",
        help="Device to tune on (e.g., edgegpu, edgetpu, eyeriss).",
    )
    p.add_argument("--pop-size", type=int, default=20, help="MOSHO population size.")
    p.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="Final-stage evaluation budget proxy (used when --stages is not provided).",
    )
    p.add_argument("--n-trials", type=int, default=40, help="Number of random configs.")
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Master seed controlling the random hyperparameter samples.",
    )
    p.add_argument(
        "--keep-fraction",
        type=float,
        default=0.30,
        help="Fraction of configs to keep per halving stage.",
    )
    p.add_argument(
        "--stages",
        default="",
        help=(
            "Optional comma-separated budgets for successive halving stages. "
            "If omitted, stages are derived from --budget."
        ),
    )
    p.add_argument(
        "--stage-seeds",
        default="0;0,1;0,1,2",
        help=(
            "Per-stage seed lists (semicolon-separated stages; comma-separated seeds). "
            "Example: '0;0,1;0,1,2'"
        ),
    )
    return p


def _parse_int_list(raw: str) -> tuple[int, ...]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(int(x) for x in items)


def _parse_stage_seeds(raw: str) -> tuple[tuple[int, ...], ...]:
    stages = [s.strip() for s in raw.split(";") if s.strip()]
    parsed: list[tuple[int, ...]] = []
    for stage in stages:
        parsed.append(_parse_int_list(stage))
    return tuple(parsed)


def main() -> None:
    args = build_parser().parse_args()

    csv_path = Path(args.csv)
    ref_path = Path(args.reference)

    datasets = _parse_list_arg(args.datasets)
    if not datasets:
        datasets = DEFAULT_DATASETS

    if args.stages.strip():
        budgets = _parse_int_list(args.stages)
    else:
        final_budget = max(1, int(args.budget))
        # Default 3-stage schedule: fast screen → mid eval → full eval.
        stage1 = min(max(100, final_budget // 20), final_budget)
        stage2 = min(max(300, final_budget // 5), final_budget)
        budgets = (stage1, stage2, final_budget)
        # Drop duplicates (e.g., small budgets where stage2==final_budget).
        budgets = tuple(dict.fromkeys(budgets))
    stage_seeds = _parse_stage_seeds(args.stage_seeds)

    if not budgets or any(b <= 0 for b in budgets):
        raise ValueError(f"Invalid stage budgets: {budgets}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    reference_fronts = _load_reference_fronts(ref_path)

    # Reuse heavy CSV loads across trials.
    search_space = CSVSearchSpace(str(csv_path))
    benchmark = CSVBenchmarkAPI(str(csv_path))

    print("TUNING_CONTEXT")
    print(f"  csv: {csv_path}")
    print(f"  reference: {ref_path}")
    print(f"  device: {args.device}")
    print(f"  datasets: {list(datasets)}")
    print(f"  pop_size: {args.pop_size}")
    print(f"  budgets: {list(budgets)}")
    print(f"  n_trials: {args.n_trials}")

    best = _successive_halving(
        search_space=search_space,
        benchmark=benchmark,
        reference_fronts=reference_fronts,
        datasets=tuple(datasets),
        device=args.device,
        pop_size=int(args.pop_size),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        budgets=tuple(int(b) for b in budgets),
        stage_seeds=stage_seeds,
        keep_fraction=float(args.keep_fraction),
    )

    p = best.params
    print("\nBEST_MOSHO_HYPERPARAMS")
    print(f"  score: {best.score:.6f}")
    print(f"  hv:    {best.hv_mean:.6f} ± {best.hv_std:.6f}")
    print(f"  igd+:  {best.igd_mean:.6f} ± {best.igd_std:.6f}")
    print("  params:")
    print(f"    e0:    {p.e0}")
    print(f"    e_min: {p.e_min}")
    print(f"    e_max: {p.e_max}")
    print(f"    delta: {p.delta}")


if __name__ == "__main__":
    main()
