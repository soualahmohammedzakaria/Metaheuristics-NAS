"""experiments/test_abc.py

Benchmarks the HiveNAS ABCSearchStrategy against RandomSearch and
GeneticAlgorithmNG across multiple evaluation budgets.

Usage
-----
    python experiments/test_abc.py

Output mirrors test_ng.py: one Unicode table per budget, with mean±std
accuracy, latency, best individual, and Pareto-front size for each strategy.
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import SinglePointMutation, ABCNeighborSampler
from nas_framework.population import Population, ABCPopulation, FoodSource
from nas_framework.replacement import ElitistReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import (
    ABCSearchStrategy,
    GeneticAlgorithmNG,
    RandomSearch,
)
from nas_framework.selection import RouletteWheelSelection, TournamentSelection

CSV = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS     = [100, 300, 500, 1000]
N_RUNS      = 10
STRATEGIES  = [
    ("RandomSearch",   "random"),
    ("GA-NG hybrid",   "ng"),
    ("HiveNAS (ABC)",  "abc"),
]
# ─────────────────────────────────────────────────────────────────────────────


def _make_search_space() -> CSVSearchSpace:
    return CSVSearchSpace(str(CSV))


def _make_evaluator(search_space: CSVSearchSpace) -> Evaluator:
    benchmark = CSVBenchmarkAPI(str(CSV))
    return Evaluator(benchmark, dataset="cifar100", device="edgegpu")


def run_once_random(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss  = _make_search_space()
    ev  = _make_evaluator(ss)
    pop = Population(ss, ev, size=20)

    strategy = RandomSearch(
        population  = pop,
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = SinglePointMutation(ss),
        replacement = ElitistReplacement(),
        evaluator   = ev,
        budget      = budget,
    )
    return _collect(strategy)


def run_once_ng(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss  = _make_search_space()
    ev  = _make_evaluator(ss)
    pop = Population(ss, ev, size=20)

    strategy = GeneticAlgorithmNG(
        population  = pop,
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = SinglePointMutation(ss),
        replacement = ElitistReplacement(),
        evaluator   = ev,
        budget      = budget,
        p_neigh     = 0.8,
        p_guide     = 0.2,
    )
    return _collect(strategy)


def run_once_abc(seed: int, budget: int) -> dict:
    """Run HiveNAS ABC strategy.

    Colony size: 10 (balances generations vs diversity across budgets).
    Abandonment limit: budget // 25, clamped to a minimum of 5.
      - Scales with budget so scouts reset at a healthy rate regardless
        of how many total evaluations are allowed.
      - Empirically tuned on the NAS-HW benchmark (grid search over
        colony x abandonment across budgets 100/300/500/1000, 10 runs each).
    """
    random.seed(seed)
    ss  = _make_search_space()
    ev  = _make_evaluator(ss)
    abandonment_limit = max(5, budget // 25)
    pop = ABCPopulation(ss, ev, size=10, abandonment_limit=abandonment_limit)

    strategy = ABCSearchStrategy(
        population         = pop,
        neighbor_sampler   = ABCNeighborSampler(ss),
        selection          = RouletteWheelSelection(),
        evaluator          = ev,
        budget             = budget,
    )
    return _collect(strategy)


def _collect(strategy) -> dict:
    final = strategy.run()
    # ABCSearchStrategy tracks the globally best individual across all
    # generations via history, since food sources can be abandoned mid-run.
    if hasattr(strategy, "_best_from_history"):
        best = strategy._best_from_history()
    else:
        best = final.best()
    front = final.pareto_front()
    return {
        "acc":         best.fitness[0],
        "lat":         best.fitness[1],
        "arch_id":     best.metadata.get("arch_id"),
        "pareto_size": len(front),
    }


_RUNNERS = {
    "random": run_once_random,
    "ng":     run_once_ng,
    "abc":    run_once_abc,
}


def run_multiple(kind: str, budget: int, n_runs: int) -> dict:
    runner  = _RUNNERS[kind]
    results = [runner(seed, budget) for seed in range(n_runs)]

    accs = [r["acc"]         for r in results]
    lats = [r["lat"]         for r in results]
    pss  = [r["pareto_size"] for r in results]
    top  = max(results, key=lambda r: r["acc"])

    return {
        "acc_mean": statistics.mean(accs),
        "acc_std":  statistics.stdev(accs),
        "lat_mean": statistics.mean(lats),
        "lat_std":  statistics.stdev(lats),
        "best_acc": top["acc"],
        "best_lat": top["lat"],
        "best_id":  top["arch_id"],
        "ps_mean":  statistics.mean(pss),
        "ps_std":   statistics.stdev(pss),
    }


# ── table helpers (identical to test_ng.py) ───────────────────────────────────

def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r

def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"

def print_table(budget: int, data: dict) -> None:
    cols   = ["Strategy", "Acc mean±std", "Lat mean±std", "Best acc",
              "Best lat", "Best arch id", "Pareto"]
    widths = [16, 18, 18, 10, 10, 13, 11]
    widths[0] = max(widths[0], max(len(k) for k in data))

    total_w = sum(w + 2 for w in widths) + len(widths) - 1
    title   = f" Budget = {budget} evals  ({N_RUNS} runs each) "

    best_acc_mean = max(v["acc_mean"] for v in data.values())
    best_lat_mean = min(v["lat_mean"] for v in data.values())

    print()
    print("┌" + title.center(total_w, "─") + "┐")
    print(_row(cols, widths))
    print(_sep(widths))

    for name, v in data.items():
        acc_tag = " ★" if v["acc_mean"] == best_acc_mean else ""
        lat_tag = " ✓" if v["lat_mean"] == best_lat_mean else ""
        cells = [
            name,
            f"{v['acc_mean']:.4f}±{v['acc_std']:.4f}{acc_tag}",
            f"{v['lat_mean']:.4f}±{v['lat_std']:.4f}{lat_tag}",
            f"{v['best_acc']:.4f}",
            f"{v['best_lat']:.4f}",
            f"{v['best_id']}",
            f"{v['ps_mean']:.1f}±{v['ps_std']:.1f}",
        ]
        print(_row(cells, widths))

    print("└" + "─" * total_w + "┘")
    print("  ★ best mean accuracy   ✓ best mean latency")
    print("  Best acc/lat/arch id = from the single best() run across all seeds")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for budget in BUDGETS:
        print(f"\nRunning budget={budget} ...", flush=True)
        budget_data: dict[str, dict] = {}
        for label, kind in STRATEGIES:
            print(f"  {label} ...", end=" ", flush=True)
            stats = run_multiple(kind, budget, N_RUNS)
            budget_data[label] = stats
            print("done")
        print_table(budget, budget_data)


if __name__ == "__main__":
    main()