"""experiments/test_mosho.py

Benchmarks MOSHOSearch against MOSHOImprovedSearch across multiple budgets.

Usage
------
    python experiments/test_mosho.py

Output mirrors test_mbo.py: one table per budget.
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
from nas_framework.evaluator import Evaluator
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import MOSHOSearch, MOSHOEnhancedSearch

CSV = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

# --- config ---------------------------------------------------------------
BUDGETS = [100, 300, 500, 1000]
N_RUNS = 10
POP_SIZE = 20
STRATEGIES = [
    ("MOSHO", "mosho"),
    ("MOSHO Enhanced", "mosho_enhanced"),
]
# -------------------------------------------------------------------------


def _make_search_space() -> CSVSearchSpace:
    return CSVSearchSpace(str(CSV))


def _make_evaluator(search_space: CSVSearchSpace) -> Evaluator:
    benchmark = CSVBenchmarkAPI(str(CSV))
    return Evaluator(benchmark, dataset="cifar100", device="edgegpu")


def _weighted_score(ind, dirs) -> float:
    return sum(v * d for v, d in zip(ind.fitness, dirs))


def _collect(strategy) -> dict:
    result = strategy.run()
    dirs = strategy.evaluator.objective_directions

    if isinstance(result, list):
        archive = result
    else:
        archive = result.pareto_front()

    if not archive:
        return {
            "acc": float("nan"),
            "lat": float("nan"),
            "arch_id": None,
            "pareto_size": 0,
        }

    best = max(archive, key=lambda ind: _weighted_score(ind, dirs))
    return {
        "acc": best.fitness[0],
        "lat": best.fitness[1],
        "arch_id": best.metadata.get("arch_id") if best.metadata else None,
        "pareto_size": len(archive),
    }


def run_once_mosho(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss = _make_search_space()
    ev = _make_evaluator(ss)
    strategy = MOSHOSearch(
        search_space=ss,
        evaluator=ev,
        pop_size=POP_SIZE,
        max_iterations=max(1, budget // POP_SIZE),
        archive_size=POP_SIZE,
    )
    return _collect(strategy)


def run_once_mosho_enhanced(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss = _make_search_space()
    ev = _make_evaluator(ss)
    strategy = MOSHOEnhancedSearch(
        search_space=ss,
        evaluator=ev,
        pop_size=POP_SIZE,
        max_iterations=max(1, budget // POP_SIZE),
        archive_size=POP_SIZE,
    )
    return _collect(strategy)


_RUNNERS = {
    "mosho": run_once_mosho,
    "mosho_enhanced": run_once_mosho_enhanced,
}


def run_multiple(kind: str, budget: int, n_runs: int) -> dict:
    runner = _RUNNERS[kind]
    results = [runner(seed, budget) for seed in range(n_runs)]

    accs = [r["acc"] for r in results]
    lats = [r["lat"] for r in results]
    pss = [r["pareto_size"] for r in results]
    top = max(results, key=lambda r: r["acc"])

    return {
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.stdev(accs),
        "lat_mean": statistics.mean(lats),
        "lat_std": statistics.stdev(lats),
        "best_acc": top["acc"],
        "best_lat": top["lat"],
        "best_id": top["arch_id"],
        "ps_mean": statistics.mean(pss),
        "ps_std": statistics.stdev(pss),
    }


# --- table helpers --------------------------------------------------------

def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r


def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"


def print_table(budget: int, data: dict) -> None:
    cols = ["Strategy", "Acc mean±std", "Lat mean±std", "Best acc",
            "Best lat", "Best arch id", "Pareto"]
    widths = [16, 18, 18, 10, 10, 13, 11]
    widths[0] = max(widths[0], max(len(k) for k in data))

    total_w = sum(w + 2 for w in widths) + len(widths) - 1
    title = f" Budget = {budget} evals  ({N_RUNS} runs each) "

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


def main() -> None:
    for budget in BUDGETS:
        data = {}
        for label, key in STRATEGIES:
            data[label] = run_multiple(key, budget, N_RUNS)
        print_table(budget, data)


if __name__ == "__main__":
    main()
