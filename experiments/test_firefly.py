"""experiments/test_firefly.py

Benchmarks FireflySearchStrategy (RB-IFA) against RandomSearch and
GeneticAlgorithmNG across multiple evaluation budgets and w_perf values.

Usage
-----
    python experiments/test_firefly.py

Output mirrors test_ng.py and test_abc.py: one Unicode table per budget.

Two experiment modes are run:
  1. Main comparison: Firefly (w_perf=0.5) vs RandomSearch vs GA-NG
  2. w_perf sweep on Firefly at budget=300 to show tradeoff steering.
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
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Population
from nas_framework.replacement import ElitistReplacement, RankBasedReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import (
    FireflySearchStrategy,
    GeneticAlgorithmNG,
    RandomSearch,
)
from nas_framework.selection import TournamentSelection

CSV = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS    = [100, 300, 500, 1000]
N_RUNS     = 10
STRATEGIES = [
    ("RandomSearch",       "random"),
    ("GA-NG hybrid",       "ng"),
    ("Firefly (w=0.6)",    "firefly_06"),
]
# ─────────────────────────────────────────────────────────────────────────────


def _make_search_space() -> CSVSearchSpace:
    return CSVSearchSpace(str(CSV))


def _make_evaluator(ss: CSVSearchSpace) -> Evaluator:
    return Evaluator(CSVBenchmarkAPI(str(CSV)), dataset="cifar100", device="edgegpu")


def run_once_random(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss = _make_search_space(); ev = _make_evaluator(ss)
    s = RandomSearch(
        population=Population(ss, ev, size=20),
        selection=TournamentSelection(k=3), crossover=UniformCrossover(),
        mutation=SinglePointMutation(ss), replacement=ElitistReplacement(),
        evaluator=ev, budget=budget,
    )
    return _collect(s)


def run_once_ng(seed: int, budget: int) -> dict:
    random.seed(seed)
    ss = _make_search_space(); ev = _make_evaluator(ss)
    s = GeneticAlgorithmNG(
        population=Population(ss, ev, size=20),
        selection=TournamentSelection(k=3), crossover=UniformCrossover(),
        mutation=SinglePointMutation(ss), replacement=ElitistReplacement(),
        evaluator=ev, budget=budget, p_neigh=0.8, p_guide=0.2,
    )
    return _collect(s)


def run_once_firefly(seed: int, budget: int, w_perf: float = 0.6) -> dict:
    """Run RB-IFA.

    Population size: 20.
    w_perf: performance weight (0=cost-only, 1=performance-only).
      Default 0.6: slight accuracy bias prevents convergence to degenerate
      architectures (acc=1.0, lat~0.5) that exist in the benchmark and which
      dominate rank scoring at w<=0.5 due to their extreme latency advantage.
      w>=0.6 ensures accuracy rank outweighs latency rank sufficiently.
    gamma=1.0: moderate light absorption (balanced local/global search).
    max_chances=5: genetic fallback after 5 stagnant generations.
    """
    random.seed(seed)
    ss = _make_search_space(); ev = _make_evaluator(ss)
    s = FireflySearchStrategy(
        population  = Population(ss, ev, size=20),
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = SinglePointMutation(ss),
        replacement = RankBasedReplacement(w_perf=w_perf),
        evaluator   = ev,
        budget      = budget,
        w_perf      = w_perf,
        gamma       = 1.0,
        beta0       = 1.0,
        max_chances = 5,
    )
    return _collect(s)


def _collect(strategy) -> dict:
    final = strategy.run()
    best  = final.best()
    front = final.pareto_front()
    return {
        "acc":         best.fitness[0],
        "lat":         best.fitness[1],
        "arch_id":     best.metadata.get("arch_id"),
        "pareto_size": len(front),
    }


_RUNNERS = {
    "random":      lambda seed, budget: run_once_random(seed, budget),
    "ng":          lambda seed, budget: run_once_ng(seed, budget),
    "firefly_06":  lambda seed, budget: run_once_firefly(seed, budget, w_perf=0.6),
}


def run_multiple(kind: str, budget: int, n_runs: int) -> dict:
    results = [_RUNNERS[kind](seed, budget) for seed in range(n_runs)]
    accs = [r["acc"] for r in results]
    lats = [r["lat"] for r in results]
    pss  = [r["pareto_size"] for r in results]
    top  = max(results, key=lambda r: r["acc"])
    return {
        "acc_mean": statistics.mean(accs), "acc_std":  statistics.stdev(accs),
        "lat_mean": statistics.mean(lats), "lat_std":  statistics.stdev(lats),
        "best_acc": top["acc"], "best_lat": top["lat"],
        "best_id":  top["arch_id"],
        "ps_mean":  statistics.mean(pss),  "ps_std":   statistics.stdev(pss),
    }


# ── table helpers ─────────────────────────────────────────────────────────────

def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r

def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"

def print_table(title_str: str, data: dict) -> None:
    cols   = ["Strategy", "Acc mean±std", "Lat mean±std", "Best acc",
              "Best lat", "Best arch id", "Pareto"]
    widths = [18, 18, 18, 10, 10, 13, 11]
    widths[0] = max(widths[0], max(len(k) for k in data))

    total_w = sum(w + 2 for w in widths) + len(widths) - 1
    best_acc_mean = max(v["acc_mean"] for v in data.values())
    best_lat_mean = min(v["lat_mean"] for v in data.values())

    print()
    print("┌" + title_str.center(total_w, "─") + "┐")
    print(_row(cols, widths))
    print(_sep(widths))
    for name, v in data.items():
        acc_tag = " ★" if v["acc_mean"] == best_acc_mean else ""
        lat_tag = " ✓" if v["lat_mean"] == best_lat_mean else ""
        cells = [
            name,
            f"{v['acc_mean']:.4f}±{v['acc_std']:.4f}{acc_tag}",
            f"{v['lat_mean']:.4f}±{v['lat_std']:.4f}{lat_tag}",
            f"{v['best_acc']:.4f}", f"{v['best_lat']:.4f}",
            f"{v['best_id']}", f"{v['ps_mean']:.1f}±{v['ps_std']:.1f}",
        ]
        print(_row(cells, widths))
    print("└" + "─" * total_w + "┘")
    print("  ★ best mean accuracy   ✓ best mean latency")
    print("  Best acc/lat/arch id = from the single best() run across all seeds")


# ── w_perf sweep ──────────────────────────────────────────────────────────────

def run_wperf_sweep(budget: int = 300, n_runs: int = 10) -> None:
    """Show how w_perf steers the Firefly search toward accuracy vs latency."""
    w_values = [0.5, 0.6, 0.7, 0.8, 1.0]
    data = {}
    for w in w_values:
        label = f"Firefly w={w:.2f}"
        results = [run_once_firefly(seed, budget, w_perf=w) for seed in range(n_runs)]
        accs = [r["acc"] for r in results]
        lats = [r["lat"] for r in results]
        data[label] = {
            "acc_mean": statistics.mean(accs), "acc_std":  statistics.stdev(accs),
            "lat_mean": statistics.mean(lats), "lat_std":  statistics.stdev(lats),
            "best_acc": max(accs), "best_lat": min(lats),
            "best_id":  max(results, key=lambda r: r["acc"])["arch_id"],
            "ps_mean":  0.0, "ps_std": 0.0,
        }
    print_table(f" w_perf sweep  budget={budget}  ({n_runs} runs each) ", data)
    print("  w=0.5 → balanced   w=0.6 → slight acc bias (default)   w=1.0 → acc-only")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Main comparison across budgets
    for budget in BUDGETS:
        print(f"\nRunning budget={budget} ...", flush=True)
        budget_data: dict[str, dict] = {}
        for label, kind in STRATEGIES:
            print(f"  {label} ...", end=" ", flush=True)
            budget_data[label] = run_multiple(kind, budget, N_RUNS)
            print("done")
        print_table(f" Budget = {budget} evals  ({N_RUNS} runs each) ", budget_data)

    # 2. w_perf sweep at budget=300
    print("\nRunning w_perf sweep (budget=300) ...", flush=True)
    run_wperf_sweep(budget=300, n_runs=N_RUNS)


if __name__ == "__main__":
    main()