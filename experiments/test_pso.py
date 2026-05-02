"""experiments/test_pso.py

Benchmarks PSOSearchStrategy (MOIPSO) against RandomSearch and
GeneticAlgorithmNG across multiple evaluation budgets.

Usage
-----
    python experiments/test_pso.py

Output mirrors test_firefly.py and test_abc.py: one Unicode table per budget.

Two experiment modes are run:
  1. Main comparison: PSO (MOIPSO) vs RandomSearch vs GA-NG
  2. Inertia weight sweep on PSO at budget=300 to show w sensitivity.
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
from nas_framework.mutation import SinglePointMutation, GaussianMutation
from nas_framework.population import Population, PSOPopulation
from nas_framework.replacement import ElitistReplacement, CrowdingReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import (
    PSOSearchStrategy,
    GeneticAlgorithmNG,
    RandomSearch,
)
from nas_framework.selection import TournamentSelection

CSV = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS    = [100, 300, 500, 1000]
N_RUNS     = 10
STRATEGIES = [
    ("RandomSearch",   "random"),
    ("GA-NG hybrid",   "ng"),
    ("PSO (MOIPSO)",   "pso"),
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


def run_once_pso(seed: int, budget: int, w: float = 0.4) -> dict:
    """Run MOIPSO.

    Population size: 20.
    w: inertia weight (default 0.4, from Shao et al. Table 3).
      Controls how much of the previous velocity is retained each step.
      Lower w = more responsive to pbest/gbest; higher w = more momentum.
    Trigonometric c1/c2: computed fresh each iteration (no fixed values).
    GaussianMutation: sigma = 0.1*(1 - t/T), applied to half population.
    CrowdingReplacement: Pareto rank + crowding distance for diversity.
    """
    random.seed(seed)
    ss = _make_search_space(); ev = _make_evaluator(ss)
    s = PSOSearchStrategy(
        population  = PSOPopulation(ss, ev, size=20, w=w),
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = GaussianMutation(ss),
        replacement = CrowdingReplacement(),
        evaluator   = ev,
        budget      = budget,
        w           = w,
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
    "random": lambda seed, budget: run_once_random(seed, budget),
    "ng":     lambda seed, budget: run_once_ng(seed, budget),
    "pso":    lambda seed, budget: run_once_pso(seed, budget, w=0.4),
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


# ── inertia weight sweep ──────────────────────────────────────────────────────

def run_w_sweep(budget: int = 300, n_runs: int = 10) -> None:
    """Show how inertia weight w steers the PSO exploration-exploitation balance.

    Seeds are offset per w value (seed * 97 + int(w*100)) so each variant
    gets an independent random stream and results genuinely differ.
    """
    w_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    data = {}
    for w in w_values:
        label = f"PSO w={w:.1f}"
        # Offset seeds so each w gets a different random stream
        results = [run_once_pso(seed * 97 + int(w * 100), budget, w=w)
                   for seed in range(n_runs)]
        accs = [r["acc"] for r in results]
        lats = [r["lat"] for r in results]
        pss  = [r["pareto_size"] for r in results]
        top  = max(results, key=lambda r: r["acc"])
        data[label] = {
            "acc_mean": statistics.mean(accs), "acc_std":  statistics.stdev(accs),
            "lat_mean": statistics.mean(lats), "lat_std":  statistics.stdev(lats),
            "best_acc": top["acc"], "best_lat": top["lat"],
            "best_id":  top["arch_id"],
            "ps_mean":  statistics.mean(pss), "ps_std": statistics.stdev(pss),
        }
    print_table(f" Inertia weight sweep  budget={budget}  ({n_runs} runs each) ", data)
    print("  w=0.2 -> fast convergence   w=0.4 -> paper default   w=1.0 -> high momentum")


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

    # 2. Inertia weight sweep at budget=300
    print("\nRunning inertia weight sweep (budget=300) ...", flush=True)
    run_w_sweep(budget=300, n_runs=N_RUNS)


if __name__ == "__main__":
    main()