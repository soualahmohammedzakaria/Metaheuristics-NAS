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
from nas_framework.replacement import ElitistReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import GeneticAlgorithmNG, RandomSearch
from nas_framework.selection import TournamentSelection

CSV = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS = [100, 300, 500, 1000]
N_RUNS  = 10
STRATEGIES = [
    ("RandomSearch",  RandomSearch,       dict()),
    ("NG hybrid",     GeneticAlgorithmNG, dict(p_neigh=0.8, p_guide=0.2)),
    ("Neighbor only", GeneticAlgorithmNG, dict(p_neigh=1.0, p_guide=0.0)),
    ("Guidance only", GeneticAlgorithmNG, dict(p_neigh=0.0, p_guide=1.0)),
]
# ─────────────────────────────────────────────────────────────────────────────


def run_once(strategy_cls, seed: int, budget: int, extra_kwargs: dict) -> dict:
    random.seed(seed)
    search_space = CSVSearchSpace(str(CSV))
    benchmark    = CSVBenchmarkAPI(str(CSV))
    evaluator    = Evaluator(benchmark, dataset="cifar100", device="edgegpu")
    population   = Population(search_space, evaluator, size=20)

    strategy = strategy_cls(
        population  = population,
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = SinglePointMutation(search_space),
        replacement = ElitistReplacement(),
        evaluator   = evaluator,
        budget      = budget,
        **extra_kwargs,
    )
    final  = strategy.run()
    best   = final.best()          # most representative point on Pareto front
    front  = final.pareto_front()

    return {
        "acc":         best.fitness[0],
        "lat":         best.fitness[1],
        "arch_id":     best.metadata.get("arch_id"),
        "pareto_size": len(front),
    }


def run_multiple(strategy_cls, extra_kwargs: dict, budget: int, n_runs: int) -> dict:
    results = [run_once(strategy_cls, seed, budget, extra_kwargs) for seed in range(n_runs)]

    accs  = [r["acc"]         for r in results]
    lats  = [r["lat"]         for r in results]
    pss   = [r["pareto_size"] for r in results]

    # best individual overall = the run whose best() had highest accuracy
    top   = max(results, key=lambda r: r["acc"])

    return {
        "acc_mean":  statistics.mean(accs),
        "acc_std":   statistics.stdev(accs),
        "lat_mean":  statistics.mean(lats),
        "lat_std":   statistics.stdev(lats),
        "best_acc":  top["acc"],
        "best_lat":  top["lat"],
        "best_id":   top["arch_id"],
        "ps_mean":   statistics.mean(pss),
        "ps_std":    statistics.stdev(pss),
    }


# ── table helpers ─────────────────────────────────────────────────────────────

def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r

def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"

def print_table(budget: int, data: dict):
    cols   = ["Strategy", "Acc mean±std", "Lat mean±std", "Best acc", "Best lat", "Best arch id", "Pareto"]
    widths = [14, 18, 18, 10, 10, 13, 11]
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

def main():
    all_results = {}
    for budget in BUDGETS:
        print(f"\nRunning budget={budget} ...", flush=True)
        budget_data = {}
        for name, cls, kwargs in STRATEGIES:
            print(f"  {name} ...", end=" ", flush=True)
            stats = run_multiple(cls, kwargs, budget, N_RUNS)
            budget_data[name] = stats
            print("done")
        all_results[budget] = budget_data
        print_table(budget, budget_data)



if __name__ == "__main__":
    main()