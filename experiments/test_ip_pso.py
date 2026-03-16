import random
import statistics
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.ip_pso_search import IPPSOSearch, IPRandomSearch
from nas_framework.ip_pso_population import PSOPopulation
from nas_framework.ip_evaluator import IPPSOEvaluator
from nas_framework.search_strategy import RandomSearch

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS = [100, 300, 500, 1000]
N_RUNS  = 10
STRATEGIES = [
    ("Random Search", IPRandomSearch, dict()),
    ("MO IPPSO",       IPPSOSearch,    dict()),
]
# ─────────────────────────────────────────────────────────────────────────────

def run_once(strategy_cls, seed: int, budget: int, extra_kwargs: dict) -> dict:
    random.seed(seed)
    # Instantiate inside loop so seed initializes correctly for the new run
    population = PSOPopulation(size=20)
    evaluator  = IPPSOEvaluator(mock=True) 

    strategy = strategy_cls(
        population=population,
        evaluator=evaluator,
        budget=budget,
        **extra_kwargs,
    )
    final  = strategy.run()
    best   = final.best()          # most representative point on Pareto front
    front  = final.pareto_front()

    return {
        "acc":          best.fitness[0],
        "lat":          best.fitness[1], 
        "arch_id":      best.metadata.get("arch_id"),
        "architecture": best.metadata.get("architecture"),
        "num_layers":   best.metadata.get("num_layers"),
        "pareto_size":  len(front),
    }

def run_multiple(strategy_cls, extra_kwargs: dict, budget: int, n_runs: int) -> dict:
    results = [run_once(strategy_cls, seed, budget, extra_kwargs) for seed in range(n_runs)]

    accs  = [r["acc"]         for r in results]
    lats  = [r["lat"]         for r in results]
    pss   = [r["pareto_size"] for r in results]

    top   = max(results, key=lambda r: r["acc"])

    return {
        "acc_mean":      statistics.mean(accs),
        "acc_std":       statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "lat_mean":      statistics.mean(lats),
        "lat_std":       statistics.stdev(lats) if len(lats) > 1 else 0.0,
        "best_acc":      top["acc"],
        "best_lat":      top["lat"],
        "best_id":       top["arch_id"],
        "best_arch":     top["architecture"],
        "best_n_layers": top["num_layers"],
        "ps_mean":       statistics.mean(pss),
        "ps_std":        statistics.stdev(pss) if len(pss) > 1 else 0.0,
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
    
    # Priority 2: Decoding winning architecture
    best_strategy = max(data.items(), key=lambda item: item[1]["best_acc"])
    name, stats = best_strategy
    print(f"\nWinning Architecture ({name}):")
    print(f"  Accuracy: {stats['best_acc']:.4f}, Latency: {stats['best_lat']:.4f}")
    print(f"  Layers: {stats['best_arch']}")
    print(f"  Num Layers: {stats['best_n_layers']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    for budget in BUDGETS:
        print(f"\nRunning budget={budget} ...", flush=True)
        budget_data = {}
        for name, cls, kwargs in STRATEGIES:
            print(f"  {name} ...", end=" ", flush=True)
            try:
                stats = run_multiple(cls, kwargs, budget, N_RUNS)
                budget_data[name] = stats
                print("done")
            except Exception as e:
                print(f"error: {e}")
        all_results[budget] = budget_data
        print_table(budget, budget_data)

if __name__ == "__main__":
    main()
