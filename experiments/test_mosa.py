from __future__ import annotations

import io
import random
import statistics
import sys
from pathlib import Path

# Ensure imports work when running: python experiments/test_mosa.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from nas_framework.mosa import mosa, random_search

# ── config ────────────────────────────────────────────────────────────────────
BUDGETS = [100, 300, 500]
N_RUNS = 10
COOLING_RATE = 0.85
DEFAULT_BURN_IN = 100
STRATEGIES = [
    ("Random Search", random_search, dict()),
    ("MOSA", mosa, dict(cooling_rate=COOLING_RATE, burn_in_iters=DEFAULT_BURN_IN)),
]
# ─────────────────────────────────────────────────────────────────────────────

def run_once(strategy_fn, seed: int, budget: int, extra_kwargs: dict) -> dict:
    front = strategy_fn(total_budget=budget, seed=seed, **extra_kwargs)

    if not front:
        return {
            "acc": 0.0,
            "flops": float("inf"),
            "pareto_size": 0,
            "best_solution": None,
        }

    best = max(front, key=lambda s: (s.f1, -s.f2))

    return {
        "acc": best.f1,
        "flops": best.f2,
        "pareto_size": len(front),
        "best_solution": best,
    }

def run_multiple(
    strategy_fn,
    extra_kwargs: dict,
    budget: int,
    n_runs: int,
) -> dict:
    results = [
        run_once(strategy_fn, seed, budget, extra_kwargs) for seed in range(n_runs)
        for seed in range(n_runs)
    ]

    accs = [r["acc"] for r in results]
    flps = [r["flops"] for r in results]
    pss = [r["pareto_size"] for r in results]

    top = max(results, key=lambda r: (r["acc"], -r["flops"]))

    return {
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "flops_mean": statistics.mean(flps),
        "flops_std": statistics.stdev(flps) if len(flps) > 1 else 0.0,
        "best_acc": top["acc"],
        "best_flops": top["flops"],
        "best_solution": top["best_solution"],
        "ps_mean": statistics.mean(pss),
        "ps_std": statistics.stdev(pss) if len(pss) > 1 else 0.0,
    }


# ── table helpers ─────────────────────────────────────────────────────────────


def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r


def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"


def print_table(budget: int, data: dict):
    cols = ["Strategy", "Acc mean±std", "FLOPs mean±std", "Best acc", "Best FLOPs", "Pareto"]
    widths = [14, 16, 18, 10, 12, 11]
    widths[0] = max(widths[0], max(len(k) for k in data))

    total_w = sum(w + 2 for w in widths) + len(widths) - 1
    title = f" Budget = {budget} evals  ({N_RUNS} runs each) "

    best_acc_mean = max(v["acc_mean"] for v in data.values())
    best_flops_mean = min(v["flops_mean"] for v in data.values())

    print()
    print("┌" + title.center(total_w, "─") + "┐")
    print(_row(cols, widths))
    print(_sep(widths))

    for name, stats in data.items():
        acc_tag = " ★" if stats["acc_mean"] == best_acc_mean else ""
        flp_tag = " ✓" if stats["flops_mean"] == best_flops_mean else ""
        cells = [
            name,
            f"{stats['acc_mean']:.4f}±{stats['acc_std']:.4f}{acc_tag}",
            f"{stats['flops_mean']:.2e}±{stats['flops_std']:.2e}{flp_tag}",
            f"{stats['best_acc']:.4f}",
            f"{stats['best_flops']:.2e}",
            f"{stats['ps_mean']:.1f}±{stats['ps_std']:.1f}",
        ]
        print(_row(cells, widths))

    print("└" + "─" * total_w + "┘")
    print("  ★ best mean accuracy   ✓ best mean FLOPs")

    best_strategy = max(data.items(), key=lambda item: (item[1]["best_acc"], -item[1]["best_flops"]))
    name, stats = best_strategy
    print(f"\nBest solution ({name}):")
    print(f"  acc={stats['best_acc']:.4f}, flops={stats['best_flops']:.2e}")
    sol = stats.get("best_solution")
    if sol is not None:
        print(f"  conv_blocks={len(sol.conv_blocks)}, fc_blocks={len(sol.fc_blocks)}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    random.seed(0)
    for budget in BUDGETS:
        print(f"\nRunning budget={budget} ...", flush=True)
        budget_data = {}
        for name, fn, kwargs in STRATEGIES:
            print(f"  {name} ...", end=" ", flush=True)
            try:
                stats = run_multiple(fn, kwargs, budget, N_RUNS)
                budget_data[name] = stats
                print("done")
            except Exception as e:
                print(f"error: {e}")
        print_table(budget, budget_data)


if __name__ == "__main__":
    main()

