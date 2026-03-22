"""
Test: EE-TGA and E³-FA swarm algorithms for CNN NAS
=====================================================
Mirrors the structure of test_ip_pso.py.

Usage
-----
  python experiments/test_swarm_nas.py          # mock mode (fast)
  python experiments/test_swarm_nas.py --real   # real CIFAR-10 (slow)
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.ip_evaluator import IPPSOEvaluator
from nas_framework.ip_layer import MAX_LENGTH, decode_layer, LayerType
from nas_framework.eetga_search import EETGASearch
from nas_framework.e3fa_search import E3FASearch

# ── config ────────────────────────────────────────────────────────────────────
N_RUNS  = 10      # independent seeds per algorithm
T_MAX   = 20      # iterations per run
N_POP   = 8       # population size

STRATEGIES = [
    ("EE-TGA",  EETGASearch,  dict(N=N_POP, t_max=T_MAX)),
    ("E³-FA",   E3FASearch,   dict(N=N_POP, t_max=T_MAX)),
]
# ─────────────────────────────────────────────────────────────────────────────


# ── architecture decoder ──────────────────────────────────────────────────────

def decode_architecture(position) -> list[str]:
    """Decode an IP byte-pair position into a human-readable layer list."""
    layers = []
    for slot in range(MAX_LENGTH):
        b0 = position[slot * 2]
        b1 = position[slot * 2 + 1]
        layer = decode_layer(b0, b1)
        if layer.layer_type != LayerType.DISABLED:
            layers.append(repr(layer))
    return layers


# ── single run ────────────────────────────────────────────────────────────────

def run_once(strategy_cls, seed: int, evaluator: IPPSOEvaluator,
             extra_kwargs: dict) -> dict:
    random.seed(seed)

    strategy = strategy_cls(evaluator=evaluator, **extra_kwargs)
    best_sol, best_fit, history = strategy.run()

    arch = decode_architecture(best_sol)

    # Approximate latency proxy: number of active layers
    lat_proxy = float(len(arch))

    return {
        "acc":          best_fit,
        "lat":          lat_proxy,
        "architecture": arch,
        "num_layers":   len(arch),
        "history":      history,
        "evaluations":  strategy.evaluations,
    }


# ── multi-run aggregation ─────────────────────────────────────────────────────

def run_multiple(strategy_cls, extra_kwargs: dict,
                 mock: bool, n_runs: int) -> dict:
    # Share one evaluator per strategy to avoid re-loading data each run
    evaluator = IPPSOEvaluator(mock=mock)
    results = [
        run_once(strategy_cls, seed, evaluator, extra_kwargs)
        for seed in range(n_runs)
    ]

    accs  = [r["acc"]  for r in results]
    lats  = [r["lat"]  for r in results]
    top   = max(results, key=lambda r: r["acc"])

    return {
        "acc_mean":      statistics.mean(accs),
        "acc_std":       statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "lat_mean":      statistics.mean(lats),
        "lat_std":       statistics.stdev(lats) if len(lats) > 1 else 0.0,
        "best_acc":      top["acc"],
        "best_lat":      top["lat"],
        "best_arch":     top["architecture"],
        "best_n_layers": top["num_layers"],
        "evaluations":   statistics.mean([r["evaluations"] for r in results]),
    }


# ── table helpers ─────────────────────────────────────────────────────────────

def _sep(widths, l="├", m="┼", r="┤"):
    return l + m.join("─" * (w + 2) for w in widths) + r


def _row(cells, widths):
    return "│" + "│".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "│"


def print_table(data: dict, n_runs: int):
    cols   = ["Strategy", "Acc mean±std", "Lat mean±std", "Best acc",
              "Best lat", "Avg evals", "Best n_layers"]
    widths = [10, 20, 18, 10, 10, 11, 14]
    widths[0] = max(widths[0], max(len(k) for k in data))

    total_w = sum(w + 2 for w in widths) + len(widths) - 1
    title   = f" Swarm NAS — {n_runs} runs × {T_MAX} iters each "

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
            f"{v['lat_mean']:.1f}±{v['lat_std']:.1f}{lat_tag}",
            f"{v['best_acc']:.4f}",
            f"{v['best_lat']:.1f}",
            f"{v['evaluations']:.0f}",
            f"{v['best_n_layers']}",
        ]
        print(_row(cells, widths))

    print("└" + "─" * total_w + "┘")
    print("  ★ best mean accuracy   ✓ best mean latency")

    # Winning architecture
    best_strategy = max(data.items(), key=lambda item: item[1]["best_acc"])
    name, stats = best_strategy
    print(f"\nWinning Architecture ({name}):")
    print(f"  Best Acc : {stats['best_acc']:.4f}   Best Lat : {stats['best_lat']:.1f}")
    print(f"  Layers   : {stats['best_arch']}")


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare EE-TGA and E³-FA on IP-NAS.")
    p.add_argument("--real", action="store_true",
                   help="Use real CIFAR-10 training (slow). Default: mock mode.")
    p.add_argument("--runs", type=int, default=N_RUNS,
                   help=f"Independent runs per algorithm (default {N_RUNS}).")
    return p


def main():
    args = build_parser().parse_args()
    mock = not args.real
    n_runs = args.runs

    mode_tag = "MOCK (random fitness)" if mock else "REAL (CIFAR-10)"
    print("=" * 68)
    print("SWARM INTELLIGENCE NAS — EE-TGA vs E³-FA")
    print(f"  Mode    : {mode_tag}")
    print(f"  Runs    : {n_runs} seeds per algorithm")
    print(f"  Iters   : {T_MAX} per run,  Population : {N_POP}")
    print("=" * 68)

    all_data: dict = {}
    for name, cls, kwargs in STRATEGIES:
        print(f"\n[{name}] Running {n_runs} seeds …", flush=True)
        try:
            stats = run_multiple(cls, kwargs, mock, n_runs)
            all_data[name] = stats
            print(f"  done — best_acc={stats['best_acc']:.4f}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if all_data:
        print_table(all_data, n_runs)

    print("\nRUN_OK")


if __name__ == "__main__":
    main()
