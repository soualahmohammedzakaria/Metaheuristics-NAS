from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Ensure imports work when running: python experiments/test_lf_mogp.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FIX 3: removed non-existent 'exact_pareto_front_2d' import
# FIX 3: removed unused imports (UniformCrossover, SinglePointMutation,
#         ElitistReplacement, TournamentSelection, Population)
from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.evaluator import Evaluator
from nas_framework.lf_mogp_search import LFMOGPSearch
from nas_framework.population import Individual, Population
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import RandomSearch, SkylineSearch
from nas_framework.selection import TournamentSelection
from nas_framework.crossover import UniformCrossover
from nas_framework.mutation import SinglePointMutation
from nas_framework.replacement import ElitistReplacement


# ── CLI ───────────────────────────────────────────────────────────────────────

def _resolve_csv_path(raw_csv: str) -> Path:
    candidate = Path(raw_csv)
    if candidate.exists():
        return candidate
    fallback_candidates = [
        ROOT / "nas_benchmarks" / "datasets" / candidate.name,
        ROOT / "nas_benchmarks" / candidate.name,
        ROOT / "datasets" / candidate.name,
    ]
    for path in fallback_candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "CSV file not found. Tried: "
        f"{candidate}, {fallback_candidates[0]}, {fallback_candidates[1]}"
    )


def _normalize_device(device: str) -> str:
    if device == "edgepu":
        return "edgegpu"
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare LF-MOGP, Random Search and Skyline on a NAS benchmark."
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Path to CSV benchmark/search-space file.",
    )
    parser.add_argument("--dataset",  default="cifar100", help="Dataset name.")
    parser.add_argument("--device",   default="edgegpu",  help="Device name.")
    parser.add_argument(
        "--pop-size", type=int, default=30,
        help="Population size for LF-MOGP and Random Search.",
    )
    parser.add_argument(
        "--elite-size", type=int, default=10,
        help="Elite size (K) for LF-MOGP.",
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Max generations for LF-MOGP (default: auto from budget).",
    )
    parser.add_argument(
        "--budget", type=int, default=500,
        help="Evaluation budget for LF-MOGP and Random Search.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=10,
        help="Maximum Pareto rows to print per algorithm.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser


# ── helpers ───────────────────────────────────────────────────────────────────

def _pareto_stats(front: list[Individual]) -> dict:
    fits = [ind.fitness for ind in front if ind.fitness is not None]
    if not fits:
        return {
            "size":     0,
            "best_acc": float("nan"),
            "best_lat": float("nan"),
            "hv_proxy": float("nan"),
        }
    best_acc = max(f[0] for f in fits)
    best_lat = min(f[1] for f in fits)
    hv_proxy = sum(f[0] / max(f[1], 1e-9) for f in fits)
    return {
        "size":     len(fits),
        "best_acc": best_acc,
        "best_lat": best_lat,
        "hv_proxy": hv_proxy,
    }


def _print_front(label: str, front: list[Individual], max_rows: int) -> None:
    rows: list[tuple] = []
    for ind in front:
        if ind.fitness is None:
            continue
        arch_id = ind.metadata.get("arch_id") if ind.metadata else None
        rows.append((arch_id, ind.fitness[0], ind.fitness[1]))
    rows.sort(key=lambda r: (-r[1], r[2]))
    shown = min(max_rows, len(rows))
    print(f"\n  [{label}] Pareto front (top {shown} of {len(rows)}):")
    for arch_id, acc, lat in rows[:shown]:
        print(f"    arch_id={arch_id}, accuracy={acc:.6f}, latency={lat:.6f}")


# ── runners ───────────────────────────────────────────────────────────────────

def run_lf_mogp(
    search_space: CSVSearchSpace,
    evaluator: Evaluator,
    pop_size: int,
    elite_size: int,
    max_generations: int | None,
    budget: int,
) -> tuple[list[Individual], float, int]:
    strategy = LFMOGPSearch(
        search_space    = search_space,
        evaluator       = evaluator,
        pop_size        = pop_size,
        elite_size      = elite_size,
        max_generations = max_generations,   # None -> auto
        budget          = budget,
    )
    t0    = time.perf_counter()
    front = strategy.run()
    return front, time.perf_counter() - t0, strategy.evaluations


def run_random_search(
    search_space: CSVSearchSpace,
    evaluator: Evaluator,
    pop_size: int,
    budget: int,
) -> tuple[list[Individual], float, int]:
    population = Population(search_space, evaluator, size=pop_size)
    strategy = RandomSearch(
        population  = population,
        selection   = TournamentSelection(k=3),
        crossover   = UniformCrossover(),
        mutation    = SinglePointMutation(search_space),
        replacement = ElitistReplacement(),
        evaluator   = evaluator,
        budget      = budget,
    )
    t0       = time.perf_counter()
    final    = strategy.run()
    elapsed  = time.perf_counter() - t0
    return final.pareto_front(), elapsed, strategy.evaluations


def run_skyline(
    search_space: CSVSearchSpace,
    evaluator: Evaluator,
) -> tuple[list[Individual], float, int]:
    strategy = SkylineSearch(search_space=search_space, evaluator=evaluator)
    t0       = time.perf_counter()
    front    = strategy.run()
    return front, time.perf_counter() - t0, strategy.evaluations


# ── summary helpers ───────────────────────────────────────────────────────────

def _print_summary(results: dict[str, dict]) -> None:
    print("\n" + "=" * 76)
    print("COMPARISON SUMMARY")
    print("=" * 76)
    col = "{:<16} {:>8} {:>10} {:>12} {:>12} {:>12} {:>10}"
    print(col.format(
        "Algorithm", "Evals", "Front sz",
        "Best Acc", "Best Lat", "HV proxy", "Time(s)",
    ))
    print("-" * 76)
    for name, r in results.items():
        print(col.format(
            name,
            r["evals"],
            r["size"],
            f"{r['best_acc']:.4f}",
            f"{r['best_lat']:.4f}",
            f"{r['hv_proxy']:.2f}",
            f"{r['time']:.2f}",
        ))


def _print_gap(results: dict[str, dict]) -> None:
    sky_r = results.get("Skyline", {})
    if not sky_r or sky_r["size"] == 0:
        return
    print("\nGap vs Skyline (optimal front):")
    for name in ("LF-MOGP", "Random Search"):
        if name not in results:
            continue
        r        = results[name]
        acc_gap  = sky_r["best_acc"] - r["best_acc"]
        lat_gap  = r["best_lat"] - sky_r["best_lat"]
        hv_ratio = (r["hv_proxy"] / sky_r["hv_proxy"]
                    if sky_r["hv_proxy"] > 0 else float("nan"))
        print(
            f"  {name:<16}  acc_gap={acc_gap:+.4f}  "
            f"lat_gap={lat_gap:+.4f}  "
            f"HV_ratio={hv_ratio:.3f}  "
            f"front_coverage={r['size']}/{sky_r['size']}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    csv_path = _resolve_csv_path(args.csv)
    dataset  = args.dataset
    device   = _normalize_device(args.device)

    print("=" * 76)
    print("NAS ALGORITHM COMPARISON")
    print(f"  csv     : {csv_path}")
    print(f"  dataset : {dataset}")
    print(f"  device  : {device}")
    print(f"  budget  : {args.budget}  (LF-MOGP & Random Search)")
    print("=" * 76)

    benchmark = CSVBenchmarkAPI(str(csv_path))
    results: dict[str, dict] = {}

    # ── 1. LF-MOGP ───────────────────────────────────────────────────────
    print("\n[1/3] Running LF-MOGP …")
    evaluator    = Evaluator(benchmark, dataset=dataset, device=device)
    search_space = CSVSearchSpace(str(csv_path))
    lf_front, lf_time, lf_evals = run_lf_mogp(
        search_space,
        evaluator,
        pop_size        = args.pop_size,
        elite_size      = args.elite_size,
        max_generations = args.generations,   # None -> auto from budget
        budget          = args.budget,
    )
    results["LF-MOGP"] = {
        **_pareto_stats(lf_front),
        "time": lf_time, "evals": lf_evals,
    }
    print(f"  done — evals={lf_evals}, "
          f"pareto_size={results['LF-MOGP']['size']}, "
          f"elapsed={lf_time:.2f}s")

    # ── 2. Random Search ─────────────────────────────────────────────────
    print("[2/3] Running Random Search …")
    evaluator    = Evaluator(benchmark, dataset=dataset, device=device)
    search_space = CSVSearchSpace(str(csv_path))
    rs_front, rs_time, rs_evals = run_random_search(
        search_space, evaluator,
        pop_size = args.pop_size,
        budget   = args.budget,
    )
    results["Random Search"] = {
        **_pareto_stats(rs_front),
        "time": rs_time, "evals": rs_evals,
    }
    print(f"  done — evals={rs_evals}, "
          f"pareto_size={results['Random Search']['size']}, "
          f"elapsed={rs_time:.2f}s")

    # ── 3. Skyline ───────────────────────────────────────────────────────
    print("[3/3] Running Skyline (exhaustive optimal front) …")
    evaluator    = Evaluator(benchmark, dataset=dataset, device=device)
    search_space = CSVSearchSpace(str(csv_path))
    sky_front, sky_time, sky_evals = run_skyline(search_space, evaluator)
    results["Skyline"] = {
        **_pareto_stats(sky_front),
        "time": sky_time, "evals": sky_evals,
    }
    print(f"  done — evals={sky_evals}, "
          f"pareto_size={results['Skyline']['size']}, "
          f"elapsed={sky_time:.2f}s")

    # ── Summary & gap ────────────────────────────────────────────────────
    _print_summary(results)
    _print_gap(results)

    # ── Per-algorithm Pareto rows ────────────────────────────────────────
    _print_front("LF-MOGP",       lf_front,  args.max_rows)
    _print_front("Random Search",  rs_front,  args.max_rows)
    _print_front("Skyline",        sky_front, args.max_rows)

    print("\nRUN_OK")


if __name__ == "__main__":
    main()