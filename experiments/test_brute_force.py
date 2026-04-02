from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure imports work when running: python experiments/test_brute_force.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.evaluator import Evaluator
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import BruteForceParetoSearch


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
    # Keep compatibility with common typo/alias used in notes.
    if device == "edgepu":
        return "edgegpu"
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run brute-force Pareto search for a single context."
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Path to CSV benchmark/search-space file.",
    )
    parser.add_argument("--dataset", default="cifar100", help="Dataset name.")
    parser.add_argument("--device", default="edgegpu", help="Device name.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of Pareto rows to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    csv_path = _resolve_csv_path(args.csv)
    dataset = args.dataset
    device = _normalize_device(args.device)

    search_space = CSVSearchSpace(str(csv_path))
    benchmark = CSVBenchmarkAPI(str(csv_path))
    evaluator = Evaluator(benchmark, dataset=dataset, device=device)

    strategy = BruteForceParetoSearch(search_space=search_space, evaluator=evaluator)
    front = strategy.run()

    print("RUN_OK")
    print(f"csv_path: {csv_path}")
    print(f"dataset: {dataset}")
    print(f"device: {device}")
    print(f"evaluations: {strategy.evaluations}")
    print(f"pareto_size: {len(front)}")

    rows: list[tuple[int | None, float, float]] = []
    for ind in front:
        if ind.fitness is None:
            continue
        arch_id = ind.metadata.get("arch_id") if ind.metadata else None
        rows.append((arch_id, ind.fitness[0], ind.fitness[1]))

    rows.sort(key=lambda r: (-r[1], r[2]))

    print("pareto_rows:")
    limit = max(0, args.max_rows)
    for arch_id, acc, lat in rows[:limit]:
        print(f"  arch_id={arch_id}, accuracy={acc:.6f}, latency={lat:.6f}")


if __name__ == "__main__":
    main()
