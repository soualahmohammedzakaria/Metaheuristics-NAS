from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure imports work when running: python experiments/test.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.evaluator import DvolverEvaluator
from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.search_space import NASSearchSpace, CSVGenotypeDvolverSearchSpace
from nas_framework.search_strategy import DvolverSearchStrategy
from nas_framework.termination import TerminationCriteria


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Dvolver multi-objective NAS search.")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--crossover-prob", type=float, default=0.1)
    parser.add_argument("--mutation-prob", type=float, default=0.1)
    parser.add_argument("--num-blocks-per-cell", type=int, default=5)
    parser.add_argument("--num-cells-N", type=int, default=2)
    parser.add_argument("--num-filters-F", type=int, default=32)
    parser.add_argument("--max-generations", type=int, default=40)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--hypervolume-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--device", default="edgegpu")
    parser.add_argument(
        "--benchmark-csv",
        default=None,
        help="If provided, Dvolver queries this NAS/HW CSV benchmark using evolved genotype vectors.",
    )
    parser.add_argument("--output-dir", default="results/dvolver")
    return parser


def _serialize_individual(ind) -> dict:
    return {
        "architecture": ind.architecture,
        "objectives": ind.objectives,
        "rank": ind.rank,
        "crowding_distance": ind.crowding_distance,
    }


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _select_top3_knee_points(pareto_front: list) -> list:
    if not pareto_front:
        return []

    acc = [float(ind.objectives[0]) for ind in pareto_front]
    speed = [float(ind.objectives[1]) for ind in pareto_front]
    n_acc = _normalize(acc)
    n_speed = _normalize(speed)

    scored = []
    for idx, ind in enumerate(pareto_front):
        # Smaller distance to ideal point (1,1) is better.
        d = ((1.0 - n_acc[idx]) ** 2 + (1.0 - n_speed[idx]) ** 2) ** 0.5
        scored.append((d, idx, ind))
    scored.sort(key=lambda x: x[0])

    selected = [item[2] for item in scored[:3]]
    return selected


def _write_outputs(history, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_json = [_serialize_individual(ind) for ind in history.all_individuals]
    with (output_dir / "all_evaluated_architectures.json").open("w", encoding="utf-8") as fh:
        json.dump(all_json, fh, indent=2)

    with (output_dir / "all_evaluated_architectures.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["idx", "accuracy", "speed", "rank", "crowding_distance", "architecture"])
        writer.writeheader()
        for idx, ind in enumerate(history.all_individuals):
            writer.writerow({
                "idx": idx,
                "accuracy": ind.objectives[0] if ind.objectives else None,
                "speed": ind.objectives[1] if ind.objectives else None,
                "rank": ind.rank,
                "crowding_distance": ind.crowding_distance,
                "architecture": json.dumps(ind.architecture),
            })

    pareto_front = history.get_current_pareto_front()
    top3 = _select_top3_knee_points(pareto_front)
    labels = ["Dvolver-A", "Dvolver-B", "Dvolver-C"]
    top3_payload = {}
    for label, ind in zip(labels, top3):
        top3_payload[label] = _serialize_individual(ind)

    with (output_dir / "top3_architectures.json").open("w", encoding="utf-8") as fh:
        json.dump(top3_payload, fh, indent=2)

    # Pareto front plot: accuracy vs speed
    if pareto_front:
        x = [float(ind.objectives[1]) for ind in pareto_front]
        y = [float(ind.objectives[0]) for ind in pareto_front]
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, c="tab:blue", alpha=0.8)
        plt.xlabel("Speed (2e9 / FLOPs)")
        plt.ylabel("Validation Accuracy")
        plt.title("Dvolver Pareto Front")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "pareto_front.png", dpi=150)
        plt.close()

    # Hypervolume convergence plot
    if history.hypervolume_history:
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(len(history.hypervolume_history))), history.hypervolume_history, color="tab:green")
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.title("Dvolver Hypervolume Convergence")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "hypervolume_curve.png", dpi=150)
        plt.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    config = {
        "population_size": args.population_size,
        "crossover_prob": args.crossover_prob,
        "mutation_prob": args.mutation_prob,
        "num_blocks_per_cell": args.num_blocks_per_cell,
        "num_cells_N": args.num_cells_N,
        "num_filters_F": args.num_filters_F,
        "train_epochs_proxy": 72,
        "batch_size": 150,
        "learning_rate_initial": 0.1,
        "learning_rate_final": 0.039,
        "val_split_size": 5000,
    }

    if args.benchmark_csv:
        benchmark_csv = Path(args.benchmark_csv)
        if not benchmark_csv.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_csv}")

        search_space = CSVGenotypeDvolverSearchSpace(str(benchmark_csv))
        benchmark_api = CSVBenchmarkAPI(str(benchmark_csv))
        evaluator = DvolverEvaluator(
            benchmark=benchmark_api,
            dataset=args.dataset,
            device=args.device,
            num_cells_N=config["num_cells_N"],
            num_filters_F=config["num_filters_F"],
        )
    else:
        search_space = NASSearchSpace(num_blocks_per_cell=config["num_blocks_per_cell"])
        evaluator = DvolverEvaluator(
            benchmark=None,
            dataset=args.dataset,
            device=args.device,
            num_cells_N=config["num_cells_N"],
            num_filters_F=config["num_filters_F"],
        )
    termination = TerminationCriteria(
        max_generations=args.max_generations,
        max_evaluations=args.max_evaluations,
        hypervolume_patience=args.hypervolume_patience,
    )

    strategy = DvolverSearchStrategy(
        population_size=config["population_size"],
        crossover_prob=config["crossover_prob"],
        mutation_prob=config["mutation_prob"],
        search_space=search_space,
        evaluator=evaluator,
        termination=termination,
    )

    history = strategy.run()
    output_dir = ROOT / args.output_dir
    _write_outputs(history, output_dir)

    pareto = history.get_current_pareto_front()
    hv = history.hypervolume_history[-1] if history.hypervolume_history else 0.0

    print("RUN_OK")
    print(f"generations: {len(history.entries) - 1}")
    print(f"evaluations: {history.evaluation_count}")
    print(f"pareto_size: {len(pareto)}")
    print(f"hypervolume_last: {hv:.6f}")
    print(f"benchmark_mode: {bool(args.benchmark_csv)}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
