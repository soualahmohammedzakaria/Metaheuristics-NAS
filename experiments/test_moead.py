from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure imports work when running: python experiments/run_moead.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Population
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import BiPopulationUniformSamplingMOEADStrategy


def _resolve_csv_path(raw_csv: str) -> Path:
    candidate = Path(raw_csv)
    if candidate.exists():
        return candidate

    fallback_candidates = [
        ROOT / "nas_benchmarks" / "datasets" / candidate.name,
        ROOT / "nas_benchmarks" / candidate.name,
        ROOT / "datasets" / candidate.name,
    ]
    for p in fallback_candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "CSV file not found. Tried: "
        f"{candidate}, {fallback_candidates[0]}, {fallback_candidates[1]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bi-population MOEA/D with uniform sampling on NAS/HW merged CSV benchmark."
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Path to CSV benchmark/search-space file.",
    )
    parser.add_argument("--dataset", default="cifar100", help="Dataset name.")
    parser.add_argument("--device", default="edgegpu", help="Device name.")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size.")
    parser.add_argument("--budget", type=int, default=120, help="Evaluation budget.")
    parser.add_argument("--neighbor-size", type=int, default=6, help="Neighborhood size.")
    parser.add_argument(
        "--neighbor-mating-prob",
        type=float,
        default=0.9,
        help="Probability of selecting parents from local neighborhood.",
    )
    parser.add_argument(
        "--max-replacements",
        type=int,
        default=2,
        help="Maximum number of neighborhood solutions replaced per offspring.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=5,
        help="How many history checkpoints to print.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser


def run_bi_population_uniform_sampling_moead(
    csv_path: Path,
    dataset: str,
    device: str,
    pop_size: int,
    budget: int,
    neighbor_size: int,
    neighbor_mating_prob: float,
    max_replacements: int,
    history_steps: int,
) -> None:
    search_space = CSVSearchSpace(str(csv_path))
    benchmark = CSVBenchmarkAPI(str(csv_path))
    evaluator = Evaluator(benchmark, dataset=dataset, device=device)
    population = Population(search_space, evaluator, size=pop_size)

    strategy = BiPopulationUniformSamplingMOEADStrategy(
        population=population,
        crossover=UniformCrossover(),
        mutation=SinglePointMutation(search_space),
        evaluator=evaluator,
        budget=budget,
        neighborhood_size=neighbor_size,
        neighborhood_mating_prob=neighbor_mating_prob,
        max_replacements=max_replacements,
    )

    final_population = strategy.run()
    best = final_population.best()
    last = strategy.history.entries[-1]

    print("RUN_OK")
    print(f"csv_path: {csv_path}")
    print(f"evaluations: {strategy.evaluations}")
    print(f"generations: {strategy.generations}")
    print(f"population_size: {len(final_population.individuals)}")
    print(f"best_genotype: {best.genotype}")
    print(f"best_fitness: {best.fitness}")
    print(f"best_arch_id: {best.metadata.get('arch_id')}")
    print(
        "last_history: "
        f"gen={last.generation}, evals={last.evaluations}, "
        f"pop={last.population_size}, pareto={last.pareto_front_size}"
    )

    entries = strategy.history.entries
    pareto_archive = strategy.history.pareto_archive
    if entries:
        n = max(1, min(history_steps, len(entries)))
        if n == 1:
            indices = [0]
        else:
            indices = sorted({int(i * (len(entries) - 1) / (n - 1)) for i in range(n)})

        print("history_checkpoints:")
        for i in indices:
            e = entries[i]
            front = pareto_archive[i] if i < len(pareto_archive) else []
            fits = [ind.fitness for ind in front if ind.fitness is not None]
            if fits:
                best_acc = max(f[0] for f in fits)
                best_lat = min(f[1] for f in fits)
                best_acc_point = max(fits, key=lambda f: f[0])
                metric_part = (
                    f"best_acc={best_acc:.4f}, best_lat={best_lat:.4f}, "
                    f"top_acc_point=(acc={best_acc_point[0]:.4f}, lat={best_acc_point[1]:.4f})"
                )
            else:
                metric_part = "best_acc=nan, best_lat=nan"
            print(
                f"  step[{i}] gen={e.generation}, evals={e.evaluations}, "
                f"pop={e.population_size}, pareto={e.pareto_front_size}, {metric_part}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    csv_path = _resolve_csv_path(args.csv)

    run_bi_population_uniform_sampling_moead(
        csv_path=csv_path,
        dataset=args.dataset,
        device=args.device,
        pop_size=args.pop_size,
        budget=args.budget,
        neighbor_size=args.neighbor_size,
        neighbor_mating_prob=args.neighbor_mating_prob,
        max_replacements=args.max_replacements,
        history_steps=args.history_steps,
    )


if __name__ == "__main__":
    main()
