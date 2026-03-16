import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.ip_pso_population import PSOPopulation
from nas_framework.ip_evaluator import IPPSOEvaluator
from nas_framework.ip_pso_search import IPPSOSearch


def main():
    population = PSOPopulation()
    evaluator = IPPSOEvaluator(mock=True)  # Use mock for testing
    search = IPPSOSearch(population, evaluator, max_generations=5)  # Small for testing

    print("Starting IPPSO search...")
    final_pop = search.run()
    print(f"Generations: {search.generations}, Evaluations: {search.evaluations}")
    print(f"Best fitness: {final_pop.global_best_fitness}")
    print(f"Best position: {final_pop.global_best_position}")


if __name__ == "__main__":
    main()