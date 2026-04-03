"""NSGA-II baseline for multi-objective NAS on NAS-Bench-201 lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random

from benchmark import (
    EvalResult,
    GENE_SIZE,
    GENE_VALUES,
    NASBench201Lookup,
    crowding_distance,
    fast_nondominated_sort,
    hypervolume_2d,
    mutate_gene,
    random_arch,
)


@dataclass
class Individual:
    arch: Tuple[int, ...]
    result: EvalResult


class NSGA2:
    """Standard NSGA-II with uniform crossover and per-gene mutation."""

    def __init__(
        self,
        evaluator: NASBench201Lookup,
        seed: int,
        pop_size: int = 50,
        max_generations: int = 300,
        max_evals: int = 15000,
        crossover_prob: float = 0.9,
        mutation_prob: float = 1.0 / GENE_SIZE,
    ):
        self.evaluator = evaluator
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_evals = max_evals
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.population: List[Individual] = []
        self.hv_curve: List[Tuple[int, float]] = []
        self.ref_accuracy = 0.0
        self.ref_flops = 1.0

    def _update_refs(self, result: EvalResult) -> None:
        self.ref_accuracy = min(self.ref_accuracy, 0.0)
        self.ref_flops = max(self.ref_flops, result.flops * 1.1)

    def _population_hv(self) -> float:
        values = [ind.result for ind in self.population]
        fronts = fast_nondominated_sort(values)
        nd = [values[i] for i in fronts[0]] if fronts else []
        return hypervolume_2d(nd, self.ref_accuracy, self.ref_flops)

    def _binary_tournament(self, ranks: List[int], crowd: List[float]) -> int:
        i, j = self.rng.sample(range(self.pop_size), 2)
        if ranks[i] < ranks[j]:
            return i
        if ranks[j] < ranks[i]:
            return j
        if crowd[i] > crowd[j]:
            return i
        if crowd[j] > crowd[i]:
            return j
        return i if self.rng.random() < 0.5 else j

    def _uniform_crossover(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.rng.random() > self.crossover_prob:
            return a
        child = [a[g] if self.rng.random() < 0.5 else b[g] for g in range(GENE_SIZE)]
        return tuple(child)

    def _mutate(self, arch: Tuple[int, ...]) -> Tuple[int, ...]:
        new_arch = tuple(arch)
        for g in range(GENE_SIZE):
            if self.rng.random() < self.mutation_prob:
                new_arch = mutate_gene(new_arch, g, self.rng)
        return new_arch

    def _compute_rank_and_crowding(self, values: Sequence[EvalResult]) -> Tuple[List[int], List[float]]:
        fronts = fast_nondominated_sort(values)
        ranks = [0] * len(values)
        crowd = [0.0] * len(values)
        for rank, front in enumerate(fronts):
            for i in front:
                ranks[i] = rank
            cd = crowding_distance([values[i] for i in front])
            for local_idx, global_idx in enumerate(front):
                crowd[global_idx] = cd[local_idx]
        return ranks, crowd

    def _select_next_population(self, combined: List[Individual]) -> List[Individual]:
        values = [ind.result for ind in combined]
        fronts = fast_nondominated_sort(values)

        next_pop: List[Individual] = []
        for front in fronts:
            if len(next_pop) + len(front) <= self.pop_size:
                next_pop.extend(combined[i] for i in front)
                continue

            remaining = self.pop_size - len(next_pop)
            front_vals = [values[i] for i in front]
            cd = crowding_distance(front_vals)
            order = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
            for idx in order[:remaining]:
                next_pop.append(combined[front[idx]])
            break

        return next_pop

    def run(self) -> Dict[str, object]:
        self.population = []
        for _ in range(self.pop_size):
            arch = random_arch(self.rng)
            result = self.evaluator.evaluate(arch)
            self._update_refs(result)
            self.population.append(Individual(arch=arch, result=result))

        self.hv_curve = [(self.evaluator.eval_count, self._population_hv())]

        for _ in range(self.max_generations):
            if self.evaluator.eval_count >= self.max_evals:
                break

            values = [ind.result for ind in self.population]
            ranks, crowd = self._compute_rank_and_crowding(values)

            offspring: List[Individual] = []
            while len(offspring) < self.pop_size and self.evaluator.eval_count < self.max_evals:
                p1 = self.population[self._binary_tournament(ranks, crowd)]
                p2 = self.population[self._binary_tournament(ranks, crowd)]

                child_arch = self._uniform_crossover(p1.arch, p2.arch)
                child_arch = self._mutate(child_arch)
                child_result = self.evaluator.evaluate(child_arch)
                self._update_refs(child_result)
                offspring.append(Individual(arch=child_arch, result=child_result))

            combined = self.population + offspring
            self.population = self._select_next_population(combined)
            self.hv_curve.append((self.evaluator.eval_count, self._population_hv()))

        values = [ind.result for ind in self.population]
        fronts = fast_nondominated_sort(values)
        nd_front = [self.population[i] for i in fronts[0]] if fronts else []

        return {
            "front": [(ind.arch, ind.result) for ind in nd_front],
            "population": [(ind.arch, ind.result) for ind in self.population],
            "hv_curve": list(self.hv_curve),
            "evaluations": self.evaluator.eval_count,
        }
