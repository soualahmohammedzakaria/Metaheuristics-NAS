from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from nas_framework.population import Individual
from nas_framework.mo_utils import (
    pareto_front as compute_pareto_front,
    compute_hypervolume,
)


@dataclass
class HistoryEntry:
    generation: int
    evaluations: int
    population_size: int
    pareto_front_size: int


class History:
    """Stores compact run history for analysis and plotting."""

    def __init__(self):
        self.entries: list[HistoryEntry] = []
        self.pareto_archive: list[list[Individual]] = []
        self.all_individuals: list[Individual] = []
        self.hypervolume_history: list[float] = []
        self.pareto_front_history: list[list[Individual]] = []
        self.evaluation_count: int = 0

    def record(self, generation: int, evaluations: int,
               population: list[Individual], pareto_front: list[Individual]) -> None:
        self.entries.append(
            HistoryEntry(
                generation=generation,
                evaluations=evaluations,
                population_size=len(population),
                pareto_front_size=len(pareto_front),
            )
        )
        # Store a detached snapshot so later changes do not mutate history.
        self.pareto_archive.append([
            Individual(deepcopy(ind.architecture), ind.objectives, metadata=deepcopy(ind.metadata))
            for ind in pareto_front
        ])

    def update(self, generation: int, population: list[Individual]) -> None:
        detached_population: list[Individual] = []
        for ind in population:
            clone = Individual(deepcopy(ind.architecture), ind.objectives, metadata=deepcopy(ind.metadata))
            clone.rank = ind.rank
            clone.crowding_distance = ind.crowding_distance
            detached_population.append(clone)

        self.all_individuals.extend(detached_population)
        self.evaluation_count = len(self.all_individuals)

        current_front = compute_pareto_front(detached_population, directions=(1, 1))
        self.pareto_front_history.append(current_front)
        hv = compute_hypervolume(current_front, reference_point=(0.0, 0.0))
        self.hypervolume_history.append(hv)

        self.record(
            generation=generation,
            evaluations=self.evaluation_count,
            population=detached_population,
            pareto_front=current_front,
        )

    def get_current_pareto_front(self) -> list[Individual]:
        return compute_pareto_front(self.all_individuals, directions=(1, 1))

    def has_converged(self, patience: int = 10) -> bool:
        if patience <= 0:
            return False
        if len(self.hypervolume_history) <= patience:
            return False

        recent = self.hypervolume_history[-(patience + 1):]
        best_before_last = max(recent[:-1])
        return recent[-1] <= best_before_last

