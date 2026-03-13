from __future__ import annotations

from dataclasses import dataclass

from nas_framework.population import Individual


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
            Individual(ind.genotype[:], ind.fitness) for ind in pareto_front
        ])

