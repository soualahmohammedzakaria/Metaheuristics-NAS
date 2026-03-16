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
        archive_layer = []
        for ind in pareto_front:
            # Handle standard GA Individual or IPPSO PSOParticle
            if hasattr(ind, 'genotype'):
                archive_layer.append(Individual(ind.genotype[:], ind.fitness))
            elif hasattr(ind, 'personal_best_position'):
                # Store the tracked position, fitness is alias of current_fitness
                archive_layer.append(Individual(ind.personal_best_position[:], ind.fitness))
        self.pareto_archive.append(archive_layer)
