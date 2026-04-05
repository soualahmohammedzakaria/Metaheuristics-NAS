from abc import ABC, abstractmethod
from nas_framework.population import Individual
from nas_framework.mo_utils import take_pareto_best, fast_non_dominated_sort_max, compute_crowding_distance


class Replacement(ABC):
    """Abstract replacement / survivor-selection operator."""

    @abstractmethod
    def replace(self, population: list[Individual],
            offspring: list[Individual], pop_size: int,
            objective_directions: tuple[int, ...]) -> list[Individual]:
        """Merge offspring into population and return the next generation."""
        ...


class ElitistReplacement(Replacement):
    """Keep the best *pop_size* Pareto-ranked individuals from parents + offspring."""

    def replace(self, population: list[Individual],
        offspring: list[Individual], pop_size: int,
        objective_directions: tuple[int, ...]) -> list[Individual]:
        combined = population + offspring
        return take_pareto_best(combined, pop_size, objective_directions)


class GenerationalReplacement(Replacement):
    """Replace population with offspring using Pareto elitism."""

    def __init__(self, elitism: bool = True):
        self.elitism = elitism

    def replace(self, population: list[Individual],
                offspring: list[Individual], pop_size: int,
                objective_directions: tuple[int, ...]) -> list[Individual]:
        if self.elitism:
            elite = take_pareto_best(population, 1, objective_directions)
            offspring.extend(elite)
        return take_pareto_best(offspring, pop_size, objective_directions)


class DvolverReplacement:
    """Algorithm 1 survivor selection from Dvolver: select N from 2N."""

    def select_survivors(self, parents: list[Individual], offspring: list[Individual], N: int) -> list[Individual]:
        combined = parents + offspring
        fronts = fast_non_dominated_sort_max(combined)
        for front in fronts:
            compute_crowding_distance(front)

        selected: list[Individual] = []
        rem = N
        i = 0
        while rem > 0 and i < len(fronts):
            front_i = sorted(fronts[i], key=lambda ind: ind.crowding_distance, reverse=True)
            if len(front_i) <= rem:
                selected.extend(front_i)
                rem -= len(front_i)
                i += 1
            else:
                selected.extend(front_i[:rem])
                rem = 0

        return selected

