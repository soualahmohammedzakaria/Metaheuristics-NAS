from abc import ABC, abstractmethod
from nas_framework.population import Individual
from nas_framework.mo_utils import take_pareto_best


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

