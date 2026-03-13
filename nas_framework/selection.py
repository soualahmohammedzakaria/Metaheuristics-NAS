from abc import ABC, abstractmethod
import random
from nas_framework.population import Individual
from nas_framework.mo_utils import assign_rank_and_crowding, pareto_sort_key


class Selection(ABC):
    """Abstract selection operator."""

    @abstractmethod
    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        """Select *n* parents from the population."""
        ...


class TournamentSelection(Selection):
    """Tournament selection with configurable tournament size."""

    def __init__(self, k: int = 3):
        self.k = k

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        assign_rank_and_crowding(individuals, objective_directions)
        parents = []
        for _ in range(n):
            candidates = random.sample(individuals, min(self.k, len(individuals)))
            winner = min(candidates, key=pareto_sort_key)
            parents.append(winner)
        return parents


class RouletteWheelSelection(Selection):
    """Compatibility alias that now delegates to Pareto tournament selection."""

    def __init__(self, k: int = 3):
        self._delegate = TournamentSelection(k=k)

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        return self._delegate.select(individuals, n, objective_directions)

