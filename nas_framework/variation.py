from __future__ import annotations

from abc import ABC, abstractmethod
import random

from nas_framework.crossover import Crossover
from nas_framework.mutation import Mutation
from nas_framework.population import Individual


class Variation(ABC):
    """Abstract variation operator (crossover + mutation or mutation-only)."""

    @abstractmethod
    def generate(self, parents: list[Individual], n_offspring: int) -> list[Individual]:
        """Generate offspring from selected parents."""
        ...


class CrossoverMutationVariation(Variation):
    """Generate offspring via pairwise crossover followed by mutation."""

    def __init__(self, crossover: Crossover, mutation: Mutation):
        self.crossover = crossover
        self.mutation = mutation

    def generate(self, parents: list[Individual], n_offspring: int) -> list[Individual]:
        offspring: list[Individual] = []
        if not parents:
            return offspring

        while len(offspring) < n_offspring:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = self.crossover.crossover(p1, p2)
            child = self.mutation.mutate(child)
            offspring.append(child)
        return offspring


class MutationOnlyVariation(Variation):
    """Generate offspring by mutating sampled parents."""

    def __init__(self, mutation: Mutation):
        self.mutation = mutation

    def generate(self, parents: list[Individual], n_offspring: int) -> list[Individual]:
        offspring: list[Individual] = []
        if not parents:
            return offspring

        while len(offspring) < n_offspring:
            p = random.choice(parents)
            child = self.mutation.mutate(p)
            offspring.append(child)
        return offspring

