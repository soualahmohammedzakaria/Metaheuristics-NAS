from abc import ABC, abstractmethod
import random
from copy import deepcopy
from nas_framework.population import Individual


class Crossover(ABC):
    """Abstract crossover operator."""

    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Produce one child from two parents."""
        ...


class UniformCrossover(Crossover):
    """Each gene is randomly picked from one of the two parents."""

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child_geno = [
            random.choice([g1, g2])
            for g1, g2 in zip(parent1.genotype, parent2.genotype)
        ]
        return Individual(child_geno)


class SinglePointCrossover(Crossover):
    """Classic single-point crossover."""

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        point = random.randint(1, len(parent1.genotype) - 1)
        child_geno = parent1.genotype[:point] + parent2.genotype[point:]
        return Individual(child_geno)

