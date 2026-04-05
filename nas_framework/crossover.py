from abc import ABC, abstractmethod
import random
from copy import deepcopy
import numpy as np
from nas_framework.population import Individual
from nas_framework.search_space import NASSearchSpace


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


class DvolverUniformCrossover:
    """Uniform crossover as described in Dvolver Appendix A.1."""

    def __init__(self, search_space: NASSearchSpace, crossover_prob: float = 0.1):
        self.search_space = search_space
        self.crossover_prob = crossover_prob

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        g1 = self.search_space.encode(parent1.architecture)
        g2 = self.search_space.encode(parent2.architecture)
        c1 = np.array(g1, copy=True)
        c2 = np.array(g2, copy=True)

        for i in range(c1.shape[0]):
            if random.random() < self.crossover_prob:
                c1[i], c2[i] = c2[i], c1[i]

        child1_arch = self.search_space.decode(c1)
        child2_arch = self.search_space.decode(c2)
        return Individual(child1_arch), Individual(child2_arch)

