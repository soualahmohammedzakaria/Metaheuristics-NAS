from abc import ABC, abstractmethod
import random
from copy import deepcopy
import numpy as np
from nas_framework.population import Individual
from nas_framework.search_space import SearchSpace, NASSearchSpace


class Mutation(ABC):
    """Abstract mutation operator."""

    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        """Return a mutated copy of the individual."""
        ...


class SinglePointMutation(Mutation):
    """Flip one random gene to a different operation."""

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def mutate(self, individual: Individual) -> Individual:
        geno = deepcopy(individual.genotype)
        pos = random.randint(0, self.search_space.num_edges - 1)
        choices = [op for op in range(self.search_space.num_ops) if op != geno[pos]]
        geno[pos] = random.choice(choices)
        return Individual(geno)


class BitFlipMutation(Mutation):
    """Each gene has independent probability *rate* of being re-sampled."""

    def __init__(self, search_space: SearchSpace, rate: float = 0.15):
        self.search_space = search_space
        self.rate = rate

    def mutate(self, individual: Individual) -> Individual:
        geno = deepcopy(individual.genotype)
        for i in range(len(geno)):
            if random.random() < self.rate:
                geno[i] = random.randint(0, self.search_space.num_ops - 1)
        return Individual(geno)


class UniformMutation:
    """Uniform per-gene mutation for Dvolver (Appendix A.1)."""

    def __init__(self, mutation_prob: float = 0.1):
        self.mutation_prob = mutation_prob

    def mutate(self, individual: Individual, search_space: NASSearchSpace) -> Individual:
        genes = np.array(search_space.encode(individual.architecture), copy=True)
        for pos in range(genes.shape[0]):
            if random.random() < self.mutation_prob:
                genes[pos] = search_space.random_value_for_gene(pos)
        return Individual(search_space.decode(genes))

