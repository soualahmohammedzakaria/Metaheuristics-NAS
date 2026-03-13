from abc import ABC, abstractmethod
import random
from copy import deepcopy
from nas_framework.population import Individual
from nas_framework.search_space import SearchSpace


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

