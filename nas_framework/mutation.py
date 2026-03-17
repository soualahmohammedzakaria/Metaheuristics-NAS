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


class ABCNeighborSampler(SinglePointMutation):
    """1-operation neighbor sampler for the HiveNAS ABC search strategy.

    HiveNAS (Shahawy & Benkhelifa, arXiv:2211.10250v2, §3.2.3) adapts the
    classical continuous ABC neighbor formula (Eq. 2) to the discrete NAS
    space using the convention from White et al. [2021]:

        "Two architectures are neighbors if there is a 1-operation
         (i.e. layer) difference between them."

    This is identical to SinglePointMutation (flip one random edge to a
    different op), so ABCNeighborSampler simply exposes a ``sample_neighbor``
    alias to match the ABC terminology used in ABCSearchStrategy.
    """

    def sample_neighbor(self, individual: Individual) -> Individual:
        """Return a new Individual differing in exactly one operation."""
        return self.mutate(individual)