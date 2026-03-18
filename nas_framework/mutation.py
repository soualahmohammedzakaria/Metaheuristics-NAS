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

        

# ---------------------------------------------------------------------------
# PSO-specific mutation  (MOIPSO, Shao et al., Scientific Reports 2025)
# ---------------------------------------------------------------------------

class GaussianMutation(Mutation):
    """Adaptive Gaussian mutation strategy from MOIPSO .

    In the original paper (continuous space), the position perturbation is:
        dx = N(0, sigma) * (ub - lb)
        sigma = 0.1 * (1 - t / T)

    In our discrete NAS search space (genes are integers in [0, num_ops)),
    sigma is reinterpreted as a per-gene re-sampling probability: each gene
    is independently replaced by a uniformly random operation with probability
    sigma.  This preserves the intended annealing behaviour — broad random
    exploration early in the search, negligible disruption near convergence —
    while being compatible with discrete genotypes.

    Parameters
    ----------
    search_space : SearchSpace
    get_progress : callable () -> float
        Returns a float in [0, 1] representing t / T (current fraction of
        the total budget consumed).  PSOSearchStrategy passes a lambda that
        reads self.evaluations / self.budget.
    """

    def __init__(self, search_space: SearchSpace,
                 get_progress=None):
        self.search_space = search_space
        # get_progress() -> float in [0,1]; defaults to 0 (no annealing)
        self._get_progress = get_progress if get_progress is not None else (lambda: 0.0)

    def mutate(self, individual: Individual) -> Individual:
        t_frac = max(0.0, min(1.0, self._get_progress()))
        sigma = 0.1 * (1.0 - t_frac)           # shrinks from 0.1 → 0
        geno = deepcopy(individual.genotype)
        for i in range(len(geno)):
            if random.random() < sigma:
                geno[i] = random.randint(0, self.search_space.num_ops - 1)
        return Individual(geno)
