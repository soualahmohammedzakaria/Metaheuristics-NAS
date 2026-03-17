from __future__ import annotations
from nas_framework.search_space import SearchSpace
from nas_framework.evaluator import Evaluator
from nas_framework.mo_utils import pareto_front as compute_pareto_front


class Individual:
    """A single candidate solution with its fitness."""

    def __init__(self, genotype: list[int],
                 fitness: tuple[float, ...] | None = None,
                 metadata: dict | None = None):
        self.genotype = genotype
        self.fitness = fitness
        self.metadata = metadata or {}
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def __repr__(self) -> str:
        return f"Individual({self.genotype}, fitness={self.fitness})"


class Population:
    """Manages a collection of individuals."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 size: int = 20):
        self.search_space = search_space
        self.evaluator = evaluator
        self.size = size
        self.individuals: list[Individual] = []

    def initialize(self) -> None:
        """Create random individuals and evaluate them."""
        self.individuals = []
        for _ in range(self.size):
            geno = self.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(geno)
            self.individuals.append(Individual(geno, fit, metadata=metadata))

    def best(self) -> Individual:
        """Return an individual from the first Pareto front (highest crowding)."""
        front = self.pareto_front()
        if front:
            return max(front, key=lambda ind: ind.crowding_distance)
        return self.individuals[0]

    def pareto_front(self) -> list[Individual]:
        return compute_pareto_front(self.individuals, self.evaluator.objective_directions)

    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)


# ---------------------------------------------------------------------------
# ABC-specific extensions  (HiveNAS, Shahawy & Benkhelifa, arXiv:2211.10250v2)
# ---------------------------------------------------------------------------

def _weighted_score(fitness: tuple[float, ...], directions: tuple[int, ...]) -> float:
    """Scalar score used for greedy keep-best comparison across objectives."""
    return sum(v * d for v, d in zip(fitness, directions))


class FoodSource:
    """Wraps an Individual with an ABC abandonment trial counter.

    Each food source tracks how many consecutive evaluations have been made
    without improvement.  Once trial_count reaches the abandonment_limit the
    scout phase resets it to a new random position.
    """

    def __init__(self, individual: Individual):
        self.individual = individual
        self.trial_count: int = 0

    def reset(self, individual: Individual) -> None:
        self.individual = individual
        self.trial_count = 0

    def update(self, candidate: Individual, directions: tuple[int, ...]) -> bool:
        """Greedy selection: keep candidate if it improves the scalar score.

        Returns True if the source was updated (improvement found).
        """
        if candidate.fitness is None:
            self.trial_count += 1
            return False
        if self.individual.fitness is None:
            self.individual = candidate
            self.trial_count = 0
            return True
        if _weighted_score(candidate.fitness, directions) > _weighted_score(self.individual.fitness, directions):
            self.individual = candidate
            self.trial_count = 0
            return True
        self.trial_count += 1
        return False


class ABCPopulation(Population):
    """Population of FoodSources for the ABC algorithm.

    Parameters
    ----------
    search_space, evaluator, size : same as Population.
    abandonment_limit : int
        Consecutive failed trials before a scout resets the source.
        Defaults to 10 * num_edges when not specified.
    """

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 size: int = 10, abandonment_limit: int | None = None):
        super().__init__(search_space, evaluator, size)
        self.abandonment_limit: int = (
            abandonment_limit if abandonment_limit is not None
            else 10 * search_space.num_edges
        )
        self.food_sources: list[FoodSource] = []

    def initialize(self) -> None:
        """Scout phase: randomly sample and evaluate all food sources."""
        self.food_sources = []
        self.individuals = []
        for _ in range(self.size):
            geno = self.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(geno)
            ind = Individual(geno, fit, metadata=metadata)
            self.food_sources.append(FoodSource(ind))
            self.individuals.append(ind)

    def sync_individuals(self) -> None:
        """Keep self.individuals aligned with current food-source states."""
        self.individuals = [fs.individual for fs in self.food_sources]

    def scout_reset(self, directions: tuple[int, ...]) -> int:
        """Reset exhausted sources; return the number of resets performed."""
        resets = 0
        for fs in self.food_sources:
            if fs.trial_count >= self.abandonment_limit:
                geno = self.search_space.random_individual()
                fit = self.evaluator.evaluate(geno)
                metadata = {}
                if hasattr(self.search_space, "metadata_from_genotype"):
                    metadata = self.search_space.metadata_from_genotype(geno)
                fs.reset(Individual(geno, fit, metadata=metadata))
                resets += 1
        return resets