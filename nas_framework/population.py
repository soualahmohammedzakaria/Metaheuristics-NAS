from __future__ import annotations
from nas_framework.search_space import SearchSpace
from nas_framework.evaluator import Evaluator
from nas_framework.mo_utils import pareto_front as compute_pareto_front


class Individual:
    """A single candidate solution with its fitness."""

    def __init__(self, genotype,
                 fitness: tuple[float, ...] | None = None,
                 metadata: dict | None = None):
        self.genotype = genotype
        self.fitness = fitness
        self.metadata = metadata or {}
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    @property
    def architecture(self):
        return self.genotype

    @architecture.setter
    def architecture(self, value) -> None:
        self.genotype = value

    @property
    def objectives(self) -> tuple[float, ...] | None:
        return self.fitness

    @objectives.setter
    def objectives(self, value: tuple[float, ...] | None) -> None:
        self.fitness = value

    def __repr__(self) -> str:
        return f"Individual({self.genotype}, fitness={self.fitness})"


class Population:
    """Manages a collection of individuals."""

    def __init__(self, search_space: SearchSpace | None = None, evaluator: Evaluator | None = None,
                 size: int = 20):
        self.search_space = search_space
        self.evaluator = evaluator
        self.size = size
        self.individuals: list[Individual] = []

    def initialize(self) -> None:
        """Create random individuals and evaluate them."""
        if self.search_space is None or self.evaluator is None:
            raise ValueError("Population.initialize requires both search_space and evaluator")
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

