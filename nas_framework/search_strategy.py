from abc import ABC, abstractmethod
import random
from nas_framework.population import Population, Individual
from nas_framework.selection import Selection
from nas_framework.variation import Variation, MutationOnlyVariation, CrossoverMutationVariation
from nas_framework.crossover import Crossover
from nas_framework.mutation import Mutation
from nas_framework.replacement import Replacement
from nas_framework.evaluator import Evaluator
from nas_framework.termination import Termination, MaxEvaluationsTermination
from nas_framework.history import History


class SearchStrategy(ABC):
    """Abstract search strategy that orchestrates population-based components."""

    def __init__(self, population: Population, selection: Selection,
                 variation: Variation, replacement: Replacement,
                 evaluator: Evaluator, termination: Termination | None = None,
                 history: History | None = None, budget: int = 500):
        self.population = population
        self.selection = selection
        self.variation = variation
        self.replacement = replacement
        self.evaluator = evaluator
        self.termination = termination or MaxEvaluationsTermination(budget)
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0

    @abstractmethod
    def run(self) -> Population:
        """Execute the search and return the final population."""
        ...

    def _record_history(self) -> None:
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population.individuals,
            pareto_front=self.population.pareto_front(),
        )

    def _evaluate_offspring(self, offspring: list[Individual]) -> None:
        for child in offspring:
            child.fitness = self.evaluator.evaluate(child.genotype)
            metadata = {}
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                metadata = self.population.search_space.metadata_from_genotype(child.genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(child.genotype)
            child.metadata = metadata
            self.evaluations += 1
            if self.termination.should_stop(self.evaluations, self.generations):
                break


class GeneticAlgorithm(SearchStrategy):
    """Pareto-based GA: selection â†’ variation â†’ replacement loop."""

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # Selection
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            # Variation
            offspring = self.variation.generate(parents, self.population.size)
            self._evaluate_offspring(offspring)

            # Replacement
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class EvolutionStrategy(SearchStrategy):
    """(mu, lambda) ES: mutation-only variation."""

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500, n_offspring: int = 40,
                 termination: Termination | None = None,
                 history: History | None = None):
        variation = MutationOnlyVariation(mutation)
        super().__init__(population, selection, variation,
                         replacement, evaluator,
                         termination=termination, history=history,
                         budget=budget)
        self.n_offspring = n_offspring

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            offspring = self.variation.generate(parents, self.n_offspring)
            self._evaluate_offspring(offspring)

            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


class RandomSearch(SearchStrategy):
    """Random search baseline â€” samples new architectures each iteration."""

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500, termination: Termination | None = None,
                 history: History | None = None):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(population, selection, variation, replacement,
                         evaluator, termination=termination,
                         history=history, budget=budget)

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            geno = self.population.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            self.evaluations += 1
            metadata = {}
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                metadata = self.population.search_space.metadata_from_genotype(geno)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(geno)
            child = Individual(geno, fit, metadata=metadata)

            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                [child],
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population

class BiPopulationUniformSamplingMOEADStrategy(SearchStrategy):
    """Bi-population MOEA/D with uniform sampling and Tchebycheff decomposition."""

    def __init__(self, population: Population, crossover: Crossover,
                 mutation: Mutation, evaluator: Evaluator,
                 budget: int = 500, neighborhood_size: int = 5,
                 neighborhood_mating_prob: float = 0.9,
                 max_replacements: int = 2,
                 termination: Termination | None = None,
                 history: History | None = None):
       
        from nas_framework.selection import TournamentSelection
        from nas_framework.replacement import ElitistReplacement

        super().__init__(
            population=population,
            selection=TournamentSelection(k=2),
            variation=CrossoverMutationVariation(crossover, mutation),
            replacement=ElitistReplacement(),
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )
        self.crossover = crossover
        self.mutation = mutation
        self.neighborhood_size = max(2, neighborhood_size)
        self.neighborhood_mating_prob = max(0.0, min(1.0, neighborhood_mating_prob))
        self.max_replacements = max(1, max_replacements)

    @staticmethod
    def _objective_to_maximization(fitness: tuple[float, ...],
                                   directions: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(value * direction for value, direction in zip(fitness, directions))

    def _init_weight_vectors(self, n_subproblems: int, n_objectives: int) -> list[tuple[float, ...]]:
        if n_objectives == 2:
            if n_subproblems == 1:
                return [(0.5, 0.5)]
            return [
                (i / (n_subproblems - 1), 1.0 - (i / (n_subproblems - 1)))
                for i in range(n_subproblems)
            ]

        vectors: list[tuple[float, ...]] = []
        for _ in range(n_subproblems):
            raw = [random.random() for _ in range(n_objectives)]
            total = sum(raw)
            if total == 0:
                vectors.append(tuple(1.0 / n_objectives for _ in range(n_objectives)))
            else:
                vectors.append(tuple(v / total for v in raw))
        return vectors

    def _init_neighborhoods(self, lambdas: list[tuple[float, ...]]) -> list[list[int]]:
        neighborhoods: list[list[int]] = []
        t = min(self.neighborhood_size, len(lambdas))
        for i, wi in enumerate(lambdas):
            distances: list[tuple[float, int]] = []
            for j, wj in enumerate(lambdas):
                d = sum((a - b) ** 2 for a, b in zip(wi, wj))
                distances.append((d, j))
            distances.sort(key=lambda x: x[0])
            neighborhoods.append([idx for _, idx in distances[:t]])
        return neighborhoods

    @staticmethod
    def _decomposition_value(fitness_max: tuple[float, ...],
                             weight: tuple[float, ...],
                             ideal_point: tuple[float, ...]) -> float:
        eps = 1e-12
        vals = [
            max(weight[i], eps) * abs(ideal_point[i] - fitness_max[i])
            for i in range(len(fitness_max))
        ]
        return max(vals)

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        if not self.population.individuals:
            return self.population

        n_subproblems = len(self.population.individuals)
        n_objectives = len(self.population.individuals[0].fitness)
        directions = self.evaluator.objective_directions

        lambdas = self._init_weight_vectors(n_subproblems, n_objectives)
        neighborhoods = self._init_neighborhoods(lambdas)

        transformed = [
            self._objective_to_maximization(ind.fitness, directions)
            for ind in self.population.individuals
        ]
        ideal_point = tuple(max(vals[k] for vals in transformed) for k in range(n_objectives))

        while not self.termination.should_stop(self.evaluations, self.generations):
            for i in range(n_subproblems):
                if random.random() < self.neighborhood_mating_prob:
                    candidate_indices = neighborhoods[i]
                else:
                    candidate_indices = list(range(n_subproblems))

                if len(candidate_indices) == 1:
                    p1_idx = p2_idx = candidate_indices[0]
                else:
                    p1_idx, p2_idx = random.sample(candidate_indices, 2)

                p1 = self.population.individuals[p1_idx]
                p2 = self.population.individuals[p2_idx]

                child = self.crossover.crossover(p1, p2)
                child = self.mutation.mutate(child)
                self._evaluate_offspring([child])

                if child.fitness is None:
                    continue

                child_max = self._objective_to_maximization(child.fitness, directions)
                ideal_point = tuple(
                    max(ideal_point[k], child_max[k]) for k in range(n_objectives)
                )

                updated = 0
                for j in neighborhoods[i]:
                    incumbent = self.population.individuals[j]
                    incumbent_max = self._objective_to_maximization(incumbent.fitness, directions)
                    g_old = self._decomposition_value(incumbent_max, lambdas[j], ideal_point)
                    g_new = self._decomposition_value(child_max, lambdas[j], ideal_point)
                    if g_new <= g_old:
                        self.population.individuals[j] = Individual(
                            child.genotype[:],
                            child.fitness,
                            metadata=child.metadata.copy(),
                        )
                        updated += 1
                        if updated >= self.max_replacements:
                            break

                if self.termination.should_stop(self.evaluations, self.generations):
                    break

            self.generations += 1
            self._record_history()

        return self.population


# Backward-compatible alias for older imports.
MOEADStrategy = BiPopulationUniformSamplingMOEADStrategy

