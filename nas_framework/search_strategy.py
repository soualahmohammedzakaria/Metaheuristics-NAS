from abc import ABC, abstractmethod
import math
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
from nas_framework.search_space import SearchSpace
from nas_framework.mo_utils import pareto_front as compute_pareto_front
from nas_framework.mo_utils import exact_pareto_front_2d
from nas_framework.mo_utils import dominates


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
    """Random search baseline  samples new architectures each iteration."""

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


class BruteForceParetoSearch:
    """Exhaustive strategy that returns the optimal Pareto front on finite spaces."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 history: History | None = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []

    def run(self) -> list[Individual]:
        if not hasattr(self.search_space, "all_genotypes"):
            raise TypeError(
                "BruteForceParetoSearch requires a finite search space exposing "
                "all_genotypes()."
            )

        individuals: list[Individual] = []
        for genotype in self.search_space.all_genotypes():
            fitness = self.evaluator.evaluate(genotype)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(genotype)
            individuals.append(Individual(genotype[:], fitness, metadata=metadata))

        self.population = individuals
        self.evaluations = len(individuals)
        self.generations = 1

        front = compute_pareto_front(individuals, self.evaluator.objective_directions)
        self.history.record(
            generation=0,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=front,
        )
        return front


class SkylineSearch:
    """Exact Pareto strategy for 2 objectives using sort+sweep skyline."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 history: History | None = None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.history = history or History()
        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []

    def run(self) -> list[Individual]:
        if not hasattr(self.search_space, "all_genotypes"):
            raise TypeError(
                "SkylineSearch requires a finite search space exposing "
                "all_genotypes()."
            )
        if len(self.evaluator.objective_directions) != 2:
            raise ValueError("SkylineSearch supports exactly 2 objectives.")

        individuals: list[Individual] = []
        for genotype in self.search_space.all_genotypes():
            fitness = self.evaluator.evaluate(genotype)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(genotype)
            elif hasattr(self.evaluator.benchmark, "get_metadata"):
                metadata = self.evaluator.benchmark.get_metadata(genotype)
            individuals.append(Individual(genotype[:], fitness, metadata=metadata))

        self.population = individuals
        self.evaluations = len(individuals)
        self.generations = 1

        directions_2d = (
            self.evaluator.objective_directions[0],
            self.evaluator.objective_directions[1],
        )
        front = exact_pareto_front_2d(individuals, directions_2d)
        self.history.record(
            generation=0,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=front,
        )
        return front
 
# Implementation of the method described in the article Multi-Objective
#  White Shark Optimizer for Global Optimization and Rural Sports-Facilities
#  Location Problem
class MOWSOSearch:
    """Multi-Objective White Shark Optimizer with archive true-distance pruning.

    This follows the paper workflow with:
    - White-shark speed and movement update phases
    - External non-dominated archive
    - True-distance based archive truncation
    - Guide selection as archive member with maximum true distance
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 50,
        max_iterations: int = 100,
        archive_size: int | None = None,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size or pop_size
        self.history = history or History()

        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []
        self.archive: list[Individual] = []

        # Parameters reported in the referenced WSO/MOWSO setup.
        self.p_min = 0.5
        self.p_max = 1.5
        self.tau = 4.125
        self.f_min = 0.07
        self.f_max = 0.75
        self.a0 = 6.25
        self.a1 = 100.0
        self.a2 = 0.0005

    def _mu(self) -> float:
        # Eq. (5): contraction factor.
        inner = self.tau * self.tau - 4.0 * self.tau
        inner = max(inner, 0.0)
        denom = abs(2.0 - self.tau - math.sqrt(inner))
        if denom <= 1e-12:
            return 1.0
        return 2.0 / denom

    def _p1_p2(self, k: int, K: int) -> tuple[float, float]:
        # Eq. (3) and Eq. (4) schedule used in the paper.
        expo = math.exp(-((4.0 * k) / K) ** 2)
        p1 = self.p_max + (self.p_max - self.p_min) * expo
        p2 = self.p_min + (self.p_max - self.p_min) * expo
        return p1, p2

    def _mv(self, k: int, K: int) -> float:
        # Eq. (11): movement force.
        return 1.0 / (self.a0 + math.exp((K / 2.0 - k) / self.a1))

    def _f(self) -> float:
        # Eq. (10) form used in the paper text.
        return self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)

    def _ss(self, k: int, K: int) -> float:
        # Eq. (14): olfactory/visual intensity.
        return abs(1.0 - math.exp(-self.a2 * k / K))

    def _bounds(self) -> tuple[float, float]:
        return 0.0, float(self.search_space.num_ops - 1)

    def _clip_position(self, pos: list[float]) -> list[float]:
        lower, upper = self._bounds()
        return [min(upper, max(lower, x)) for x in pos]

    def _to_genotype(self, pos: list[float]) -> list[int]:
        lower, upper = self._bounds()
        return [int(min(upper, max(lower, round(x)))) for x in pos]

    def _individual_from_pos(self, pos: list[float]) -> Individual:
        genotype = self._to_genotype(pos)
        fitness = self.evaluator.evaluate(genotype)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(genotype)
        elif hasattr(self.evaluator.benchmark, "get_metadata"):
            metadata = self.evaluator.benchmark.get_metadata(genotype)
        self.evaluations += 1
        return Individual(genotype, fitness, metadata=metadata)

    def _dominates(self, a: Individual, b: Individual) -> bool:
        return dominates(a, b, self.evaluator.objective_directions)

    def _distance(self, a: Individual, b: Individual) -> float:
        if a.fitness is None or b.fitness is None:
            return float("inf")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.fitness, b.fitness)))

    def _true_distance_scores(self, inds: list[Individual]) -> list[float]:
        if len(inds) <= 1:
            return [float("inf")] * len(inds)
        scores: list[float] = []
        for i, ind_i in enumerate(inds):
            dists = [self._distance(ind_i, inds[j]) for j in range(len(inds)) if j != i]
            dists.sort()
            scores.append(sum(dists))
        return scores

    def _archive_update(self, candidates: list[Individual]) -> None:
        # Merge by genotype key to avoid duplicates.
        merged_by_key: dict[tuple[int, ...], Individual] = {}
        for ind in self.archive + candidates:
            merged_by_key[tuple(ind.genotype)] = ind
        merged = list(merged_by_key.values())

        nd = compute_pareto_front(merged, self.evaluator.objective_directions)
        self.archive = nd

        while len(self.archive) > self.archive_size:
            td = self._true_distance_scores(self.archive)
            remove_idx = min(range(len(self.archive)), key=lambda i: td[i])
            del self.archive[remove_idx]

    def _guide_position(self, positions: list[list[float]]) -> list[float]:
        if not self.archive:
            return random.choice(positions)[:]
        td = self._true_distance_scores(self.archive)
        guide = self.archive[max(range(len(self.archive)), key=lambda i: td[i])]
        return [float(x) for x in guide.genotype]

    def run(self) -> list[Individual]:
        K = max(1, int(self.max_iterations))
        mu = self._mu()

        # Initialization
        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        individuals: list[Individual] = []
        for _ in range(self.pop_size):
            geno = self.search_space.random_individual()
            pos = [float(x) for x in geno]
            ind = self._individual_from_pos(pos)
            positions.append(pos)
            velocities.append([0.0] * len(pos))
            individuals.append(ind)

        self.population = individuals
        self.archive = []
        self._archive_update(individuals)

        pbest_pos = [p[:] for p in positions]
        pbest_ind = [Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()) for ind in individuals]

        self.generations = 0
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=individuals,
            pareto_front=self.archive,
        )

        for k in range(1, K + 1):
            p1, p2 = self._p1_p2(k, K)
            mv = self._mv(k, K)
            f = self._f()
            ss = self._ss(k, K)
            guide = self._guide_position(positions)

            next_positions: list[list[float]] = []
            next_individuals: list[Individual] = []

            for i in range(self.pop_size):
                v_idx = random.randint(0, self.pop_size - 1)
                c1 = random.random()
                c2 = random.random()

                cur = positions[i]
                vel = velocities[i]
                pbest_v = pbest_pos[v_idx]

                new_vel = []
                for d in range(len(cur)):
                    step = vel[d] + p1 * (guide[d] - cur[d]) * c1 + p2 * (pbest_v[d] - cur[d]) * c2
                    new_vel.append(mu * step)
                velocities[i] = new_vel

                if random.random() < mv:
                    # Eq. (6), (7), (8), (9): boundary-aware movement.
                    lower, upper = self._bounds()
                    new_pos = []
                    for d in range(len(cur)):
                        a = 1.0 if cur[d] > upper else 0.0
                        b = 1.0 if cur[d] < lower else 0.0
                        w_o = 1.0 if (bool(a) ^ bool(b)) else 0.0
                        new_pos.append(cur[d] * (1.0 - w_o) + upper * a + lower * b)
                else:
                    new_pos = [cur[d] + new_vel[d] / f for d in range(len(cur))]

                if random.random() < ss:
                    r1 = random.random()
                    r2 = random.random()
                    rand_dw = random.random()
                    sign_dir = 1.0 if r2 >= 0.5 else -1.0
                    d_w = [abs(rand_dw * (guide[d] - cur[d])) for d in range(len(cur))]
                    w_prime = [guide[d] + r1 * d_w[d] * sign_dir for d in range(len(cur))]
                    denom = 2.0 * max(random.random(), 1e-12)
                    new_pos = [(new_pos[d] + w_prime[d]) / denom for d in range(len(cur))]

                new_pos = self._clip_position(new_pos)
                ind_new = self._individual_from_pos(new_pos)

                if self._dominates(ind_new, pbest_ind[i]):
                    pbest_pos[i] = new_pos[:]
                    pbest_ind[i] = Individual(ind_new.genotype[:], ind_new.fitness, ind_new.metadata.copy())
                elif (not self._dominates(pbest_ind[i], ind_new)) and random.random() < 0.5:
                    pbest_pos[i] = new_pos[:]
                    pbest_ind[i] = Individual(ind_new.genotype[:], ind_new.fitness, ind_new.metadata.copy())

                next_positions.append(new_pos)
                next_individuals.append(ind_new)

            positions = next_positions
            individuals = next_individuals
            self.population = individuals
            self._archive_update(individuals)

            self.generations = k
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=individuals,
                pareto_front=self.archive,
            )

        return [Individual(ind.genotype[:], ind.fitness, ind.metadata.copy()) for ind in self.archive]

