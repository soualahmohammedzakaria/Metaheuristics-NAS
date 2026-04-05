from abc import ABC, abstractmethod
import math
import random
from nas_framework.population import Population, Individual
from nas_framework.selection import Selection
from nas_framework.variation import Variation, MutationOnlyVariation, CrossoverMutationVariation
from nas_framework.crossover import Crossover
from nas_framework.mutation import Mutation
from nas_framework.replacement import Replacement
from nas_framework.evaluator import Evaluator, DvolverEvaluator
from nas_framework.termination import Termination, MaxEvaluationsTermination
from nas_framework.termination import TerminationCriteria
from nas_framework.history import History
from nas_framework.search_space import SearchSpace, NASSearchSpace
from nas_framework.mo_utils import pareto_front as compute_pareto_front
from nas_framework.mo_utils import exact_pareto_front_2d
from nas_framework.mo_utils import dominates
from nas_framework.mo_utils import fast_non_dominated_sort_max, compute_crowding_distance
from nas_framework.selection import BinaryTournamentSelection
from nas_framework.variation import DvolverVariation
from nas_framework.crossover import DvolverUniformCrossover
from nas_framework.mutation import UniformMutation
from nas_framework.replacement import DvolverReplacement


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


class DvolverSearchStrategy:
    """Dvolver multi-objective NAS search loop."""

    def __init__(
        self,
        population_size: int = 32,
        crossover_prob: float = 0.1,
        mutation_prob: float = 0.1,
        search_space: NASSearchSpace | None = None,
        evaluator: DvolverEvaluator | None = None,
        termination: TerminationCriteria | None = None,
    ):
        self.population_size = population_size
        self.search_space = search_space or NASSearchSpace()
        self.evaluator = evaluator or DvolverEvaluator()
        self.termination = termination or TerminationCriteria(max_generations=50)

        self.population = Population(size=population_size)
        self.history = History()
        self.selection = BinaryTournamentSelection()
        self.variation = DvolverVariation(
            crossover=DvolverUniformCrossover(self.search_space, crossover_prob=crossover_prob),
            mutation=UniformMutation(mutation_prob=mutation_prob),
        )
        self.replacement = DvolverReplacement()

    @staticmethod
    def _assign_rank_and_crowding(individuals: list[Individual]) -> None:
        fronts = fast_non_dominated_sort_max(individuals)
        for front in fronts:
            compute_crowding_distance(front)

    def _evaluate_population(self, individuals: list[Individual]) -> None:
        for ind in individuals:
            ind.objectives = self.evaluator.evaluate(ind.architecture)

    def run(self) -> History:
        init_architectures = [self.search_space.random_architecture() for _ in range(self.population_size)]
        self.population.individuals = [Individual(arch) for arch in init_architectures]
        self._evaluate_population(self.population.individuals)
        self._assign_rank_and_crowding(self.population.individuals)

        generation = 0
        self.history.update(generation=generation, population=self.population.individuals)

        while not self.termination.should_terminate(self.history, generation):
            parents = self.selection.select_parents(self.population.individuals, self.population_size)
            offspring = self.variation.generate_offspring(parents, self.search_space)

            self._evaluate_population(offspring)

            survivors = self.replacement.select_survivors(
                self.population.individuals,
                offspring,
                self.population_size,
            )
            self.population.individuals = survivors
            self._assign_rank_and_crowding(self.population.individuals)

            generation += 1
            self.history.update(generation=generation, population=self.population.individuals)

        return self.history


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
        max_iterations: int = 300,
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


class MOSHOSearch:
    """Multi-Objective Shark Hunting Optimization for NAS-Bench-201."""

    OPS = [
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    GENE_SIZE = 6
    GENE_VALUES = list(range(len(OPS)))

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        pop_size: int = 50,
        max_iterations: int = 300,
        archive_size: int | None = None,
        e0: float = 1.0,
        e_min: float = 0.1,
        e_max: float = 2.0,
        eta: float = 15.0,
        delta: float = 0.03,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.archive_size = archive_size or pop_size
        self.history = history or History()

        self.e0 = e0
        self.e_min = e_min
        self.e_max = e_max
        self.eta = eta
        self.delta = delta

        if self.search_space.num_edges != self.GENE_SIZE:
            raise ValueError(
                f"MOSHOSearch expects {self.GENE_SIZE} genes, got "
                f"{self.search_space.num_edges}."
            )
        if self.search_space.num_ops != len(self.GENE_VALUES):
            raise ValueError(
                f"MOSHOSearch expects {len(self.GENE_VALUES)} operations, got "
                f"{self.search_space.num_ops}."
            )

        self.base_probs = {
            "patrol": 0.20,
            "scent": 0.25,
            "circle": 0.15,
            "burst": 0.10,
            "crossover": 0.20,
            "scout": 0.10,
        }
        self._op_credit: dict[str, float] = {op: 0.5 for op in self.base_probs}

        self.evaluations: int = 0
        self.generations: int = 0
        self.population: list[Individual] = []
        self.archive: list[Individual] = []
        self._energy: list[float] = []

        self._archive_sampler_dirty = True
        self._archive_sampler_weights: list[float] = []

    def _clone_individual(self, ind: Individual) -> Individual:
        metadata = ind.metadata.copy() if ind.metadata else {}
        return Individual(ind.genotype[:], ind.fitness, metadata=metadata)

    def _evaluate_arch(self, arch: list[int]) -> Individual:
        fitness = self.evaluator.evaluate(arch)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(arch)
        elif hasattr(self.evaluator.benchmark, "get_metadata"):
            metadata = self.evaluator.benchmark.get_metadata(arch)
        self.evaluations += 1
        return Individual(arch[:], fitness, metadata=metadata)

    def _random_arch(self) -> list[int]:
        return self.search_space.random_individual()

    def _mutate_gene(self, arch: list[int], idx: int) -> list[int]:
        new_arch = arch[:]
        choices = [v for v in self.GENE_VALUES if v != new_arch[idx]]
        new_arch[idx] = random.choice(choices)
        return new_arch

    def _crowding_scores(self, inds: list[Individual]) -> list[float]:
        if not inds:
            return []
        if len(inds) <= 2:
            return [float("inf")] * len(inds)

        scores = [0.0 for _ in inds]
        n_obj = len(self.evaluator.objective_directions)
        for obj in range(n_obj):
            direction = self.evaluator.objective_directions[obj]
            order = sorted(
                range(len(inds)),
                key=lambda i: inds[i].fitness[obj] * direction,
            )
            scores[order[0]] = float("inf")
            scores[order[-1]] = float("inf")

            min_v = inds[order[0]].fitness[obj] * direction
            max_v = inds[order[-1]].fitness[obj] * direction
            if max_v == min_v:
                continue

            for k in range(1, len(order) - 1):
                idx = order[k]
                if scores[idx] == float("inf"):
                    continue
                prev_v = inds[order[k - 1]].fitness[obj] * direction
                next_v = inds[order[k + 1]].fitness[obj] * direction
                scores[idx] += (next_v - prev_v) / (max_v - min_v)
        return scores

    def _archive_update(self, candidate: Individual) -> bool:
        for archived in self.archive:
            if dominates(archived, candidate, self.evaluator.objective_directions):
                return False

        kept: list[Individual] = []
        changed = False
        for archived in self.archive:
            if dominates(candidate, archived, self.evaluator.objective_directions):
                changed = True
                continue
            kept.append(archived)

        if any(
            tuple(archived.genotype) == tuple(candidate.genotype)
            and archived.fitness == candidate.fitness
            for archived in kept
        ):
            return changed

        kept.append(self._clone_individual(candidate))
        self.archive = kept
        self._archive_sampler_dirty = True
        return True

    def _truncate_archive(self) -> None:
        if len(self.archive) <= self.archive_size:
            return

        scores = self._crowding_scores(self.archive)
        order = sorted(range(len(self.archive)), key=lambda i: scores[i], reverse=True)
        keep = set(order[: self.archive_size])
        self.archive = [self.archive[i] for i in range(len(self.archive)) if i in keep]
        self._archive_sampler_dirty = True

    def _refresh_archive_sampler(self) -> None:
        if not self.archive:
            self._archive_sampler_weights = []
            self._archive_sampler_dirty = False
            return

        scores = self._crowding_scores(self.archive)
        self._archive_sampler_weights = [
            10.0 if math.isinf(score) else max(1e-3, score) for score in scores
        ]
        self._archive_sampler_dirty = False

    def _sample_archive(self) -> Individual:
        if self._archive_sampler_dirty or len(self._archive_sampler_weights) != len(self.archive):
            self._refresh_archive_sampler()
        if not self.archive:
            raise ValueError("Cannot sample from an empty archive.")
        return random.choices(self.archive, weights=self._archive_sampler_weights, k=1)[0]

    def _patrol(self, arch: list[int]) -> list[int]:
        u = max(1e-9, random.random())
        k = int(min(self.GENE_SIZE, max(2, round((u ** -0.35) % self.GENE_SIZE))))
        k = min(self.GENE_SIZE, max(2, k))
        idxs = random.sample(range(self.GENE_SIZE), k=k)
        new_arch = arch[:]
        for idx in idxs:
            new_arch = self._mutate_gene(new_arch, idx)
        return new_arch

    def _scent(self, arch: list[int]) -> list[int]:
        target = self._sample_archive().genotype if self.archive else self._random_arch()
        partner = random.choice(self.population).genotype

        new_arch = arch[:]
        for j in range(self.GENE_SIZE):
            r = random.random()
            if r < 0.55:
                new_arch[j] = target[j]
            elif r < 0.75:
                new_arch[j] = partner[j]
            elif r < 0.80:
                new_arch[j] = random.choice(self.GENE_VALUES)
        return new_arch

    def _circle(self, arch: list[int]) -> list[int]:
        k = 1 if random.random() < 0.7 else 2
        idxs = random.sample(range(self.GENE_SIZE), k=k)
        new_arch = arch[:]
        for idx in idxs:
            new_arch = self._mutate_gene(new_arch, idx)
        return new_arch

    def _burst(self, arch: list[int]) -> list[int]:
        target = self._sample_archive().genotype if self.archive else self._random_arch()

        k = max(1, int(round(0.33 * self.GENE_SIZE)))
        copy_idx = set(random.sample(range(self.GENE_SIZE), k=k))

        new_arch = [0] * self.GENE_SIZE
        for j in range(self.GENE_SIZE):
            if j in copy_idx:
                new_arch[j] = target[j]
            else:
                new_arch[j] = arch[j] if random.random() < 0.65 else random.choice(self.GENE_VALUES)
        return new_arch

    def _crossover(self, arch: list[int]) -> list[int]:
        if self.archive and random.random() < 0.55:
            partner = self._sample_archive().genotype
        else:
            partner = random.choice(self.population).genotype

        child = [arch[j] if random.random() < 0.5 else partner[j] for j in range(self.GENE_SIZE)]
        if random.random() < 0.3:
            idx = random.randrange(self.GENE_SIZE)
            child[idx] = random.choice([v for v in self.GENE_VALUES if v != child[idx]])
        return child

    def _scout(self) -> list[int]:
        return self._random_arch()

    def _choose_operator(self, probs: dict[str, float]) -> str:
        ops = list(probs.keys())
        weights = list(probs.values())
        return random.choices(ops, weights=weights, k=1)[0]

    def _normalize_probs(self, probs: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in probs.values())
        if total <= 0.0:
            n = max(1, len(probs))
            return {key: 1.0 / n for key in probs}
        return {key: max(0.0, value) / total for key, value in probs.items()}

    def _operator_probability_adaptation(
        self,
        probs: dict[str, float],
        imp_rate: float,
        progress: float,
    ) -> dict[str, float]:
        p = max(0.0, min(1.0, progress))
        improved = max(0.0, min(1.0, imp_rate))

        adjusted = dict(probs)
        adjusted["scout"] *= 1.0 + (1.0 - p) * (1.0 - improved)
        adjusted["patrol"] *= 1.0 + 0.6 * (1.0 - p)
        adjusted["scent"] *= 1.0 + 0.6 * p
        adjusted["crossover"] *= 1.0 + 0.5 * p
        adjusted["burst"] *= 1.0 + 0.3 * max(0.0, 0.2 - improved)
        adjusted["circle"] *= 1.0 + 0.25 * improved

        return self._normalize_probs(adjusted)

    def _credit_biased_probs(self, probs: dict[str, float], progress: float) -> dict[str, float]:
        if not probs:
            return probs

        p = max(0.0, min(1.0, progress))
        mean_credit = sum(self._op_credit.get(op, 0.5) for op in probs) / max(1, len(probs))
        strength = 0.35 + 0.35 * p

        adjusted: dict[str, float] = {}
        for op, base_p in probs.items():
            credit = self._op_credit.get(op, 0.5)
            multiplier = 1.0 + strength * (credit - mean_credit)
            adjusted[op] = max(1e-9, base_p * multiplier)

        adjusted = self._normalize_probs(adjusted)

        floor = 0.04
        n = len(adjusted)
        if floor * n < 1.0:
            rem = 1.0 - floor * n
            adjusted = {op: floor + rem * prob for op, prob in adjusted.items()}
            adjusted = self._normalize_probs(adjusted)

        return adjusted

    def _update_operator_credit(
        self,
        op: str,
        accepted: bool,
        archive_improved: bool,
        dominates_current: bool,
    ) -> None:
        reward = 0.0
        if archive_improved:
            reward = 1.0
        elif dominates_current and accepted:
            reward = 0.8
        elif accepted:
            reward = 0.35

        old = self._op_credit.get(op, 0.5)
        alpha = 0.08
        self._op_credit[op] = (1.0 - alpha) * old + alpha * reward

    def _replace_low_energy(self, shark_idx: int) -> None:
        if len(self.archive) >= 2:
            a = self._sample_archive().genotype
            b = self._sample_archive().genotype
            child = [a[j] if random.random() < 0.5 else b[j] for j in range(self.GENE_SIZE)]
            for j in range(self.GENE_SIZE):
                if random.random() < 0.15:
                    child[j] = random.choice(self.GENE_VALUES)
            arch = child
        else:
            arch = self._scout()

        result = self._evaluate_arch(arch)
        self._archive_update(result)
        self.population[shark_idx] = result
        self._energy[shark_idx] = max(self.e0 * 0.5, self.e_min)

    def _accept(
        self,
        current: Individual,
        candidate: Individual,
        archive_improved: bool,
        progress: float,
    ) -> bool:
        if dominates(candidate, current, self.evaluator.objective_directions):
            return True
        if archive_improved:
            return True
        if not dominates(current, candidate, self.evaluator.objective_directions):
            accept_prob = 0.4 * (1.0 - 0.7 * progress)
            return random.random() < accept_prob
        return False

    def run(self) -> list[Individual]:
        self.population = []
        self.archive = []
        self._energy = []
        self._archive_sampler_dirty = True
        self._archive_sampler_weights = []
        self.evaluations = 0
        self.generations = 0

        for _ in range(self.pop_size):
            arch = self._random_arch()
            ind = self._evaluate_arch(arch)
            self.population.append(ind)
            self._energy.append(self.e0)
            self._archive_update(ind)

        if not self.population:
            return []

        self._truncate_archive()
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population,
            pareto_front=self.archive,
        )

        recent_improvements: list[int] = []

        total_iters = max(1, int(self.max_iterations))
        for k in range(1, total_iters + 1):

            archive_change_count = 0
            improvement_trials = 0
            progress = min(1.0, k / total_iters)

            if recent_improvements:
                imp_rate = sum(recent_improvements) / len(recent_improvements)
            else:
                imp_rate = 0.2
            probs = self._operator_probability_adaptation(self.base_probs, imp_rate, progress)
            probs = self._credit_biased_probs(probs, progress)

            for i in range(len(self.population)):
                shark = self.population[i]
                op = self._choose_operator(probs)

                if op == "patrol":
                    candidate_arch = self._patrol(shark.genotype)
                elif op == "scent":
                    candidate_arch = self._scent(shark.genotype)
                elif op == "circle":
                    candidate_arch = self._circle(shark.genotype)
                elif op == "burst":
                    candidate_arch = self._burst(shark.genotype)
                elif op == "crossover":
                    candidate_arch = self._crossover(shark.genotype)
                else:
                    candidate_arch = self._scout()

                if candidate_arch == shark.genotype:
                    idx = random.randrange(self.GENE_SIZE)
                    candidate_arch = self._mutate_gene(candidate_arch, idx)

                candidate = self._evaluate_arch(candidate_arch)
                archive_improved = self._archive_update(candidate)
                dominates_current = dominates(candidate, shark, self.evaluator.objective_directions)
                accepted = self._accept(shark, candidate, archive_improved, progress)

                if accepted:
                    energy_bonus = 0.2 if archive_improved else 0.05
                    self.population[i] = candidate
                    self._energy[i] = min(self.e_max, self._energy[i] + energy_bonus)
                else:
                    self._energy[i] = max(0.0, self._energy[i] - self.delta)

                if archive_improved:
                    archive_change_count += 1
                improvement_trials += 1
                self._update_operator_credit(op, accepted, archive_improved, dominates_current)

                if self._energy[i] < self.e_min:
                    self._replace_low_energy(i)

            self._truncate_archive()

            recent_improvements.append(1 if (improvement_trials > 0 and archive_change_count > 0) else 0)
            if len(recent_improvements) > 20:
                recent_improvements.pop(0)

            self.generations = k
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=self.population,
                pareto_front=self.archive,
            )

        return [self._clone_individual(ind) for ind in self.archive]

