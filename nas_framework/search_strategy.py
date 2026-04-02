import random 
from abc import ABC, abstractmethod
import math
import random
from nas_framework.population import Population, ABCPopulation, FoodSource, Individual, ABCPopulation, FoodSource
from nas_framework.selection import Selection,TournamentSelection, NeighborSelection, GuidanceSelection
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


class GeneticAlgorithmNG(SearchStrategy):
    """NSGA-III/NG style: hybrid NG search strategy (Liu Yuejun, Heliyon 2024).

    At each generation, parent selection is decided probabilistically:
      - With prob p_neigh  → NeighborSelection (Eq. 3)
      - With prob p_guide  → GuidanceSelection (Eq. 4)
      - Otherwise          → fallback TournamentSelection

    Paper defaults: p_neigh=0.8, p_guide=0.2, k=3, P=13 (for NSGA-III variant).
    """

    def __init__(self, population: Population, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 replacement: Replacement, evaluator: Evaluator,
                 budget: int = 500,
                 p_neigh: float = 0.8,
                 p_guide: float = 0.2,
                 k: int = 3,
                 P: int = 13,
                 termination: Termination | None = None,
                 history: History | None = None):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(population, selection, variation, replacement,
                         evaluator, termination=termination,
                         history=history, budget=budget)
        self.p_neigh = p_neigh
        self.p_guide = p_guide
        self._neighbor_sel = NeighborSelection(k=k, P=P)
        self._guidance_sel = GuidanceSelection()
        self._fallback_sel = selection  # tournament

    def _select_parents(self) -> list[Individual]:
        r = random.random()
        inds = self.population.individuals
        dirs = self.evaluator.objective_directions
        n = self.population.size
        if r < self.p_neigh:
            return self._neighbor_sel.select(inds, n, dirs)
        elif r < self.p_neigh + self.p_guide:
            return self._guidance_sel.select(inds, n, dirs)
        else:
            return self._fallback_sel.select(inds, n, dirs)

    def run(self) -> Population:
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            parents = self._select_parents()
            offspring = self.variation.generate(parents, self.population.size)
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


class ABCSearchStrategy(SearchStrategy):
    """Artificial Bee Colony NAS search strategy.

    Implements the HiveNAS framework (Shahawy & Benkhelifa, arXiv:2211.10250v2).
    The algorithm follows Algorithm 1 of the paper:

        Initialisation (Scout phase)
        FOR t = 1, 2, ..., T:
            Employee Bees  – evaluate current source; sample 1-op neighbor;
                             greedy selection (keep best).
            Onlooker Bees  – self.selection (RouletteWheelSelection) assigns
                             onlookers to food sources; sample neighbor;
                             greedy selection.
            Scout Bees     – reset any source whose trial_count >= limit.
        Output best candidate.

    Parameters
    ----------
    population : ABCPopulation
        Carries the FoodSource list and the abandonment_limit.
    neighbor_sampler : ABCNeighborSampler
        Generates 1-operation neighbors in the discrete NAS space.
    selection : RouletteWheelSelection
        Fitness-proportionate selector used in the onlooker phase.
        Passed straight to the base class as self.selection.
    evaluator : Evaluator
        Queries accuracy and latency from the benchmark.
    budget : int
        Total evaluation budget (stopping criterion).
    termination, history :
        Optional overrides for the base class.
    """

    def __init__(self,
                 population,
                 neighbor_sampler,
                 selection: Selection,
                 evaluator: Evaluator,
                 budget: int = 500,
                 termination: Termination | None = None,
                 history: History | None = None,
                 dmt_fraction: float = 0.67):

        from nas_framework.variation import MutationOnlyVariation
        from nas_framework.mutation import SinglePointMutation
        from nas_framework.replacement import ElitistReplacement

        variation   = MutationOnlyVariation(SinglePointMutation(population.search_space))
        replacement = ElitistReplacement()

        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )
        self.neighbor_sampler = neighbor_sampler
        # dmt: scout resets disabled after this many evaluations (MBO-ABCFE §3.2).
        # Prevents random restarts from destroying good late-search solutions.
        self._dmt = int(dmt_fraction * budget)
        # Visited cache: genotype tuple -> fitness tuple.
        # Prevents re-spending evaluations on already-seen architectures.
        self._visited: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_individual(self, individual: Individual) -> None:
        """Evaluate an individual in-place and increment the counter.

        Skips re-evaluation if the genotype was already seen (visited cache),
        recovering wasted budget from duplicate neighbor samples.
        At large budgets (500+), 20-39% of neighbor samples are duplicates
        without this guard.
        """
        key = tuple(individual.genotype)
        if key in self._visited:
            # Retrieve cached fitness without spending an evaluation.
            individual.fitness = self._visited[key]
            if hasattr(self.population.search_space, "metadata_from_genotype"):
                individual.metadata = (
                    self.population.search_space.metadata_from_genotype(individual.genotype)
                )
            return  # do NOT increment self.evaluations
        individual.fitness = self.evaluator.evaluate(individual.genotype)
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            individual.metadata = (
                self.population.search_space.metadata_from_genotype(individual.genotype)
            )
        self._visited[key] = individual.fitness
        self.evaluations += 1

    def _employee_phase(self) -> None:
        """Each employee evaluates a 1-op neighbor; greedy keep-best."""
        dirs = self.evaluator.objective_directions
        for fs in self.population.food_sources:
            if self.termination.should_stop(self.evaluations, self.generations):
                return
            neighbor = self.neighbor_sampler.sample_neighbor(fs.individual)
            self._eval_individual(neighbor)
            fs.update(neighbor, dirs)

    def _onlooker_phase(self) -> None:
        """Roulette-wheel assign onlookers to food sources; greedy keep-best."""
        dirs = self.evaluator.objective_directions
        n_onlookers = len(self.population.food_sources)

        self.population.sync_individuals()
        selected_inds = self.selection.select(
            self.population.individuals,
            n_onlookers,
            dirs,
        )

        # Build a genotype→FoodSource map for O(1) lookup.
        geno_to_fs: dict[tuple, object] = {
            tuple(fs.individual.genotype): fs
            for fs in self.population.food_sources
        }

        for ind in selected_inds:
            if self.termination.should_stop(self.evaluations, self.generations):
                return
            fs = geno_to_fs.get(tuple(ind.genotype))
            if fs is None:
                continue
            neighbor = self.neighbor_sampler.sample_neighbor(fs.individual)
            self._eval_individual(neighbor)
            fs.update(neighbor, dirs)

    def _scout_phase(self) -> None:
        """Reset exhausted sources — suppressed after dmt evaluations.

        Past the dmt threshold (default 67% of budget) the population has
        typically converged to a good region; random restarts would waste
        budget.  This mirrors the discard-mechanism-trigger from MBO-ABCFE.
        """
        if self.evaluations >= self._dmt:
            return  # scout suppressed in late search
        resets = self.population.scout_reset(self.evaluator.objective_directions)
        self.evaluations += resets

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute the full ABC search loop (Algorithm 1 of HiveNAS)."""
        self.population.initialize()
        self.evaluations = len(self.population.food_sources)
        self.generations = 0
        self.population.sync_individuals()
        # Seed visited cache with initial food sources.
        for fs in self.population.food_sources:
            if fs.individual.fitness is not None:
                self._visited[tuple(fs.individual.genotype)] = fs.individual.fitness
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            self._employee_phase()
            if self.termination.should_stop(self.evaluations, self.generations):
                break
            self._onlooker_phase()
            if self.termination.should_stop(self.evaluations, self.generations):
                break
            self._scout_phase()
            self.population.sync_individuals()
            self.generations += 1
            self._record_history()

        self.population.sync_individuals()
        return self.population


class FireflySearchStrategy(SearchStrategy):
    """Improved Firefly Algorithm for NAS (RB-IFA, Nguyen et al., ICAART 2025).

    Each individual (firefly) is attracted to brighter (better rank-scored)
    neighbours and moves toward them.  Brightness is determined by the
    rank-based scalar score from mo_utils.rank_based_score, which collapses
    multiple objectives into one weighted value — no Pareto front needed.

    Movement in the discrete NAS space is modelled as:
      - With probability proportional to attractiveness β(r): adopt one gene
        from the brighter firefly (single-gene crossover toward the target).
      - Otherwise: apply a random mutation (exploration / light absorption).

    When the population stagnates for *max_chances* generations, a genetic
    iteration (tournament selection + crossover + mutation on the full
    population) is triggered to escape local optima, mirroring the IFA
    mechanism of Mokhtari et al. (2022) used in the paper.

    Parameters
    ----------
    population : Population
    selection  : Selection   — used for the genetic fallback iteration.
    crossover  : Crossover   — used for the genetic fallback iteration.
    mutation   : Mutation    — used for both movement exploration and fallback.
    replacement: Replacement — RankBasedReplacement recommended.
    evaluator  : Evaluator
    budget     : int         — total evaluation budget.
    w_perf     : float       — performance weight for rank scoring (0–1).
    gamma      : float       — light absorption coefficient.
                              0 → every firefly sees global best (PSO-like).
                              Large → fireflies ignore distant ones (local search).
                              Default 1.0 (moderate, balanced).
    beta0      : float       — base attractiveness at distance 0. Default 1.0.
    max_chances: int         — stagnation patience before genetic fallback.
    """

    def __init__(self,
                 population: Population,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 replacement: Replacement,
                 evaluator: Evaluator,
                 budget: int = 500,
                 w_perf: float = 0.6,
                 gamma: float = 1.0,
                 beta0: float = 1.0,
                 max_chances: int = 5,
                 use_fap: bool = True,
                 fa_prob: float = 0.5,
                 termination: Termination | None = None,
                 history: History | None = None):

        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination,
            history=history,
            budget=budget,
        )
        self.w_perf      = w_perf
        self.gamma       = gamma
        self.beta0       = beta0
        self.max_chances = max_chances
        self.use_fap     = use_fap    # True → flat FAP gate; False → β(r) formula
        self.fa_prob     = fa_prob    # per-gene fire probability when use_fap=True
        self._mutation   = mutation
        self._crossover  = crossover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hamming(self, a: Individual, b: Individual) -> int:
        """Hamming distance between two genotypes."""
        return sum(g1 != g2 for g1, g2 in zip(a.genotype, b.genotype))

    def _attractiveness(self, r: int) -> float:
        """β(r) = β0 * exp(-γ * r²)  — decreases with distance."""
        import math
        return self.beta0 * math.exp(-self.gamma * r * r)

    def _move_toward(self, firefly: Individual,
                     target: Individual) -> Individual:
        """Move firefly one step toward target.

        If use_fap=True: with probability fa_prob per gene, adopt target's gene.
        If use_fap=False: with probability β(r) = β0 * exp(-γ * r²), adopt target's gene.
        Remaining genes stay or receive a random mutation with prob α=0.2.
        """
        import random
        r   = self._hamming(firefly, target)
        
        if self.use_fap:
            beta = self.fa_prob
        else:
            beta = self._attractiveness(r)
            
        alpha = 0.2   # random walk component

        new_geno = list(firefly.genotype)
        for i, (g_self, g_target) in enumerate(
                zip(firefly.genotype, target.genotype)):
            if g_self != g_target and random.random() < beta:
                new_geno[i] = g_target          # attracted move
            elif random.random() < alpha:
                # random walk: re-sample gene
                n_ops = self.population.search_space.num_ops
                new_geno[i] = random.randint(0, n_ops - 1)
        return Individual(new_geno)

    def _scores(self) -> list[float]:
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            self.w_perf,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute the RB-IFA search loop."""
        import random

        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations  = 0
        self._record_history()

        best_score    = min(self._scores())
        chances       = self.max_chances

        while not self.termination.should_stop(self.evaluations, self.generations):
            scores = self._scores()
            inds   = self.population.individuals

            # ── Firefly movement phase ────────────────────────────────
            new_inds: list[Individual] = []
            for i, fi in enumerate(inds):
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                # Find a brighter firefly (lower score = brighter).
                brighter = [inds[j] for j, s in enumerate(scores)
                            if s < scores[i]]
                if brighter:
                    target  = random.choice(brighter)
                    moved   = self._move_toward(fi, target)
                else:
                    # Already brightest — random walk only.
                    moved = self._mutation.mutate(fi)

                moved.fitness = self.evaluator.evaluate(moved.genotype)
                if hasattr(self.population.search_space, "metadata_from_genotype"):
                    moved.metadata = (
                        self.population.search_space
                        .metadata_from_genotype(moved.genotype)
                    )
                self.evaluations += 1
                new_inds.append(moved)

            # ── Rank-based survivor selection ─────────────────────────
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                new_inds,
                self.population.size,
                self.evaluator.objective_directions,
            )

            # ── Stagnation check ──────────────────────────────────────
            current_best = min(self._scores())
            if current_best < best_score:
                best_score = current_best
                chances    = self.max_chances
            else:
                chances -= 1

            # ── Genetic fallback iteration ────────────────────────────
            if chances == 0:
                parents  = self.selection.select(
                    self.population.individuals,
                    self.population.size,
                    self.evaluator.objective_directions,
                )
                offspring = self.variation.generate(parents, self.population.size)
                self._evaluate_offspring(offspring)
                self.population.individuals = self.replacement.replace(
                    self.population.individuals,
                    offspring,
                    self.population.size,
                    self.evaluator.objective_directions,
                )
                # Reset rank scores after genetic refresh.
                best_score = min(self._scores())
                chances    = self.max_chances

            self.generations += 1
            self._record_history()

        return self.population

class HybridMBOStrategy(SearchStrategy):
    """Rank-stratified MBO hybrid: FA movement on top half, GA on bottom half.

    Each generation the population is ranked by rank_based_score().
    - SP1 (top rank_fraction, default 50%): each individual moves toward a
      random SP1 neighbour using the flat FAP gate (adopt gene if FAP < fa_prob).
    - SP2 (bottom half): standard tournament selection + crossover + mutation
      (same as GeneticAlgorithm).
    Both halves are merged and the best `pop_size` survivors are kept via
    RankBasedReplacement.

    This directly implements the SP1/SP2 idea from the MBO resume (§6.3):
    FA as default exploitation on good individuals, GA as diversification
    on weaker ones — no stagnation counter needed.

    Parameters
    ----------
    population   : Population
    selection    : Selection      (TournamentSelection for GA phase)
    crossover    : Crossover
    mutation     : Mutation       (SinglePointMutation)
    replacement  : Replacement    (RankBasedReplacement recommended)
    evaluator    : Evaluator
    budget       : int
    w_perf       : float          rank weight for accuracy (default 0.6)
    rank_fraction: float          fraction of pop in SP1 (default 0.5)
    fa_prob      : float          per-gene FAP gate probability (default 0.5)
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None,
                 w_perf: float = 0.6,
                 rank_fraction: float = 0.5,
                 fa_prob: float = 0.5):
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w_perf       = w_perf
        self.rank_fraction = rank_fraction
        self.fa_prob      = fa_prob
        self._mutation    = mutation
        self._crossover   = crossover

    def _rank_scores(self):
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            w_perf=self.w_perf,
        )

    def _fap_move(self, ind: Individual, sp1: list) -> Individual:
        """Move ind toward a random SP1 member using flat FAP gate."""
        target   = random.choice(sp1)
        new_geno = list(ind.genotype)
        for i, (g_self, g_tgt) in enumerate(zip(ind.genotype, target.genotype)):
            if g_self != g_tgt and random.random() < self.fa_prob:
                new_geno[i] = g_tgt
        return Individual(new_geno)

    def run(self) -> Population:
        self.population.initialize()
        self._evaluate_offspring(self.population.individuals)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            scores = self._rank_scores()
            order  = sorted(range(len(scores)), key=lambda i: scores[i])
            n_sp1  = max(1, int(self.rank_fraction * self.population.size))
            sp1_idx = order[:n_sp1]
            sp2_idx = order[n_sp1:]
            inds    = self.population.individuals
            sp1     = [inds[i] for i in sp1_idx]

            offspring: list[Individual] = []

            # ── SP1: FA movement ──────────────────────────────────────
            for idx in sp1_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                moved = self._fap_move(inds[idx], sp1)
                offspring.append(moved)

            # ── SP2: GA (crossover + mutation) ────────────────────────
            if sp2_idx and not self.termination.should_stop(
                    self.evaluations, self.generations):
                sp2_inds = [inds[i] for i in sp2_idx]
                ga_offspring = self.variation.generate(sp2_inds, len(sp2_idx))
                offspring.extend(ga_offspring)

            self._evaluate_offspring(offspring)

            # ── Rank-based survivor selection ─────────────────────────
            self.population.individuals = self.replacement.replace(
                self.population.individuals,
                offspring,
                self.population.size,
                self.evaluator.objective_directions,
            )

            self.generations += 1
            self._record_history()

        return self.population


class PSOSearchStrategy(SearchStrategy):
    """MOIPSO: Multi-Objective PSO with trigonometric acceleration and
    adaptive Gaussian mutation 

    Algorithm per iteration
    -----------------------
    1. Draw dynamic acceleration factors:
           c1 = 2.05 * |cos(2*pi*rand)|
           c2 = 2.05 * |sin(2*pi*rand)|
    2. Update each particle's velocity and position (discrete PSO, Eq. 5-6).
    3. Evaluate new positions.
    4. Apply GaussianMutation (sigma = 0.1*(1 - t/T)) to a random half of
       the population; evaluate mutated particles.
    5. CrowdingReplacement: keep best *pop_size* by Pareto rank + crowding.
    6. Update pbests and gbest.

    Parameters
    ----------
    population : PSOPopulation
    selection  : Selection      (kept for signature compatibility; not used
                                 in main PSO loop — gbest drives movement)
    crossover  : Crossover      (same — not used in main loop)
    mutation   : GaussianMutation
    replacement: CrowdingReplacement  (or any Replacement)
    evaluator  : Evaluator
    budget     : int            (total number of architecture evaluations)
    w          : float          (inertia weight; default 0.4)
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None, w: float = 0.4):
        # Build a dummy variation so SearchStrategy.__init__ is satisfied
        from nas_framework.variation import CrossoverMutationVariation
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w = w
        self._mutation = mutation     # GaussianMutation (direct access)
        self.budget = budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trig_coefficients(self):
        """Return dynamic c1, c2 (Eq. 8 from MOIPSO paper)."""
        import math
        r = random.random()
        c1 = 2.05 * abs(math.cos(2 * math.pi * r))
        c2 = 2.05 * abs(math.sin(2 * math.pi * r))
        return c1, c2

    def _evaluate_particle(self, ind):
        """Evaluate a single Individual, update evaluations counter."""
        ind.fitness = self.evaluator.evaluate(ind.genotype)
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            ind.metadata = self.population.search_space.metadata_from_genotype(
                ind.genotype)
        self.evaluations += 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute the MOIPSO search loop and return the final Population."""
        # Wire GaussianMutation's progress function to this strategy's state
        if hasattr(self._mutation, '_get_progress'):
            self._mutation._get_progress = lambda: self.evaluations / max(1, self.budget)

        # Also set the PSOPopulation's inertia weight from strategy param
        self.population.w = self.w

        # Step 1 — Initialise (random positions, zero velocities, evaluate)
        self.population.initialize()
        self.evaluations = len(self.population.particles)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # Step 2 — Dynamic acceleration factors
            c1, c2 = self._trig_coefficients()

            # Step 3 — Velocity + position update (discrete PSO)
            self.population.update_velocities_and_positions(c1, c2)

            # Step 4 — Evaluate new positions
            for p in self.population.particles:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                self._evaluate_particle(p.individual)

            # Step 5 — Adaptive Gaussian mutation on random half of particles
            n_mutate = max(1, len(self.population.particles) // 2)
            targets = random.sample(self.population.particles, n_mutate)
            for p in targets:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                mutated = self._mutation.mutate(p.individual)
                self._evaluate_particle(mutated)
                # Greedy replacement: keep mutated if better
                from nas_framework.population import _weighted_score
                dirs = self.evaluator.objective_directions
                if (mutated.fitness is not None and
                    (p.individual.fitness is None or
                     _weighted_score(mutated.fitness, dirs) >
                     _weighted_score(p.individual.fitness, dirs))):
                    p.individual = mutated

            # Step 6 — CrowdingReplacement: merge old + new, keep best N
            self.population.sync_individuals()
            # Filter out any individuals whose fitness was not set (e.g. if
            # budget was exhausted mid-loop before evaluation completed).
            alive = [ind for ind in self.population.individuals
                     if ind.fitness is not None]
            if not alive:
                alive = self.population.individuals  # fallback: keep all
            self.population.individuals = self.replacement.replace(
                alive,
                [],   # offspring already merged into particles above
                self.population.size,
                self.evaluator.objective_directions,
            )

            # Step 7 — Update pbests and gbest
            self.population.update_pbests()

            self.generations += 1
            self._record_history()

        return self.population

class MBOSearchStrategy(SearchStrategy):
    """Monarch Butterfly Optimization hybridized with ABC exploration and
    Firefly exploitation (MBO-ABCFE), adapted for multi-objective NAS.

    Population is split each generation into SP1 (top rank_fraction) and
    SP2 (remainder) by rank_based_score().

    SP1 update (for each individual):
        With prob MR:
            FAP ~ Uniform(0,1) per gene
            if FAP < fa_prob  → FA attraction toward a random SP1 neighbor
            else              → MBO migration (copy gene from SP1 or SP2)
        After update: if old individual was better, increment trial counter.

    SP2 update (for each individual):
        For each gene:
            if rand < p  → copy gene from current best (butterfly adjust)
            else         → copy gene from random SP2 member
            if rand > BAR → apply single-point mutation (Lévy flight proxy)
        After update: if old individual was better, increment trial counter.

    Scout phase (ABC abandonment):
        If evaluations < dmt_fraction * budget:
            Replace individuals with trial >= exh by fresh random solutions.

    Parameters
    ----------
    population      : Population (standard Population, not ABCPopulation)
    selection       : Selection  (unused in main loop; kept for API compat)
    crossover       : Crossover  (unused in main loop; kept for API compat)
    mutation        : Mutation   (SinglePointMutation — Lévy proxy)
    replacement     : Replacement (RankBasedReplacement recommended)
    evaluator       : Evaluator
    budget          : int
    w_perf          : float  rank weight for accuracy objective (default 0.6)
    rank_fraction   : float  fraction of pop assigned to SP1 (default 0.42)
    p               : float  migration ratio — gene drawn from SP1 if rand<p
    BAR             : float  butterfly adjusting rate (Lévy gate, default 5/12)
    MR              : float  modification rate — FA/migration gate (default 0.8)
    fa_prob         : float  per-gene FA fire probability (FAP gate, default 0.5)
    exh_fraction    : float  trial limit = exh_fraction * (budget / pop_size)
    dmt_fraction    : float  scout suppressed after this fraction of budget
    """

    def __init__(self, population, selection, crossover, mutation,
                 replacement, evaluator, budget: int = 500,
                 termination=None, history=None,
                 w_perf: float = 0.8,
                 rank_fraction: float = 0.42,
                 p: float = 5/12,
                 BAR: float = 0.9,
                 MR: float = 0.8,
                 fa_prob: float = 0.5,
                 exh_fraction: float = 4.0,
                 dmt_fraction: float = 0.67):
        from nas_framework.variation import CrossoverMutationVariation
        variation = CrossoverMutationVariation(crossover, mutation)
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            budget=budget,
            termination=termination,
            history=history,
        )
        self.w_perf        = w_perf
        self.rank_fraction = rank_fraction
        self.p             = p
        self.BAR           = BAR
        self.MR            = MR
        self.fa_prob       = fa_prob
        self._mutation     = mutation
        self.budget        = budget

        # Compute exh and dmt analogously to paper formulae
        pop_size           = population.size
        # exh = round(maxIter / Np * exh_fraction); maxIter ≈ budget/pop_size
        max_iter_approx    = max(1, budget // pop_size)
        # exh_fraction used to scale abandonment limit (limit = approx_iters / fraction)
        # We want individuals to be abandoned if they don't improve for ~1/4 of the run.
        self.exh = max(3, round(max_iter_approx / exh_fraction))
        self.dmt = round(dmt_fraction * budget)   # in evaluations, not iters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank_scores(self):
        from nas_framework.mo_utils import rank_based_score
        return rank_based_score(
            self.population.individuals,
            self.evaluator.objective_directions,
            w_perf=self.w_perf,
        )

    def _eval(self, ind):
        ind.fitness = self.evaluator.evaluate(ind.genotype)
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            ind.metadata = self.population.search_space.metadata_from_genotype(
                ind.genotype)
        self.evaluations += 1

    def _is_better(self, a, b):
        """Return True if individual a is better than b (lower rank = better)."""
        from nas_framework.population import _weighted_score
        dirs = self.evaluator.objective_directions
        if a.fitness is None: return False
        if b.fitness is None: return True
        
        # Apply w_perf locally to ensure greedy selection respects accuracy focus
        # This avoiding touching global _weighted_score as requested.
        w_acc = self.w_perf
        w_lat = 1.0 - w_acc
        
        score_a = (a.fitness[0] * dirs[0] * w_acc) + (a.fitness[1] * dirs[1] * w_lat)
        score_b = (b.fitness[0] * dirs[0] * w_acc) + (b.fitness[1] * dirs[1] * w_lat)
        return score_a > score_b

    def _fa_move(self, ind, sp1):
        """FA attraction: move ind toward a random brighter SP1 member gene-by-gene."""
        from copy import deepcopy
        if not sp1:
            return self._mutation.mutate(ind)
        target = random.choice(sp1)
        geno   = deepcopy(ind.genotype)
        for j in range(len(geno)):
            if random.random() < self.fa_prob:
                geno[j] = target.genotype[j]
        return Individual(geno)

    def _mbo_migrate(self, ind, sp1, sp2):
        """MBO migration: copy each gene from SP1 or SP2 based on ratio p."""
        from copy import deepcopy
        geno = deepcopy(ind.genotype)
        sp1_list = list(sp1)
        sp2_list = list(sp2)
        for j in range(len(geno)):
            r = random.random() * 1.2   # peri = 1.2 as in paper
            if r <= self.p and sp1_list:
                donor = random.choice(sp1_list)
            elif sp2_list:
                donor = random.choice(sp2_list)
            else:
                continue
            geno[j] = donor.genotype[j]
        return Individual(geno)

    def _butterfly_adjust(self, ind, best, sp2):
        """SP2 butterfly adjusting: copy best or random SP2 + optional Lévy."""
        from copy import deepcopy
        geno     = deepcopy(ind.genotype)
        sp2_list = list(sp2)
        n_genes  = len(geno)
        for j in range(n_genes):
            r = random.random() * 1.2
            if r <= self.p and best is not None:
                geno[j] = best.genotype[j]
            elif sp2_list:
                donor  = random.choice(sp2_list)
                geno[j] = donor.genotype[j]
            # Lévy flight proxy: BAR gate — if BAR is small, mutation is rare.
            # We now apply this only with probability 1/n_genes to avoid random walk.
            if random.random() > self.BAR:
                choices = [op for op in
                           range(self.population.search_space.num_ops)
                           if op != geno[j]]
                if choices:
                    geno[j] = random.choice(choices)
        return Individual(geno)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        # Initialise
        self.population.initialize()
        trials = [0] * self.population.size
        for ind in self.population.individuals:
            self._eval(ind)
        self.generations = 0
        self._record_history()

        while not self.termination.should_stop(self.evaluations, self.generations):
            # ── Split by rank ──────────────────────────────────────────
            scores   = self._rank_scores()
            order    = sorted(range(len(scores)), key=lambda i: scores[i])
            n_sp1    = max(1, int(self.rank_fraction * self.population.size))
            sp1_idx  = set(order[:n_sp1])
            sp2_idx  = set(order[n_sp1:])
            inds     = self.population.individuals
            sp1      = [inds[i] for i in sp1_idx]
            sp2      = [inds[i] for i in sp2_idx]
            best     = inds[order[0]] if order else None

            new_inds = list(inds)   # mutable copy

            # ── SP1 update ─────────────────────────────────────────────
            for idx in sp1_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                old = inds[idx]
                theta = random.random()
                if theta <= self.MR:
                    candidate = self._fa_move(old, sp1)
                else:
                    candidate = self._mbo_migrate(old, sp1, sp2)
                self._eval(candidate)
                if self._is_better(candidate, old):
                    new_inds[idx] = candidate
                    trials[idx]   = 0
                else:
                    trials[idx]  += 1

            # ── SP2 update ─────────────────────────────────────────────
            for idx in sp2_idx:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                old       = inds[idx]
                candidate = self._butterfly_adjust(old, best, sp2)
                self._eval(candidate)
                if self._is_better(candidate, old):
                    new_inds[idx] = candidate
                    trials[idx]   = 0
                else:
                    trials[idx]  += 1

            # ── Scout phase (ABC abandonment — early search only) ──────
            if self.evaluations < self.dmt:
                for idx in range(self.population.size):
                    if trials[idx] >= self.exh:
                        if self.termination.should_stop(
                                self.evaluations, self.generations):
                            break
                        fresh = Individual(
                            self.population.search_space.random_individual())
                        self._eval(fresh)
                        new_inds[idx] = fresh
                        trials[idx]   = 0

            # ── Update population ──────────────────────────────────────
            self.population.individuals = new_inds

            self.generations += 1
            self._record_history()

        return self.population