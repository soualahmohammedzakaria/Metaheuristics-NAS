import random 
from abc import ABC, abstractmethod
from nas_framework.population import Population, ABCPopulation, FoodSource, Individual, ABCPopulation, FoodSource
from nas_framework.selection import Selection,TournamentSelection, NeighborSelection, GuidanceSelection
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
                 history: History | None = None):

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
        """Reset exhausted sources and count extra evaluations."""
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
        self.w_perf     = w_perf
        self.gamma      = gamma
        self.beta0      = beta0
        self.max_chances = max_chances
        self._mutation  = mutation
        self._crossover = crossover

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

        With probability β(r) per gene: adopt target's gene value.
        Remaining genes stay or receive a random mutation with prob α=0.2.
        This models the FA position update in a discrete space.
        """
        import random
        r   = self._hamming(firefly, target)
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