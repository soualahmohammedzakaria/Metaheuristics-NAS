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