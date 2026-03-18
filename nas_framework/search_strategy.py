from abc import ABC, abstractmethod
import random
import math
from typing import Callable
from nas_framework.population import Population, Individual
from nas_framework.selection import Selection
from nas_framework.variation import Variation, MutationOnlyVariation, CrossoverMutationVariation
from nas_framework.crossover import Crossover
from nas_framework.mutation import Mutation
from nas_framework.replacement import Replacement
from nas_framework.evaluator import Evaluator
from nas_framework.termination import Termination, MaxEvaluationsTermination
from nas_framework.history import History
from nas_framework.mo_utils import dominates, assign_rank_and_crowding, pareto_front


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


class NPRASearchStrategy(SearchStrategy):
    """
    Non-dominated Population-based Relative Advantage (NPRA) search strategy.
    
    NPRA is a multi-objective optimization algorithm that:
    - Maintains a non-dominated population (Pareto front)
    - Partitions the population into regions based on reference points
    - Calculates relative advantage of individuals within their region
    - Selects individuals with high relative advantage and dominance
    - Uses adaptive replacement to maintain diversity and convergence
    """

    def __init__(self, population: Population, crossover: Crossover,
                 mutation: Mutation, evaluator: Evaluator,
                 budget: int = 500, n_reference_points: int = 5,
                 neighborhood_size: int = 3,
                 relative_advantage_weight: float = 0.6,
                 termination: Termination | None = None,
                 history: History | None = None):
        """
        Initialize NPRA search strategy.
        
        Args:
            population: Initial population
            crossover: Crossover operator
            mutation: Mutation operator
            evaluator: Evaluator function
            budget: Maximum evaluations
            n_reference_points: Number of reference points for niching
            neighborhood_size: Size of neighborhood around each reference point
            relative_advantage_weight: Weight of relative advantage vs dominance
            termination: Termination condition
            history: History tracker
        """
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
        self.n_reference_points = max(2, n_reference_points)
        self.neighborhood_size = max(1, neighborhood_size)
        self.relative_advantage_weight = max(0.0, min(1.0, relative_advantage_weight))
        self.reference_points: list[tuple[float, ...]] = []
        self.region_assignments: dict[int, int] = {}  # individual idx -> region idx

    def _generate_reference_points(self, n_objectives: int) -> list[tuple[float, ...]]:
        """Generate uniformly distributed reference points in objective space."""
        if n_objectives == 2:
            return [
                (i / max(1, self.n_reference_points - 1), 
                 1.0 - i / max(1, self.n_reference_points - 1))
                for i in range(self.n_reference_points)
            ]
        
        # For higher dimensions, use random normalized vectors
        points = []
        for _ in range(self.n_reference_points):
            raw = [random.random() for _ in range(n_objectives)]
            total = sum(raw)
            if total > 0:
                point = tuple(v / total for v in raw)
            else:
                point = tuple(1.0 / n_objectives for _ in range(n_objectives))
            points.append(point)
        return points

    def _objective_to_maximization(self, fitness: tuple[float, ...],
                                   directions: tuple[int, ...]) -> tuple[float, ...]:
        """Convert fitness to maximization form."""
        return tuple(value * direction for value, direction in zip(fitness, directions))

    def _scalarize_fitness(self, fitness: tuple[float, ...], 
                           reference_point: tuple[float, ...],
                           directions: tuple[int, ...]) -> float:
        """
        Scalarize multi-objective fitness using weighted Tchebycheff decomposition.
        
        This measures how well the individual aligns with a reference point region.
        """
        eps = 1e-12
        fitness_max = self._objective_to_maximization(fitness, directions)
        
        # Weighted Tchebycheff scalarization
        scalarized = max(
            (reference_point[i] + eps) * abs(1.0 - fitness_max[i])
            for i in range(len(fitness))
        )
        return scalarized

    def _assign_regions(self) -> None:
        """Assign each individual to the closest reference point region."""
        self.region_assignments.clear()
        directions = self.evaluator.objective_directions
        
        for idx, ind in enumerate(self.population.individuals):
            if ind.fitness is None:
                self.region_assignments[idx] = 0
                continue
            
            best_region = 0
            best_value = float('inf')
            
            for region_idx, ref_point in enumerate(self.reference_points):
                value = self._scalarize_fitness(ind.fitness, ref_point, directions)
                if value < best_value:
                    best_value = value
                    best_region = region_idx
            
            self.region_assignments[idx] = best_region

    def _calculate_relative_advantage(self, 
                                     individual_idx: int,
                                     region_idx: int) -> float:
        """
        Calculate relative advantage of an individual within its region.
        
        Relative advantage is based on:
        - Dominance over others in the same region
        - Scalarized fitness alignment with reference point
        """
        individual = self.population.individuals[individual_idx]
        directions = self.evaluator.objective_directions
        
        if individual.fitness is None:
            return 0.0
        
        # Count individuals dominated in this region
        dominated_count = 0
        region_members = [
            idx for idx, region in self.region_assignments.items()
            if region == region_idx
        ]
        
        for other_idx in region_members:
            if other_idx != individual_idx:
                other = self.population.individuals[other_idx]
                if dominates(individual, other, directions):
                    dominated_count += 1
        
        # Scalarized fitness contribution
        ref_point = self.reference_points[region_idx]
        scalarized = self._scalarize_fitness(individual.fitness, ref_point, directions)
        
        # Combine dominance count and scalarized fitness
        dominance_advantage = dominated_count / max(1, len(region_members) - 1)
        scalarized_advantage = 1.0 / (1.0 + scalarized)  # Invert: smaller scalarized is better
        
        # Weighted combination
        advantage = (
            self.relative_advantage_weight * dominance_advantage +
            (1.0 - self.relative_advantage_weight) * scalarized_advantage
        )
        return advantage

    def _select_high_advantage_parents(self, n_parents: int) -> list[Individual]:
        """Select parents based on relative advantage within their regions."""
        if not self.population.individuals:
            return []
        
        # Assign individuals to regions
        self._assign_regions()
        
        # Calculate relative advantage for each individual
        advantages = {}
        for idx in range(len(self.population.individuals)):
            region = self.region_assignments[idx]
            advantage = self._calculate_relative_advantage(idx, region)
            advantages[idx] = advantage
        
        # Select parents using advantage-based selection
        candidates = []
        for idx, advantage in advantages.items():
            # Include in tournament with probability proportional to advantage
            if random.random() < min(1.0, advantage + 0.1):  # +0.1 to ensure everyone has a chance
                candidates.append(idx)
        
        # Ensure we have at least n_parents candidates
        if len(candidates) < n_parents:
            # Fill remaining slots with tournament selection
            remaining = n_parents - len(candidates)
            for _ in range(remaining):
                idx = random.choice(list(range(len(self.population.individuals))))
                candidates.append(idx)
        
        # Sample parents from candidates
        selected_indices = random.sample(candidates, min(len(candidates), n_parents))
        return [self.population.individuals[idx] for idx in selected_indices]

    def _replace_with_advantage(self, offspring: list[Individual]) -> None:
        """
        Replace population members using both dominance and relative advantage.
        
        Replacement strategy:
        - Offspring that dominate population members replace them
        - Otherwise, replace worst individuals in same region if offspring has high advantage
        - Always maintain Pareto front diversity
        """
        if not offspring:
            return
        
        directions = self.evaluator.objective_directions
        self._assign_regions()
        
        for child in offspring:
            if child.fitness is None:
                continue
            
            child_region = self._assign_single_individual_region(child)
            child_advantage = self._calculate_relative_advantage_for_new(child, child_region)
            
            # Strategy 1: Direct dominance replacement
            replaced = False
            for pop_idx, pop_ind in enumerate(self.population.individuals):
                if dominates(child, pop_ind, directions):
                    self.population.individuals[pop_idx] = child
                    replaced = True
                    break
            
            if replaced:
                continue
            
            # Strategy 2: Replace worst in same region if child has advantage
            region_members = [
                idx for idx, region in self.region_assignments.items()
                if region == child_region
            ]
            
            if region_members:
                # Find worst member in region
                worst_idx = min(region_members, 
                               key=lambda idx: self._calculate_relative_advantage(idx, child_region))
                worst_advantage = self._calculate_relative_advantage(worst_idx, child_region)
                
                # Replace if child has better advantage
                if child_advantage > worst_advantage:
                    self.population.individuals[worst_idx] = child

    def _assign_single_individual_region(self, individual: Individual) -> int:
        """Assign a single new individual to the closest reference point region."""
        directions = self.evaluator.objective_directions
        
        if individual.fitness is None:
            return 0
        
        best_region = 0
        best_value = float('inf')
        
        for region_idx, ref_point in enumerate(self.reference_points):
            value = self._scalarize_fitness(individual.fitness, ref_point, directions)
            if value < best_value:
                best_value = value
                best_region = region_idx
        
        return best_region

    def _calculate_relative_advantage_for_new(self, 
                                              individual: Individual,
                                              region_idx: int) -> float:
        """Calculate relative advantage for a new individual."""
        directions = self.evaluator.objective_directions
        
        if individual.fitness is None:
            return 0.0
        
        dominated_count = 0
        region_members = [
            idx for idx, region in self.region_assignments.items()
            if region == region_idx
        ]
        
        for other_idx in region_members:
            other = self.population.individuals[other_idx]
            if dominates(individual, other, directions):
                dominated_count += 1
        
        ref_point = self.reference_points[region_idx]
        scalarized = self._scalarize_fitness(individual.fitness, ref_point, directions)
        
        dominance_advantage = dominated_count / max(1, len(region_members))
        scalarized_advantage = 1.0 / (1.0 + scalarized)
        
        advantage = (
            self.relative_advantage_weight * dominance_advantage +
            (1.0 - self.relative_advantage_weight) * scalarized_advantage
        )
        return advantage

    def run(self) -> Population:
        """Execute NPRA search."""
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations = 0
        
        # Generate reference points based on objectives
        n_objectives = len(self.population.individuals[0].fitness)
        self.reference_points = self._generate_reference_points(n_objectives)
        
        self._record_history()
        
        while not self.termination.should_stop(self.evaluations, self.generations):
            # Selection: select parents based on relative advantage
            parents = self._select_high_advantage_parents(self.population.size)
            
            if not parents:
                parents = self.population.individuals[:self.population.size]
            
            # Variation: crossover and mutation
            offspring = self.variation.generate(parents, self.population.size)
            self._evaluate_offspring(offspring)
            
            # Replacement: dominance + relative advantage based
            self._replace_with_advantage(offspring)
            
            # Ensure population doesn't exceed size (keep elite)
            if len(self.population.individuals) > self.population.size:
                # Keep best individuals using rank and crowding
                assign_rank_and_crowding(self.population.individuals, 
                                        self.evaluator.objective_directions)
                self.population.individuals.sort(
                    key=lambda ind: (ind.rank, -ind.crowding_distance)
                )
                self.population.individuals = self.population.individuals[:self.population.size]
            
            self.generations += 1
            self._record_history()
        
        return self.population


# Backward-compatible alias for older imports.
MOEADStrategy = BiPopulationUniformSamplingMOEADStrategy

