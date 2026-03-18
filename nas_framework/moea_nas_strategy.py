"""
moea_nas_strategy.py
====================
Multiobjective Evolutionary Algorithm strategies for NAS,
integrated into the Orama NAS framework.

Based on:
    Wang, X., Wang, X., Jin, L., Lv, R., Dai, B., He, M., & Lv, T. (2021).
    "Evolutionary Algorithm-Based and Network Architecture Search-Enabled
     Multiobjective Traffic Classification"
    IEEE Access, 9, 52310–52325. DOI: 10.1109/ACCESS.2021.3068267

Three strategies are implemented, all inheriting from SearchStrategy:
    - NSGAIIStrategy  : NSGA-II (best F1-score, largest Pareto set)
    - SPEAIIStrategy  : SPEA-II (strength-based fitness + k-NN truncation)
    - MOPSOStrategy   : MOPSO   (particle swarm + external Pareto archive)

How it fits the framework
--------------------------
- All three strategies use CSVSearchSpace / CSVBenchmarkAPI / Evaluator
  unchanged.
- History, Termination, and Population are handled identically to the
  existing GeneticAlgorithm / RandomSearch strategies.
- The framework's mo_utils (fast_non_dominated_sort, crowding_distance,
  dominates, take_pareto_best) are reused directly.
- Each strategy exposes `pareto_front()` on the final population, so
  downstream analysis is identical across all strategies.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from nas_framework.population import Individual, Population
from nas_framework.evaluator import Evaluator
from nas_framework.termination import Termination, MaxEvaluationsTermination
from nas_framework.history import History
from nas_framework.search_strategy import SearchStrategy
from nas_framework.selection import Selection
from nas_framework.variation import Variation
from nas_framework.replacement import Replacement
from nas_framework.mo_utils import (
    fast_non_dominated_sort,
    crowding_distance,
    assign_rank_and_crowding,
    dominates,
    take_pareto_best,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_OPS   = 5
NUM_EDGES = 6


def _evaluate_and_attach(
    ind: Individual,
    evaluator: Evaluator,
    search_space,
) -> None:
    """Evaluate an individual in-place and attach benchmark metadata."""
    ind.fitness = evaluator.evaluate(ind.genotype)
    if hasattr(search_space, "metadata_from_genotype"):
        ind.metadata.update(search_space.metadata_from_genotype(ind.genotype))
    elif hasattr(evaluator.benchmark, "get_metadata"):
        ind.metadata.update(evaluator.benchmark.get_metadata(ind.genotype))


def _single_point_crossover(g1: list[int], g2: list[int]) -> list[int]:
    """Single-point crossover used by NSGA-II and SPEA-II."""
    pt = random.randint(1, len(g1) - 1)
    return g1[:pt] + g2[pt:]


def _single_bit_mutation(genotype: list[int], n_ops: int = NUM_OPS) -> list[int]:
    """
    Flip exactly one randomly chosen gene to a different operation value.
    Mirrors the paper's "restrict flipped bits to one per mutation".
    """
    g = genotype[:]
    idx = random.randint(0, len(g) - 1)
    choices = [op for op in range(n_ops) if op != g[idx]]
    g[idx] = random.choice(choices)
    return g


# Stub operators to satisfy the SearchStrategy ABC
class _NoOpSelection(Selection):
    def select(self, individuals, n, objective_directions):
        return individuals

class _NoOpReplacement(Replacement):
    def replace(self, population, offspring, pop_size, objective_directions):
        return population

class _NoOpVariation(Variation):
    def generate(self, parents, n_offspring):
        return []


# ---------------------------------------------------------------------------
# 1. NSGA-II Strategy
# ---------------------------------------------------------------------------

class NSGAIIStrategy(SearchStrategy):
    """
    NSGA-II for multiobjective NAS.

    Objectives: maximise accuracy (obj[0]), minimise latency (obj[1]).
    Variation : single-point crossover + single-bit mutation.
    Survival  : non-dominated sorting + crowding distance (framework's
                take_pareto_best / assign_rank_and_crowding).

    Parameters
    ----------
    population    : Population (paper uses 20)
    evaluator     : Evaluator
    p_cross       : crossover probability (paper: 0.4)
    p_mut         : mutation  probability (paper: 0.4)
    budget        : max evaluations
    termination   : optional Termination override
    history       : optional History
    """

    def __init__(
        self,
        population: Population,
        evaluator: Evaluator,
        p_cross: float = 0.4,
        p_mut: float = 0.4,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        super().__init__(
            population=population,
            selection=_NoOpSelection(),
            variation=_NoOpVariation(),
            replacement=_NoOpReplacement(),
            evaluator=evaluator,
            termination=termination or MaxEvaluationsTermination(budget),
            history=history or History(),
            budget=budget,
        )
        self.p_cross = p_cross
        self.p_mut = p_mut

    def _make_offspring(self, pop: list[Individual]) -> list[Individual]:
        """Generate NP offspring via tournament + crossover + mutation."""
        shuffled = pop[:]
        random.shuffle(shuffled)
        offspring: list[Individual] = []
        N = len(pop)

        for k in range(0, N - 1, 2):
            p1 = shuffled[k]
            p2 = shuffled[(k + 1) % N]

            # Crossover
            if random.random() < self.p_cross:
                c1_geno = _single_point_crossover(p1.genotype, p2.genotype)
                c2_geno = _single_point_crossover(p2.genotype, p1.genotype)
            else:
                c1_geno, c2_geno = p1.genotype[:], p2.genotype[:]

            # Mutation
            if random.random() < self.p_mut:
                c1_geno = _single_bit_mutation(c1_geno)
            if random.random() < self.p_mut:
                c2_geno = _single_bit_mutation(c2_geno)

            offspring.extend([Individual(c1_geno), Individual(c2_geno)])

        return offspring[:N]

    def run(self) -> Population:
        # ── Initialization ─────────────────────────────────────────
        self.population.initialize()
        self.evaluations = len(self.population.individuals)
        self.generations = 0
        self._record_history()

        # ── Generational loop ───────────────────────────────────────
        while not self.termination.should_stop(self.evaluations, self.generations):
            pop = self.population.individuals
            offspring = self._make_offspring(pop)

            # Evaluate offspring
            for child in offspring:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                _evaluate_and_attach(
                    child, self.evaluator, self.population.search_space
                )
                self.evaluations += 1

            # NSGA-II survival: rank + crowding on combined pool
            combined = pop + [c for c in offspring if c.fitness is not None]
            self.population.individuals = take_pareto_best(
                combined,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            self._record_history()

        return self.population


# ---------------------------------------------------------------------------
# 2. SPEA-II Strategy
# ---------------------------------------------------------------------------

class SPEAIIStrategy(SearchStrategy):
    """
    SPEA-II for multiobjective NAS.

    Key differences from NSGA-II:
    - External archive stores non-dominated solutions across all generations.
    - Fitness = raw (strength-based) + density (1/k-NN distance).
    - Truncation by k-NN distance when archive overflows.

    Parameters
    ----------
    population    : Population (N)
    evaluator     : Evaluator
    archive_size  : size of external archive (paper: 20)
    p_cross       : crossover probability
    p_mut         : mutation probability
    k_nn          : k for k-NN density estimate (default: 1)
    budget        : max evaluations
    termination   : optional Termination override
    history       : optional History
    """

    def __init__(
        self,
        population: Population,
        evaluator: Evaluator,
        archive_size: int = 20,
        p_cross: float = 0.4,
        p_mut: float = 0.4,
        k_nn: int = 1,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        super().__init__(
            population=population,
            selection=_NoOpSelection(),
            variation=_NoOpVariation(),
            replacement=_NoOpReplacement(),
            evaluator=evaluator,
            termination=termination or MaxEvaluationsTermination(budget),
            history=history or History(),
            budget=budget,
        )
        self.archive_size = archive_size
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.k_nn = k_nn
        self._archive: list[Individual] = []

    # ------------------------------------------------------------------
    # SPEA-II fitness computation
    # ------------------------------------------------------------------

    def _spea2_fitness(self, individuals: list[Individual]) -> list[float]:
        """
        Fitness = raw_fitness + density_estimate.

        raw_fitness[i] = sum of strength[j] for all j that dominate i
        density[i]     = 1 / (sigma_k + 2)  where sigma_k = k-th NN dist
        """
        n = self.evaluator.objective_directions
        directions = n

        # Strength: how many solutions each individual dominates
        strength = [0.0] * len(individuals)
        for i, a in enumerate(individuals):
            for j, b in enumerate(individuals):
                if i != j and dominates(a, b, directions):
                    strength[i] += 1.0

        # Raw fitness: sum of strengths of all dominators
        raw = [0.0] * len(individuals)
        for i, a in enumerate(individuals):
            for j, b in enumerate(individuals):
                if i != j and dominates(b, a, directions):
                    raw[i] += strength[j]

        # Density estimate using k-NN on the objective space
        obj_arr = np.array([
            [v * d for v, d in zip(ind.fitness, directions)]
            for ind in individuals
        ])
        density = []
        for i in range(len(individuals)):
            dists = sorted(
                float(np.linalg.norm(obj_arr[i] - obj_arr[j]))
                for j in range(len(individuals)) if j != i
            )
            k = min(self.k_nn, len(dists) - 1)
            sigma_k = dists[k] if dists else 0.0
            density.append(1.0 / (sigma_k + 2.0))

        return [r + d for r, d in zip(raw, density)]

    # ------------------------------------------------------------------
    # Archive update
    # ------------------------------------------------------------------

    def _update_archive(
        self,
        combined: list[Individual],
    ) -> list[Individual]:
        """
        Select non-dominated individuals from combined into archive.
        If overflow: truncate by k-NN (remove closest pair).
        If underflow: fill with dominated individuals by fitness.
        """
        directions = self.evaluator.objective_directions
        fronts = fast_non_dominated_sort(combined, directions)
        nd = fronts[0] if fronts else []

        if len(nd) < self.archive_size:
            # Fill with dominated solutions sorted by fitness
            fitness_vals = self._spea2_fitness(combined)
            dominated = [ind for ind in combined if ind not in nd]
            dominated.sort(
                key=lambda ind: fitness_vals[combined.index(ind)]
            )
            nd = nd + dominated[: self.archive_size - len(nd)]
        elif len(nd) > self.archive_size:
            # Truncate: iteratively remove the individual with smallest
            # distance to its nearest neighbour
            obj_arr = np.array([
                [v * d for v, d in zip(ind.fitness, directions)]
                for ind in nd
            ])
            while len(nd) > self.archive_size:
                n = len(nd)
                min_dist = float("inf")
                remove_idx = 0
                for i in range(n):
                    dists = sorted(
                        float(np.linalg.norm(obj_arr[i] - obj_arr[j]))
                        for j in range(n) if j != i
                    )
                    if dists and dists[0] < min_dist:
                        min_dist = dists[0]
                        remove_idx = i
                nd.pop(remove_idx)
                obj_arr = np.delete(obj_arr, remove_idx, axis=0)

        return nd

    # ------------------------------------------------------------------
    # Mating
    # ------------------------------------------------------------------

    def _binary_tournament(
        self,
        pool: list[Individual],
        fitness: list[float],
    ) -> Individual:
        """Binary tournament on SPEA-II fitness (lower is better)."""
        i, j = random.sample(range(len(pool)), 2)
        return pool[i] if fitness[i] <= fitness[j] else pool[j]

    def _make_offspring(
        self,
        archive: list[Individual],
        fitness: list[float],
        N: int,
    ) -> list[Individual]:
        offspring = []
        while len(offspring) < N:
            p1 = self._binary_tournament(archive, fitness)
            p2 = self._binary_tournament(archive, fitness)

            if random.random() < self.p_cross:
                c_geno = _single_point_crossover(p1.genotype, p2.genotype)
            else:
                c_geno = p1.genotype[:]

            if random.random() < self.p_mut:
                c_geno = _single_bit_mutation(c_geno)

            offspring.append(Individual(c_geno))
        return offspring

    def run(self) -> Population:
        # ── Initialization ─────────────────────────────────────────
        self.population.initialize()
        self.evaluations = len(self.population.individuals)
        self.generations = 0
        self._archive = []
        self._record_history()

        # ── Generational loop ───────────────────────────────────────
        while not self.termination.should_stop(self.evaluations, self.generations):
            pop = self.population.individuals

            # Update archive from archive ∪ population
            combined = self._archive + pop
            self._archive = self._update_archive(combined)

            # Compute SPEA-II fitness on archive for mating selection
            archive_fitness = self._spea2_fitness(self._archive)

            # Generate offspring from archive
            offspring = self._make_offspring(
                self._archive, archive_fitness, self.population.size
            )

            # Evaluate offspring
            for child in offspring:
                if self.termination.should_stop(self.evaluations, self.generations):
                    break
                _evaluate_and_attach(
                    child, self.evaluator, self.population.search_space
                )
                self.evaluations += 1

            self.population.individuals = [
                c for c in offspring if c.fitness is not None
            ]
            # Pad with archive if offspring short (budget exhausted mid-loop)
            if len(self.population.individuals) < self.population.size:
                self.population.individuals = take_pareto_best(
                    self._archive + self.population.individuals,
                    self.population.size,
                    self.evaluator.objective_directions,
                )

            self.generations += 1
            self._record_history()

        # Expose archive as the final Pareto front
        if self._archive:
            self.population.individuals = take_pareto_best(
                self._archive + self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )
        return self.population


# ---------------------------------------------------------------------------
# 3. MOPSO Strategy
# ---------------------------------------------------------------------------

class MOPSOStrategy(SearchStrategy):
    """
    Multi-Objective Particle Swarm Optimization for NAS.

    Adaptation for discrete NAS (integer genotypes):
    - Each particle has a real-valued velocity.
    - Position update is done in real space; genes are rounded to integers.
    - External archive stores non-dominated solutions.
    - Leader is selected from the archive by crowding distance roulette.
    - Inertia weight decreases linearly from w_max to w_min (Eq. 5).

    Parameters
    ----------
    population  : Population (N particles)
    evaluator   : Evaluator
    archive_max : max external archive size (paper: 20)
    w_max       : initial inertia weight (paper: 1.0)
    w_min       : final inertia weight   (paper: 0.0)
    c1, c2      : cognitive / social acceleration coefficients
    budget      : max evaluations
    termination : optional Termination override
    history     : optional History
    """

    def __init__(
        self,
        population: Population,
        evaluator: Evaluator,
        archive_max: int = 20,
        w_max: float = 1.0,
        w_min: float = 0.0,
        c1: float = 2.0,
        c2: float = 2.0,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        super().__init__(
            population=population,
            selection=_NoOpSelection(),
            variation=_NoOpVariation(),
            replacement=_NoOpReplacement(),
            evaluator=evaluator,
            termination=termination or MaxEvaluationsTermination(budget),
            history=history or History(),
            budget=budget,
        )
        self.archive_max = archive_max
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self._archive: list[Individual] = []

    # ------------------------------------------------------------------
    # Archive helpers
    # ------------------------------------------------------------------

    def _add_to_archive(self, ind: Individual) -> None:
        """Add individual if non-dominated; prune archive if over capacity."""
        directions = self.evaluator.objective_directions

        # Reject if dominated by any archive member
        for a in self._archive:
            if dominates(a, ind, directions):
                return

        # Remove archive members dominated by ind
        self._archive = [
            a for a in self._archive if not dominates(ind, a, directions)
        ]
        self._archive.append(ind)

        # Trim by crowding distance if over capacity
        while len(self._archive) > self.archive_max:
            assign_rank_and_crowding(self._archive, directions)
            # Remove individual with smallest crowding distance
            worst = min(
                (a for a in self._archive if a.crowding_distance < float("inf")),
                key=lambda a: a.crowding_distance,
                default=self._archive[-1],
            )
            self._archive.remove(worst)

    def _select_leader(self) -> Individual:
        """
        Roulette wheel on crowding distance to select a leader from archive.
        """
        if len(self._archive) == 1:
            return self._archive[0]

        assign_rank_and_crowding(
            self._archive, self.evaluator.objective_directions
        )
        weights = [
            a.crowding_distance if a.crowding_distance < float("inf") else 10.0
            for a in self._archive
        ]
        total = sum(weights)
        if total == 0:
            return random.choice(self._archive)

        r = random.uniform(0, total)
        cumulative = 0.0
        for ind, w in zip(self._archive, weights):
            cumulative += w
            if cumulative >= r:
                return ind
        return self._archive[-1]

    def run(self) -> Population:
        # ── Initialization ─────────────────────────────────────────
        self.population.initialize()
        self.evaluations = len(self.population.individuals)
        self.generations = 0
        self._archive = []

        N = self.population.size
        D = NUM_EDGES

        # Real-valued positions and velocities
        pos = np.array([
            [float(g) for g in ind.genotype]
            for ind in self.population.individuals
        ], dtype=float)                          # shape (N, D)
        vel = np.random.uniform(-1.0, 1.0, (N, D))

        pbest_pos = pos.copy()
        pbest_fit = [ind.fitness for ind in self.population.individuals]

        # Populate initial archive
        for ind in self.population.individuals:
            self._add_to_archive(
                Individual(ind.genotype[:], ind.fitness, deepcopy(ind.metadata))
            )

        self._record_history()

        # Estimate total iterations for inertia decay
        T_est = max(1, (self.termination.max_evaluations - self.evaluations) // N) \
            if hasattr(self.termination, "max_evaluations") else 100

        t = 0

        # ── Particle update loop ────────────────────────────────────
        while not self.termination.should_stop(self.evaluations, self.generations):
            # Linearly decreasing inertia weight (Equation 5 of the paper)
            w = self.w_max - t * (self.w_max - self.w_min) / T_est

            for i in range(N):
                if self.termination.should_stop(self.evaluations, self.generations):
                    break

                leader = self._select_leader()
                leader_pos = np.array([float(g) for g in leader.genotype])

                r1 = np.random.rand(D)
                r2 = np.random.rand(D)

                # Velocity update (Eq. 3)
                vel[i] = (
                    w * vel[i]
                    + self.c1 * r1 * (pbest_pos[i] - pos[i])
                    + self.c2 * r2 * (leader_pos  - pos[i])
                )

                # Position update (Eq. 4)
                pos[i] = pos[i] + vel[i]

                # Discretize and clamp to valid gene range
                new_geno = [
                    int(np.clip(round(pos[i, d]), 0, NUM_OPS - 1))
                    for d in range(D)
                ]

                # Evaluate new position
                new_fitness = self.evaluator.evaluate(new_geno)
                self.evaluations += 1

                new_ind = Individual(new_geno, new_fitness)
                _evaluate_and_attach(
                    new_ind, self.evaluator, self.population.search_space
                )

                # Update personal best (non-dominated improvement)
                curr_best = Individual(
                    [int(round(p)) for p in pbest_pos[i]], pbest_fit[i]
                )
                if dominates(
                    new_ind, curr_best, self.evaluator.objective_directions
                ) or not dominates(
                    curr_best, new_ind, self.evaluator.objective_directions
                ):
                    pbest_pos[i] = pos[i].copy()
                    pbest_fit[i] = new_fitness

                self._add_to_archive(
                    Individual(new_geno, new_fitness, deepcopy(new_ind.metadata))
                )

            # Update population from particles + archive for history
            particle_inds = []
            for i in range(N):
                g = [int(np.clip(round(pos[i, d]), 0, NUM_OPS - 1)) for d in range(D)]
                particle_inds.append(Individual(g, self.evaluator.evaluate(g)))

            self.population.individuals = take_pareto_best(
                self._archive + particle_inds,
                self.population.size,
                self.evaluator.objective_directions,
            )
            self.generations += 1
            t += 1
            self._record_history()

        # Final population: fill to pop_size from archive + current individuals
        combined = self._archive + self.population.individuals
        if combined:
            self.population.individuals = take_pareto_best(
                combined,
                self.population.size,
                self.evaluator.objective_directions,
            )
        # If still short (archive tiny), pad by repeating archive members
        while len(self.population.individuals) < self.population.size and self._archive:
            self.population.individuals.append(
                random.choice(self._archive)
            )
        return self.population
