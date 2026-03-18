"""
de_nas_strategy.py
==================
Differential Evolution for Neural Architecture Search (DE-NAS)
integrated into the Orama NAS framework.

Based on:
    Awad, N., Mallik, N., & Hutter, F. (2021).
    "Differential Evolution for Neural Architecture Search"
    1st Workshop on NAS at ICLR 2020. arXiv:2012.06400v2.

How it fits the framework
--------------------------
- Uses the framework's CSVSearchSpace / CSVBenchmarkAPI / Evaluator unchanged.
- Inherits from SearchStrategy so History, Termination and Population
  are all handled identically to GeneticAlgorithm / RandomSearch.
- The continuous DE population lives in [0,1]^D internally; individuals
  are discretized to the 5-operation, 6-edge NAS-Bench-201 encoding
  only at evaluation time (the key trick from the paper).
- A new DifferentialEvolutionVariation component is added so the
  mutation+crossover logic can be reused independently if needed.
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


# ---------------------------------------------------------------------------
# Continuous ↔ Discrete encoding helpers
# ---------------------------------------------------------------------------

NUM_OPS = 5    # NAS-Bench-201: none, skip, conv1x1, conv3x3, avgpool
NUM_EDGES = 6  # NAS-Bench-201 cell has 6 edges


def _continuous_to_genotype(x: np.ndarray) -> list[int]:
    """
    Map a real vector x ∈ [0,1]^D to a discrete genotype.

    Each gene selects one of NUM_OPS operations via uniform binning:
        bin = floor(x_i * NUM_OPS), clamped to [0, NUM_OPS-1].

    This is the `discretize_architecture` step described in the paper.
    """
    x_clipped = np.clip(x, 0.0, 1.0)
    return [int(min(xi * NUM_OPS, NUM_OPS - 1)) for xi in x_clipped]


def _genotype_to_continuous(genotype: list[int]) -> np.ndarray:
    """
    Map a discrete genotype back to the centre of its continuous bin.
    Used when seeding the DE population from an existing discrete population.
    """
    return np.array([(g + 0.5) / NUM_OPS for g in genotype])


# ---------------------------------------------------------------------------
# DE Variation component (reusable outside DESearchStrategy if needed)
# ---------------------------------------------------------------------------

class DifferentialEvolutionVariation(Variation):
    """
    DE/rand/1/bin variation operator.

    Keeps the population in a continuous space; offspring are produced by
    mutation (rand/1) + binomial crossover, then discretized for evaluation.

    Parameters
    ----------
    F  : scaling factor         (paper: 0.5)
    Cr : crossover probability  (paper: 0.5)
    """

    def __init__(self, F: float = 0.5, Cr: float = 0.5):
        self.F = F
        self.Cr = Cr
        # Continuous shadow population: maintained by DESearchStrategy
        self._continuous_pop: np.ndarray | None = None

    def _mutate(self, pop: np.ndarray, i: int) -> np.ndarray:
        """DE/rand/1 mutation: V = X_r1 + F*(X_r2 - X_r3)"""
        indices = list(range(len(pop)))
        indices.remove(i)
        if len(indices) >= 3:
            r1, r2, r3 = random.sample(indices, 3)
        else:
            r1, r2, r3 = (random.choice(indices) for _ in range(3))
        mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
        return np.clip(mutant, 0.0, 1.0)

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover; j_rand ensures at least one gene from mutant."""
        D = len(target)
        j_rand = random.randint(0, D - 1)
        mask = np.array(
            [True if j == j_rand else random.random() < self.Cr
             for j in range(D)]
        )
        return np.where(mask, mutant, target)

    def generate(self, parents: list[Individual], n_offspring: int) -> list[Individual]:
        """
        Generate n_offspring trial vectors.

        NOTE: This method is called by the strategy loop. The continuous
        population must be set via `_continuous_pop` before calling.
        """
        if self._continuous_pop is None:
            raise RuntimeError(
                "DifferentialEvolutionVariation requires _continuous_pop "
                "to be set by DESearchStrategy before calling generate()."
            )
        pop = self._continuous_pop
        NP = len(pop)
        offspring: list[Individual] = []

        for i in range(min(n_offspring, NP)):
            mutant = self._mutate(pop, i)
            trial_cont = self._crossover(pop[i], mutant)
            genotype = _continuous_to_genotype(trial_cont)
            # Attach continuous vector as metadata for the strategy to retrieve
            child = Individual(genotype, metadata={"_trial_cont": trial_cont})
            offspring.append(child)

        return offspring


# ---------------------------------------------------------------------------
# DE Search Strategy
# ---------------------------------------------------------------------------

class DESearchStrategy(SearchStrategy):
    """
    Differential Evolution search strategy for NAS.

    Implements the full DE/rand/1/bin loop described in Algorithm 1 of
    Awad et al. (2021), adapted to the Orama framework.

    Key design decisions
    --------------------
    1. The continuous shadow population `_cont_pop` mirrors `population.individuals`
       but stores real vectors ∈ [0,1]^D.  Discrete genotypes are only used
       for benchmark queries.
    2. Selection is greedy one-to-one (standard DE): trial replaces target
       if it is not dominated; Pareto dominance is used as the comparison
       criterion (since we have two objectives).
    3. History, Termination, and Population are handled exactly like the
       existing GeneticAlgorithm strategy.

    Parameters
    ----------
    population  : Population (size NP, paper uses NP=20)
    evaluator   : Evaluator
    F           : scaling factor (paper: 0.5)
    Cr          : crossover rate (paper: 0.5)
    budget      : max evaluations (termination)
    termination : optional custom Termination
    history     : optional History
    """

    def __init__(
        self,
        population: Population,
        evaluator: Evaluator,
        F: float = 0.5,
        Cr: float = 0.5,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        # DE does not use external selection/replacement/variation in the
        # conventional sense; we pass dummy stubs to satisfy the ABC.
        de_variation = DifferentialEvolutionVariation(F=F, Cr=Cr)

        super().__init__(
            population=population,
            selection=_NoOpSelection(),
            variation=de_variation,
            replacement=_NoOpReplacement(),
            evaluator=evaluator,
            termination=termination or MaxEvaluationsTermination(budget),
            history=history or History(),
            budget=budget,
        )
        self.de_variation = de_variation
        self._cont_pop: np.ndarray | None = None   # continuous shadow

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pareto_dominates(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> bool:
        """Return True if a dominates b given objective directions (+1, -1)."""
        directions = self.evaluator.objective_directions
        a_norm = tuple(v * d for v, d in zip(a, directions))
        b_norm = tuple(v * d for v, d in zip(b, directions))
        no_worse = all(x >= y for x, y in zip(a_norm, b_norm))
        strictly_better = any(x > y for x, y in zip(a_norm, b_norm))
        return no_worse and strictly_better

    def _evaluate_genotype(self, genotype: list[int]) -> tuple[float, float]:
        """Query benchmark and increment evaluation counter."""
        fitness = self.evaluator.evaluate(genotype)
        self.evaluations += 1
        return fitness

    def _attach_metadata(self, ind: Individual) -> None:
        if hasattr(self.population.search_space, "metadata_from_genotype"):
            ind.metadata.update(
                self.population.search_space.metadata_from_genotype(ind.genotype)
            )
        elif hasattr(self.evaluator.benchmark, "get_metadata"):
            ind.metadata.update(
                self.evaluator.benchmark.get_metadata(ind.genotype)
            )

    # ------------------------------------------------------------------
    # Main loop (Algorithm 1, Awad et al.)
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute DE/rand/1/bin until the termination criterion is met."""

        # ── Initialization ─────────────────────────────────────────
        self.population.initialize()
        self.evaluations = len(self.population.individuals)
        self.generations = 0

        # Build the continuous shadow population from the initial genotypes
        self._cont_pop = np.array([
            _genotype_to_continuous(ind.genotype)
            for ind in self.population.individuals
        ])
        self._record_history()

        NP = len(self.population.individuals)

        # ── Generational loop ───────────────────────────────────────
        while not self.termination.should_stop(self.evaluations, self.generations):
            for i in range(NP):
                if self.termination.should_stop(self.evaluations, self.generations):
                    break

                target = self.population.individuals[i]

                # Mutation + Crossover (continuous space)
                mutant_cont = self.de_variation._mutate(self._cont_pop, i)
                trial_cont  = self.de_variation._crossover(
                    self._cont_pop[i], mutant_cont
                )
                trial_geno  = _continuous_to_genotype(trial_cont)

                # Evaluate trial vector
                trial_fitness = self._evaluate_genotype(trial_geno)
                trial = Individual(trial_geno, trial_fitness)
                self._attach_metadata(trial)

                # One-to-one Pareto selection (greedy replacement)
                # trial replaces target if it dominates, or if neither dominates
                target_fitness = target.fitness
                if (
                    self._pareto_dominates(trial_fitness, target_fitness)
                    or not self._pareto_dominates(target_fitness, trial_fitness)
                ):
                    self.population.individuals[i] = trial
                    self._cont_pop[i] = trial_cont

            self.generations += 1
            self._record_history()

        return self.population


# ---------------------------------------------------------------------------
# Stub operators required by SearchStrategy ABC
# ---------------------------------------------------------------------------

class _NoOpSelection(Selection):
    def select(self, individuals, n, objective_directions):
        return individuals


class _NoOpReplacement(Replacement):
    def replace(self, population, offspring, pop_size, objective_directions):
        return population
