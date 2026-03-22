from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nas_framework.search_space import SearchSpace
    from nas_framework.evaluator import Evaluator

from nas_framework.population import Individual
from nas_framework.history import History
from nas_framework.mo_utils import (
    crowding_distance,
    assign_rank_and_crowding,
    take_pareto_best,
    pareto_front as compute_pareto_front,
)
from nas_framework.mutation import SinglePointMutation
from nas_framework.crossover import SinglePointCrossover


class LFMOGPSearch:
    """Leader-Follower Multi-Objective Genetic Programming (LF-MOGP).

    Based on: Liu et al., "Evolutionary convolutional neural network for image
    classification based on multi-objective genetic programming with leader-follower
    mechanism", Complex & Intelligent Systems, 2023.

    Algorithm 1 from the paper:
      - E  : external archive of non-dominated solutions  (Leader)
      - Ft : elite population of dominated solutions      (Follower)
      - xi threshold t/(t+T) controls:
            early phase (xi > t/(t+T))  -> parents from E only  -> fast convergence
            late  phase (xi <= t/(t+T)) -> parents from E + Ft  -> diversity

    All 10 fixes applied:
      FIX 1  : assign_rank_and_crowding() used correctly (side-effect + rank==0 filter)
      FIX 2  : two distinct crossover offspring (parents swapped on second call)
      FIX 3  : no import of non-existent exact_pareto_front_2d
      FIX 4  : History always initialised (never None)
      FIX 5  : genotype deduplication in E removes duplicate arch entries
      FIX 6  : pareto_front() applied after dedup guarantees E is truly non-dominated
      FIX 7  : max_generations auto-derived from budget when not explicitly set
      FIX 8  : Ft deduplication by genotype before elite selection
      FIX 9  : skip generation update when no offspring evaluated (budget exhausted)
      FIX 10 : _select_elite() guards against empty candidate list
    """

    def __init__(
        self,
        search_space,
        evaluator,
        pop_size: int = 30,
        elite_size: int = 10,
        max_generations: int | None = None,
        budget: int = 500,
        history: History | None = None,
        **kwargs,
    ):
        self.search_space = search_space
        self.evaluator    = evaluator
        self.pop_size     = pop_size
        self.elite_size   = elite_size
        self.budget       = budget

        # FIX 7: auto-derive max_generations from budget so the loop never
        # terminates early due to a too-small default generation count.
        # Each generation costs 4 real evaluations (2 mutation + 2 crossover).
        if max_generations is not None:
            self.max_generations = max_generations
        else:
            remaining = max(budget - pop_size, 4)
            self.max_generations = max(remaining // 4, 10)

        # FIX 4: always initialise history so record() is never skipped silently
        self.history = history or History()

        self.evaluations = 0
        self.generations = 0

        # Operators
        self.mutation  = SinglePointMutation(search_space)
        self.crossover = SinglePointCrossover()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_new(self, genotype: list[int]) -> Individual:
        """Evaluate one genotype, wrap in Individual, increment counter."""
        fitness  = self.evaluator.evaluate(genotype)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(genotype)
        self.evaluations += 1
        return Individual(genotype, fitness, metadata=metadata)

    @staticmethod
    def _dedup(individuals: list[Individual]) -> list[Individual]:
        """FIX 5 / FIX 8: Remove duplicate architectures, keeping first occurrence."""
        seen:   set[tuple]        = set()
        result: list[Individual]  = []
        for ind in individuals:
            key = tuple(ind.genotype)
            if key not in seen:
                seen.add(key)
                result.append(ind)
        return result

    def _clean_archive(self, candidates: list[Individual]) -> list[Individual]:
        """FIX 5 + FIX 6: Deduplicate then apply true Pareto filter on E."""
        deduped = self._dedup(candidates)
        if not deduped:
            return []
        # compute_pareto_front internally calls assign_rank_and_crowding
        front = compute_pareto_front(deduped, self.evaluator.objective_directions)
        if len(front) > 2:
            crowding_distance(front, self.evaluator.objective_directions)
        return front

    def _select_elite(self, candidates: list[Individual], k: int) -> list[Individual]:
        """FIX 10: Select k best by Pareto rank + crowding distance; safe on empty input."""
        if not candidates:
            return []
        deduped = self._dedup(candidates)
        if not deduped:
            return []
        if len(deduped) <= k:
            assign_rank_and_crowding(deduped, self.evaluator.objective_directions)
            return list(deduped)
        return take_pareto_best(deduped, k, self.evaluator.objective_directions)

    # ─────────────────────────────────────────────────────────────────────────
    # Main algorithm  (Algorithm 1 from the paper)
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> list[Individual]:
        """Execute LF-MOGP and return the final cleaned Pareto archive E."""

        # ── INITIALIZATION ────────────────────────────────────────────────
        P: list[Individual] = []
        for _ in range(self.pop_size):
            geno = self.search_space.random_individual()
            P.append(self._evaluate_new(geno))

        # FIX 1: side-effect sets .rank and .crowding_distance on every element
        assign_rank_and_crowding(P, self.evaluator.objective_directions)

        # E <- all non-dominated solutions, cleaned and deduplicated
        E_raw = [ind for ind in P if ind.rank == 0]
        E: list[Individual] = self._clean_archive(E_raw)

        # Ft <- K elite individuals from the dominated part of P
        dominated       = [ind for ind in P if ind.rank > 0]
        Ft: list[Individual] = self._select_elite(dominated, self.elite_size)

        # Record generation 0
        self.history.record(
            generation   = self.generations,
            evaluations  = self.evaluations,
            population   = E + Ft,
            pareto_front = E,
        )

        # ── MAIN LOOP ─────────────────────────────────────────────────────
        while (self.generations < self.max_generations
               and self.evaluations < self.budget):

            t = self.generations + 1   # 1-based
            T = self.max_generations

            # Adaptive phase switch
            xi          = random.uniform(0.0, 1.0)
            early_phase = xi > (t / (t + T))

            # ── Parent selection ─────────────────────────────────────────
            pool_leader   = E  if E  else P
            pool_follower = Ft if Ft else (E if E else P)

            if early_phase:
                # EARLY: exploit leader only -> fast convergence
                p1 = random.choice(pool_leader)
                p2 = random.choice(pool_leader)
                q1 = random.choice(pool_leader)
                q2 = random.choice(pool_leader)
            else:
                # LATE: mix leader + follower -> diversity / avoid local optima
                p1 = random.choice(pool_leader)
                p2 = random.choice(pool_follower)
                q1 = random.choice(pool_leader)
                q2 = random.choice(pool_follower)

            # ── Generate 4 offspring ──────────────────────────────────────
            child_p1 = self.mutation.mutate(p1)
            child_p2 = self.mutation.mutate(p2)
            # FIX 2: distinct crossover children via swapped parents
            child_q1 = self.crossover.crossover(q1, q2)
            child_q2 = self.crossover.crossover(q2, q1)

            raw_offspring = [child_p1, child_p2, child_q1, child_q2]

            # ── Evaluate offspring ────────────────────────────────────────
            evaluated: list[Individual] = []
            for child in raw_offspring:
                if self.evaluations >= self.budget:
                    break
                fitness  = self.evaluator.evaluate(child.genotype)
                metadata = {}
                if hasattr(self.search_space, "metadata_from_genotype"):
                    metadata = self.search_space.metadata_from_genotype(child.genotype)
                child.fitness  = fitness
                child.metadata = metadata
                self.evaluations += 1
                evaluated.append(child)

            # FIX 9: if budget was hit before any child was evaluated, stop
            if not evaluated:
                break

            # ── Update E (Leader archive) ─────────────────────────────────
            # FIX 5 + FIX 6: deduplicate then true Pareto filter
            E = self._clean_archive(E + evaluated)

            # ── Update Ft (Follower elite) ────────────────────────────────
            # Ft gets: old Ft + offspring that did NOT enter E
            e_keys   = {tuple(ind.genotype) for ind in E}
            rejected = [ind for ind in evaluated if tuple(ind.genotype) not in e_keys]

            # FIX 8: dedup F_candidate before selection
            Ft = self._select_elite(Ft + rejected, self.elite_size)

            self.generations += 1

            self.history.record(
                generation   = self.generations,
                evaluations  = self.evaluations,
                population   = E + Ft,
                pareto_front = E,
            )

        return E