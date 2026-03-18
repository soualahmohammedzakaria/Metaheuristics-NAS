"""
tests/test_moea_nas_strategy.py
================================
Unit and integration tests for NSGAIIStrategy, SPEAIIStrategy, MOPSOStrategy
(Article 2 — Wang et al. 2021).

Test categories
---------------
Shared helper tests:
    - _single_point_crossover: length preserved, genes from parents
    - _single_bit_mutation: exactly one gene changed, stays in range

Per-strategy integration tests (require CSV):
    - Budget respected
    - Population size preserved
    - All individuals have valid genotypes and evaluated fitness
    - Pareto front non-empty
    - History recorded and monotone
    - Determinism (same seed → same result)

SPEA-II specific:
    - Archive size bounded after update
    - k-NN truncation preserves archive_max

MOPSO specific:
    - Archive never exceeds archive_max
    - Particle positions remain valid integer genotypes
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.moea_nas_strategy import (
    NSGAIIStrategy,
    SPEAIIStrategy,
    MOPSOStrategy,
    _single_point_crossover,
    _single_bit_mutation,
    NUM_OPS,
    NUM_EDGES,
)


# ═══════════════════════════════════════════════════════════════
# 1.  Pure-unit tests
# ═══════════════════════════════════════════════════════════════

class TestSharedHelpers:

    def test_crossover_length_preserved(self):
        g1 = [0, 1, 2, 3, 4, 0]
        g2 = [4, 3, 2, 1, 0, 4]
        child = _single_point_crossover(g1, g2)
        assert len(child) == NUM_EDGES

    def test_crossover_genes_from_parents(self):
        g1 = [0, 0, 0, 0, 0, 0]
        g2 = [4, 4, 4, 4, 4, 4]
        for _ in range(50):
            child = _single_point_crossover(g1, g2)
            for gene in child:
                assert gene in (0, 4), f"Gene {gene} not from either parent"

    def test_crossover_point_not_trivial(self):
        """Child must not always equal one entire parent."""
        g1 = [0] * NUM_EDGES
        g2 = [4] * NUM_EDGES
        children = [_single_point_crossover(g1, g2) for _ in range(100)]
        # At least one child should be a mix (not all-0 or all-4)
        mixed = [c for c in children if 0 < sum(c) < 4 * NUM_EDGES]
        assert len(mixed) > 0, "Single-point crossover never produces mixed child"

    def test_mutation_changes_exactly_one_gene(self):
        for _ in range(100):
            g = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
            m = _single_bit_mutation(g)
            diffs = [i for i in range(NUM_EDGES) if g[i] != m[i]]
            assert len(diffs) == 1, f"Expected 1 changed gene, got {len(diffs)}"

    def test_mutation_gene_stays_in_range(self):
        for _ in range(100):
            g = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
            m = _single_bit_mutation(g)
            for gene in m:
                assert 0 <= gene < NUM_OPS

    def test_mutation_differs_from_original(self):
        """Mutated gene must be different from the original value."""
        for _ in range(100):
            g = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
            m = _single_bit_mutation(g)
            # Find changed position
            changed = [i for i in range(NUM_EDGES) if g[i] != m[i]]
            assert len(changed) == 1
            pos = changed[0]
            assert m[pos] != g[pos], "Mutated gene must differ from original"


# ═══════════════════════════════════════════════════════════════
# 2.  Shared integration test mixin
# ═══════════════════════════════════════════════════════════════

class _StrategyIntegrationBase:
    """
    Base class providing common integration checks.
    Subclasses override `_build()` to return the correct strategy instance.
    """

    BUDGET = 50
    POP_SIZE = 10

    def _build(self, population, evaluator, budget, seed):
        raise NotImplementedError

    def _fresh(self, csv_path, evaluator, budget=None, seed=0, pop_size=None):
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(seed)
        np.random.seed(seed)
        b = budget or self.BUDGET
        n = pop_size or self.POP_SIZE
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=n)
        return self._build(pop, evaluator, budget=b, seed=seed)

    # ── Common tests ──────────────────────────────────────────

    def test_budget_respected(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        s.run()
        assert s.evaluations <= self.BUDGET + self.POP_SIZE  # init + loop

    def test_population_size_preserved(self, csv_path, evaluator):
        n = self.POP_SIZE
        s = self._fresh(csv_path, evaluator, pop_size=n)
        final = s.run()
        assert len(final.individuals) == n

    def test_all_individuals_have_fitness(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        final = s.run()
        for ind in final.individuals:
            assert ind.fitness is not None
            assert len(ind.fitness) == 2

    def test_genotypes_valid(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        final = s.run()
        for ind in final.individuals:
            assert len(ind.genotype) == NUM_EDGES
            for gene in ind.genotype:
                assert 0 <= gene < NUM_OPS, f"Invalid gene: {gene}"

    def test_pareto_front_nonempty(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        final = s.run()
        pf = final.pareto_front()
        assert len(pf) >= 1

    def test_history_recorded(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        s.run()
        assert len(s.history.entries) >= 1

    def test_history_evaluations_monotone(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator, budget=60)
        s.run()
        evals = [e.evaluations for e in s.history.entries]
        assert evals == sorted(evals), "History evaluations not monotone"

    def test_determinism(self, csv_path, evaluator):
        results = []
        for _ in range(2):
            s = self._fresh(csv_path, evaluator, budget=40, seed=99)
            final = s.run()
            pf = sorted(final.pareto_front(), key=lambda i: (i.fitness[0], i.fitness[1]))
            results.append([(ind.genotype, ind.fitness) for ind in pf])
        assert results[0] == results[1], "Strategy is non-deterministic"

    def test_accuracy_and_latency_positive(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator)
        final = s.run()
        for ind in final.individuals:
            acc, lat = ind.fitness
            assert acc >= 0.0
            assert lat >= 0.0

    def test_pareto_front_individuals_dominate_correctly(self, csv_path, evaluator):
        """No individual in the Pareto front should be dominated by another."""
        from nas_framework.mo_utils import dominates
        s = self._fresh(csv_path, evaluator)
        final = s.run()
        pf = final.pareto_front()
        dirs = evaluator.objective_directions
        for i, a in enumerate(pf):
            for j, b in enumerate(pf):
                if i != j:
                    assert not dominates(b, a, dirs), (
                        f"Individual {b.fitness} dominates {a.fitness} in Pareto front"
                    )


# ═══════════════════════════════════════════════════════════════
# 3.  NSGA-II
# ═══════════════════════════════════════════════════════════════

class TestNSGAIIStrategy(_StrategyIntegrationBase):

    def _build(self, population, evaluator, budget, seed):
        return NSGAIIStrategy(
            population=population,
            evaluator=evaluator,
            p_cross=0.4, p_mut=0.4,
            budget=budget,
        )

    def test_offspring_count_equals_pop_size(self, csv_path, evaluator):
        """NSGA-II generates NP offspring per generation."""
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(0)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=10)
        strategy = self._build(pop, evaluator, budget=40, seed=0)
        # Access internal method to verify offspring count
        strategy.population.initialize()
        offspring = strategy._make_offspring(strategy.population.individuals)
        assert len(offspring) == strategy.population.size

    def test_pareto_size_grows_or_stays(self, csv_path, evaluator):
        """Pareto front size should never collapse to zero after gen 0."""
        s = self._fresh(csv_path, evaluator, budget=80)
        s.run()
        pf_sizes = [e.pareto_front_size for e in s.history.entries]
        assert all(sz >= 1 for sz in pf_sizes)


# ═══════════════════════════════════════════════════════════════
# 4.  SPEA-II
# ═══════════════════════════════════════════════════════════════

class TestSPEAIIStrategy(_StrategyIntegrationBase):

    ARCHIVE_SIZE = 10

    def _build(self, population, evaluator, budget, seed):
        return SPEAIIStrategy(
            population=population,
            evaluator=evaluator,
            archive_size=self.ARCHIVE_SIZE,
            p_cross=0.4, p_mut=0.4,
            budget=budget,
        )

    def test_archive_size_bounded(self, csv_path, evaluator):
        """Internal archive must never exceed archive_size."""
        s = self._fresh(csv_path, evaluator, budget=60)
        s.run()
        assert len(s._archive) <= self.ARCHIVE_SIZE

    def test_archive_all_non_dominated(self, csv_path, evaluator):
        """
        After the run, all solutions in the archive should be non-dominated
        among themselves (they form a Pareto front).
        """
        from nas_framework.mo_utils import dominates
        s = self._fresh(csv_path, evaluator, budget=60)
        s.run()
        dirs = evaluator.objective_directions
        archive = [ind for ind in s._archive if ind.fitness is not None]
        for i, a in enumerate(archive):
            for j, b in enumerate(archive):
                if i != j:
                    assert not dominates(b, a, dirs), (
                        f"Archive not Pareto-clean: {b.fitness} dominates {a.fitness}"
                    )

    def test_spea2_fitness_length(self, csv_path, evaluator):
        """_spea2_fitness must return one value per individual."""
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(0)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=8)
        pop.initialize()
        strategy = SPEAIIStrategy(pop, evaluator, archive_size=8, budget=20)
        fitness_vals = strategy._spea2_fitness(pop.individuals)
        assert len(fitness_vals) == len(pop.individuals)

    def test_spea2_fitness_nonnegative(self, csv_path, evaluator):
        """SPEA-II fitness must be >= 0 for all individuals."""
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(1)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=8)
        pop.initialize()
        strategy = SPEAIIStrategy(pop, evaluator, archive_size=8, budget=20)
        fitness_vals = strategy._spea2_fitness(pop.individuals)
        assert all(f >= 0.0 for f in fitness_vals)


# ═══════════════════════════════════════════════════════════════
# 5.  MOPSO
# ═══════════════════════════════════════════════════════════════

class TestMOPSOStrategy(_StrategyIntegrationBase):

    ARCHIVE_MAX = 10

    def _build(self, population, evaluator, budget, seed):
        np.random.seed(seed)
        return MOPSOStrategy(
            population=population,
            evaluator=evaluator,
            archive_max=self.ARCHIVE_MAX,
            w_max=1.0, w_min=0.0,
            c1=2.0, c2=2.0,
            budget=budget,
        )

    def test_archive_never_exceeds_max(self, csv_path, evaluator):
        """Particle archive must never exceed archive_max."""
        s = self._fresh(csv_path, evaluator, budget=60)
        s.run()
        assert len(s._archive) <= self.ARCHIVE_MAX

    def test_archive_nonempty_after_run(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator, budget=40)
        s.run()
        assert len(s._archive) >= 1

    def test_archive_individuals_have_fitness(self, csv_path, evaluator):
        s = self._fresh(csv_path, evaluator, budget=40)
        s.run()
        for ind in s._archive:
            assert ind.fitness is not None

    def test_particle_genotypes_valid(self, csv_path, evaluator):
        """All archive genotypes must be valid NAS-Bench-201 encodings."""
        s = self._fresh(csv_path, evaluator, budget=60)
        s.run()
        for ind in s._archive:
            assert len(ind.genotype) == NUM_EDGES
            for gene in ind.genotype:
                assert 0 <= gene < NUM_OPS, f"Invalid gene {gene} in MOPSO archive"

    def test_leader_selection_returns_archive_member(self, csv_path, evaluator):
        """_select_leader must return an object from the archive."""
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(0)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=8)
        strategy = self._build(pop, evaluator, budget=20, seed=0)
        strategy.population.initialize()
        for ind in strategy.population.individuals:
            strategy._add_to_archive(ind)
        leader = strategy._select_leader()
        assert leader in strategy._archive


# ═══════════════════════════════════════════════════════════════
# 6.  Cross-strategy comparison tests
# ═══════════════════════════════════════════════════════════════

class TestCrossStrategyConsistency:
    """High-level checks valid for all three MOEA strategies."""

    BUDGET = 60
    POP_SIZE = 10

    def _run_strategy(self, cls, csv_path, evaluator, seed, **kwargs):
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(seed)
        np.random.seed(seed)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=self.POP_SIZE)
        strategy = cls(pop, evaluator, budget=self.BUDGET, **kwargs)
        strategy.run()
        return strategy

    def test_all_strategies_return_nonempty_pareto(self, csv_path, evaluator):
        for cls, kwargs in [
            (NSGAIIStrategy, {"p_cross": 0.4, "p_mut": 0.4}),
            (SPEAIIStrategy, {"archive_size": 10, "p_cross": 0.4, "p_mut": 0.4}),
            (MOPSOStrategy,  {"archive_max": 10}),
        ]:
            s = self._run_strategy(cls, csv_path, evaluator, seed=0, **kwargs)
            pf = s.population.pareto_front()
            assert len(pf) >= 1, f"{cls.__name__} produced empty Pareto front"

    def test_all_strategies_respect_budget(self, csv_path, evaluator):
        for cls, kwargs in [
            (NSGAIIStrategy, {"p_cross": 0.4, "p_mut": 0.4}),
            (SPEAIIStrategy, {"archive_size": 10, "p_cross": 0.4, "p_mut": 0.4}),
            (MOPSOStrategy,  {"archive_max": 10}),
        ]:
            s = self._run_strategy(cls, csv_path, evaluator, seed=0, **kwargs)
            assert s.evaluations <= self.BUDGET + self.POP_SIZE, (
                f"{cls.__name__} exceeded budget: {s.evaluations} > {self.BUDGET}"
            )
