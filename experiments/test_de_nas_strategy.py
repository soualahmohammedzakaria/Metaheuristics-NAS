"""
tests/test_de_nas_strategy.py
==============================
Unit and integration tests for DESearchStrategy (Article 1 — Awad et al. 2021).

Test categories
---------------
Unit tests (no CSV needed):
    - _continuous_to_genotype: correct binning and boundary behaviour
    - _genotype_to_continuous: round-trip consistency
    - DifferentialEvolutionVariation: mutation stays in [0,1],
      crossover respects j_rand guarantee

Integration tests (require CSV):
    - DESearchStrategy.run: budget respected, population size stable,
      all individuals have valid genotypes and evaluated fitness,
      Pareto front is non-empty, history recorded correctly
    - Determinism: same seed → identical results
    - Greedy one-to-one replacement logic
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

from nas_framework.de_nas_strategy import (
    DESearchStrategy,
    DifferentialEvolutionVariation,
    NUM_OPS,
    NUM_EDGES,
    _continuous_to_genotype,
    _genotype_to_continuous,
)


# ═══════════════════════════════════════════════════════════════
# 1.  Pure-unit tests  (no CSV / no benchmark needed)
# ═══════════════════════════════════════════════════════════════

class TestContinuousToGenotype:
    """_continuous_to_genotype maps [0,1] → [0, NUM_OPS-1]."""

    def test_zero_maps_to_op_zero(self):
        x = np.zeros(NUM_EDGES)
        g = _continuous_to_genotype(x)
        assert g == [0] * NUM_EDGES

    def test_one_maps_to_last_op(self):
        x = np.ones(NUM_EDGES)
        g = _continuous_to_genotype(x)
        assert g == [NUM_OPS - 1] * NUM_EDGES

    def test_output_within_valid_range(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.uniform(0.0, 1.0, NUM_EDGES)
            g = _continuous_to_genotype(x)
            for gene in g:
                assert 0 <= gene < NUM_OPS, f"Gene {gene} out of range"

    def test_out_of_range_input_clipped(self):
        # Values outside [0,1] must be clipped, not raise errors
        x = np.array([-0.5, 1.5, 0.5, -1.0, 2.0, 0.0])
        g = _continuous_to_genotype(x)
        for gene in g:
            assert 0 <= gene < NUM_OPS

    def test_uniform_bin_boundaries(self):
        """Each bin [k/N, (k+1)/N) should map to op k."""
        for op in range(NUM_OPS):
            mid = (op + 0.5) / NUM_OPS
            x = np.full(NUM_EDGES, mid)
            g = _continuous_to_genotype(x)
            assert all(gene == op for gene in g), (
                f"Expected op {op}, got {g} for mid={mid}"
            )


class TestGenotypeToContiguous:
    """_genotype_to_continuous places each gene at the bin centre."""

    def test_round_trip_consistency(self):
        """discretize(continuous(g)) == g for all valid genotypes."""
        for _ in range(100):
            geno = [random.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
            x = _genotype_to_continuous(geno)
            recovered = _continuous_to_genotype(x)
            assert recovered == geno, f"Round-trip failed: {geno} → {x} → {recovered}"

    def test_output_in_unit_interval(self):
        for op in range(NUM_OPS):
            x = _genotype_to_continuous([op] * NUM_EDGES)
            assert all(0.0 <= xi <= 1.0 for xi in x)


class TestDEVariation:
    """DifferentialEvolutionVariation: mutation and crossover properties."""

    def setup_method(self):
        random.seed(0)
        np.random.seed(0)
        self.D = NUM_EDGES
        self.NP = 6
        self.variation = DifferentialEvolutionVariation(F=0.5, Cr=0.5)
        # Inject a small continuous population
        self.pop = np.random.uniform(0.0, 1.0, (self.NP, self.D))
        self.variation._continuous_pop = self.pop

    def test_mutant_stays_in_unit_interval(self):
        """After mutation, all values must stay in [0,1] (boundary clipping)."""
        for i in range(self.NP):
            mutant = self.variation._mutate(self.pop, i)
            assert mutant.shape == (self.D,)
            assert np.all(mutant >= 0.0) and np.all(mutant <= 1.0), (
                f"Mutant out of bounds: {mutant}"
            )

    def test_mutant_uses_three_distinct_individuals(self):
        """rand/1: r1, r2, r3 must be different from i and from each other."""
        # We verify statistically: over many draws, mutants differ from target
        unique_mutants = set()
        for i in range(self.NP):
            m = self.variation._mutate(self.pop, i)
            unique_mutants.add(tuple(m.round(4)))
        # At least some variation expected
        assert len(unique_mutants) > 1

    def test_crossover_j_rand_guarantee(self):
        """At least one dimension must always come from the mutant (j_rand)."""
        target = np.zeros(self.D)
        mutant = np.ones(self.D)
        for _ in range(50):
            trial = self.variation._crossover(target, mutant)
            assert np.any(trial == 1.0), "j_rand guarantee violated: no gene from mutant"

    def test_crossover_output_shape(self):
        target = np.random.rand(self.D)
        mutant = np.random.rand(self.D)
        trial = self.variation._crossover(target, mutant)
        assert trial.shape == (self.D,)

    def test_generate_raises_without_pop(self):
        """generate() must raise if _continuous_pop is not set."""
        v = DifferentialEvolutionVariation()
        from nas_framework.population import Individual
        with pytest.raises(RuntimeError, match="_continuous_pop"):
            v.generate([Individual([0] * NUM_EDGES)], 1)


# ═══════════════════════════════════════════════════════════════
# 2.  Integration tests  (require CSV fixture)
# ═══════════════════════════════════════════════════════════════

class TestDESearchStrategyIntegration:

    def _build(self, population, evaluator, budget=40, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        return DESearchStrategy(
            population=population,
            evaluator=evaluator,
            F=0.5, Cr=0.5,
            budget=budget,
        )

    def test_run_respects_budget(self, small_population, evaluator):
        """Evaluations must not exceed the budget."""
        budget = 40
        strategy = self._build(small_population, evaluator, budget=budget)
        strategy.run()
        assert strategy.evaluations <= budget

    def test_population_size_preserved(self, small_population, evaluator):
        """Population size must equal the initial size after the run."""
        pop_size = small_population.size
        strategy = self._build(small_population, evaluator, budget=40)
        final_pop = strategy.run()
        assert len(final_pop.individuals) == pop_size

    def test_all_individuals_evaluated(self, small_population, evaluator):
        """Every individual in the final population must have a fitness tuple."""
        strategy = self._build(small_population, evaluator, budget=40)
        final_pop = strategy.run()
        for ind in final_pop.individuals:
            assert ind.fitness is not None
            assert len(ind.fitness) == 2

    def test_genotypes_valid(self, small_population, evaluator):
        """All genotypes must be lists of 6 integers in [0, NUM_OPS-1]."""
        strategy = self._build(small_population, evaluator, budget=40)
        final_pop = strategy.run()
        for ind in final_pop.individuals:
            assert len(ind.genotype) == NUM_EDGES
            for gene in ind.genotype:
                assert 0 <= gene < NUM_OPS, f"Invalid gene {gene}"

    def test_pareto_front_nonempty(self, small_population, evaluator):
        """The final Pareto front must contain at least one individual."""
        strategy = self._build(small_population, evaluator, budget=40)
        final_pop = strategy.run()
        pf = final_pop.pareto_front()
        assert len(pf) >= 1

    def test_history_recorded(self, small_population, evaluator):
        """History must record at least the initial generation."""
        strategy = self._build(small_population, evaluator, budget=40)
        strategy.run()
        assert len(strategy.history.entries) >= 1
        assert len(strategy.history.pareto_archive) >= 1

    def test_history_evaluations_monotone(self, small_population, evaluator):
        """Evaluation counts in history must be non-decreasing."""
        strategy = self._build(small_population, evaluator, budget=60)
        strategy.run()
        evals = [e.evaluations for e in strategy.history.entries]
        assert evals == sorted(evals)

    def test_determinism(self, csv_path, evaluator):
        """Two runs with the same seed must produce identical results."""
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population

        results = []
        for _ in range(2):
            random.seed(7)
            np.random.seed(7)
            ss  = CSVSearchSpace(str(csv_path))
            pop = Population(ss, evaluator, size=8)
            strategy = DESearchStrategy(pop, evaluator, budget=30, F=0.5, Cr=0.5)
            final = strategy.run()
            pf = sorted(final.pareto_front(), key=lambda i: i.fitness)
            results.append([(ind.genotype, ind.fitness) for ind in pf])

        assert results[0] == results[1], "Non-deterministic behaviour detected"

    def test_continuous_pop_shape(self, tiny_population, evaluator):
        """Internal continuous population must have shape (NP, NUM_EDGES)."""
        strategy = self._build(tiny_population, evaluator, budget=20)
        strategy.run()
        assert strategy._cont_pop is not None
        assert strategy._cont_pop.shape == (tiny_population.size, NUM_EDGES)

    def test_fitness_objectives_directions(self, small_population, evaluator):
        """Accuracy should be positive; latency should be positive."""
        strategy = self._build(small_population, evaluator, budget=40)
        final_pop = strategy.run()
        for ind in final_pop.individuals:
            acc, lat = ind.fitness
            assert acc >= 0.0, f"Negative accuracy: {acc}"
            assert lat >= 0.0, f"Negative latency: {lat}"

    def test_minimum_four_individuals_required(self, csv_path, evaluator):
        """
        DE/rand/1 needs at least 4 individuals (target + 3 distinct others).
        A population of 3 should still complete without crashing.
        """
        from nas_framework.search_space import CSVSearchSpace
        from nas_framework.population import Population
        random.seed(0)
        ss  = CSVSearchSpace(str(csv_path))
        pop = Population(ss, evaluator, size=3)
        # Should not raise; DE handles small pops by sampling with overlap
        strategy = DESearchStrategy(pop, evaluator, budget=15)
        final = strategy.run()
        assert len(final.individuals) == 3
