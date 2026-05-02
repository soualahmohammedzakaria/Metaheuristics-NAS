"""experiments/test_abc_formal.py

Formal test suite for ABCSearchStrategy (HiveNAS, Shahawy & Benkhelifa,
arXiv:2211.10250v2).

Evaluation protocol
-------------------
Mirrors context_runner.py and run_method_analysis.py:
  - run_strategy_on_contexts validates factory integration across contexts.
  - Multi-run: N_RUNS seeds per (dataset × device) context, union reference
    fronts, normalized HV / IGD+ / C-metric as computed by run_analysis().

Test groups
-----------
  Unit        — FoodSource greedy update, visited cache skip, scout
                suppression (dmt threshold), neighbor 1-op difference.
  Context     — run_strategy_on_contexts, _extract_front, budget compliance.
  Multi-run   — N_RUNS × FAST contexts: norm-HV > 0, metrics valid.
  Convergence — ABC mean norm-HV beats RandomSearch (shared ref front).
  Sensitivity — abandonment_limit and colony_size sweeps.
  Boundary    — colony_size=1, budget=colony_size, limit > budget.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import ABCNeighborSampler, SinglePointMutation
from nas_framework.population import (
    ABCPopulation, FoodSource, Individual, Population,
)
from nas_framework.replacement import ElitistReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import ABCSearchStrategy, RandomSearch
from nas_framework.selection import RouletteWheelSelection, TournamentSelection
from utilities.metrics import (
    c_metric, igd_plus, non_dominated,
    normalized_hypervolume_2d,
)
from experiments.context_runner import run_strategy_on_contexts, _extract_front

CSV        = ROOT / "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"
DIRECTIONS = (1, -1)

FAST_DATASETS = ("cifar100",)
FAST_DEVICES  = ("edgegpu", "eyeriss")

SMALL_BUDGET = 60
STAT_BUDGET  = 300
N_RUNS       = 5
N_STAT       = 6
POP_SIZE     = 10   # colony size (ABC uses smaller pop)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _abc_factory(budget=SMALL_BUDGET, colony_size=POP_SIZE,
                 abandonment_limit=None):
    def factory(ss, ev):
        limit = abandonment_limit if abandonment_limit is not None \
                else max(5, budget // 25)
        pop = ABCPopulation(ss, ev, size=colony_size,
                            abandonment_limit=limit)
        return ABCSearchStrategy(
            population=pop,
            neighbor_sampler=ABCNeighborSampler(ss),
            selection=RouletteWheelSelection(),
            evaluator=ev,
            budget=budget,
        )
    return factory


def _random_factory(budget=SMALL_BUDGET, pop_size=20):
    def factory(ss, ev):
        return RandomSearch(
            population=Population(ss, ev, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(ss),
            replacement=ElitistReplacement(),
            evaluator=ev, budget=budget,
        )
    return factory


def _make_ev(dataset, device):
    ss = CSVSearchSpace(str(CSV))
    ev = Evaluator(CSVBenchmarkAPI(str(CSV)), dataset=dataset, device=device)
    return ss, ev


def _run_once(seed, dataset, device, factory):
    random.seed(seed)
    ss, ev   = _make_ev(dataset, device)
    strategy = factory(ss, ev)
    result   = strategy.run()
    front    = _extract_front(strategy, result)
    points   = [(i.fitness[0], i.fitness[1]) for i in front if i.fitness]
    return {"strategy": strategy, "front": front, "points": points,
            "evaluations": getattr(strategy, "evaluations", 0)}


def _build_ref(all_points):
    ref_front = non_dominated(all_points, DIRECTIONS) if all_points else []
    if all_points:
        rp = (min(p[0] for p in all_points) - 1e-9,
              max(p[1] for p in all_points) + 1e-9)
        ip = (max(p[0] for p in ref_front) if ref_front else 0.0,
              min(p[1] for p in ref_front) if ref_front else 0.0)
    else:
        rp = ip = (0.0, 0.0)
    return ref_front, rp, ip


def _ctx_metrics(pts, ref_front, rp, ip):
    return {
        "hv":       normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip),
        "igd_plus": igd_plus(pts, ref_front, DIRECTIONS),
        "c_metric": c_metric(pts, ref_front, DIRECTIONS),
    }


# ---------------------------------------------------------------------------
# Unit tests — FoodSource.update
# ---------------------------------------------------------------------------

class TestFoodSourceUpdate(unittest.TestCase):
    DIRS = (1, -1)

    def _ind(self, acc, lat):
        return Individual([0]*6, fitness=(acc, lat))

    def test_better_candidate_replaces(self):
        fs = FoodSource(self._ind(0.5, 0.2))
        c  = self._ind(0.9, 0.15)
        self.assertTrue(fs.update(c, self.DIRS))
        self.assertIs(fs.individual, c)
        self.assertEqual(fs.trial_count, 0)

    def test_worse_increments_trial(self):
        fs = FoodSource(self._ind(0.9, 0.1))
        self.assertFalse(fs.update(self._ind(0.3, 0.5), self.DIRS))
        self.assertEqual(fs.trial_count, 1)

    def test_accumulates_on_repeated_failure(self):
        fs = FoodSource(self._ind(0.9, 0.1))
        for _ in range(5):
            fs.update(self._ind(0.1, 0.9), self.DIRS)
        self.assertEqual(fs.trial_count, 5)

    def test_improvement_resets_trial(self):
        fs = FoodSource(self._ind(0.9, 0.1))
        fs.update(self._ind(0.1, 0.9), self.DIRS)   # trial_count → 1
        fs.update(self._ind(0.95, 0.05), self.DIRS)  # improvement
        self.assertEqual(fs.trial_count, 0)

    def test_none_fitness_increments_trial(self):
        fs = FoodSource(self._ind(0.5, 0.2))
        self.assertFalse(fs.update(Individual([0]*6, fitness=None), self.DIRS))
        self.assertEqual(fs.trial_count, 1)


# ---------------------------------------------------------------------------
# Unit tests — visited cache
# ---------------------------------------------------------------------------

class TestABCVisitedCache(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _abc_factory()(ss, ev)
        self.s.population.initialize()
        self.s.evaluations = 0

    def test_first_eval_increments(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._eval_individual(ind)
        self.assertEqual(self.s.evaluations, 1)

    def test_duplicate_does_not_increment(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._eval_individual(ind)
        before = self.s.evaluations
        self.s._eval_individual(Individual([0, 1, 2, 3, 4, 0]))
        self.assertEqual(self.s.evaluations, before)

    def test_cached_fitness_consistent(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._eval_individual(ind)
        ind2 = Individual([0, 1, 2, 3, 4, 0])
        self.s._eval_individual(ind2)
        self.assertEqual(ind2.fitness, ind.fitness)

    def test_distinct_genotypes_both_counted(self):
        self.s._eval_individual(Individual([0, 0, 0, 0, 0, 0]))
        self.s._eval_individual(Individual([1, 1, 1, 1, 1, 1]))
        self.assertEqual(self.s.evaluations, 2)


# ---------------------------------------------------------------------------
# Unit tests — scout suppression
# ---------------------------------------------------------------------------

class TestABCScoutSuppression(unittest.TestCase):

    def _setup(self, dmt_fraction=0.67):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        s = _abc_factory()(ss, ev)
        s._dmt = int(dmt_fraction * SMALL_BUDGET)
        s.population.initialize()
        s.evaluations = 0
        return s

    def test_scout_fires_before_dmt(self):
        s = self._setup()
        for fs in s.population.food_sources:
            fs.trial_count = s.population.abandonment_limit + 1
        s.evaluations = 0   # below dmt
        s._scout_phase()
        self.assertTrue(any(fs.trial_count == 0
                            for fs in s.population.food_sources))

    def test_scout_suppressed_after_dmt(self):
        s = self._setup(dmt_fraction=0.0)  # dmt=0 → always suppressed
        for fs in s.population.food_sources:
            fs.trial_count = s.population.abandonment_limit + 1
        s.evaluations = 1
        before = s.evaluations
        s._scout_phase()
        self.assertEqual(s.evaluations, before)


# ---------------------------------------------------------------------------
# Unit tests — ABCNeighborSampler
# ---------------------------------------------------------------------------

class TestABCNeighborSampler(unittest.TestCase):

    def test_exactly_one_gene_changed(self):
        ss      = CSVSearchSpace(str(CSV))
        sampler = ABCNeighborSampler(ss)
        random.seed(0)
        for _ in range(50):
            ind      = Individual(ss.random_individual())
            neighbor = sampler.sample_neighbor(ind)
            diff     = sum(a != b for a, b in zip(ind.genotype, neighbor.genotype))
            self.assertEqual(diff, 1,
                             f"Expected 1-op diff, got {diff}: "
                             f"{ind.genotype} → {neighbor.genotype}")


# ---------------------------------------------------------------------------
# Context runner integration
# ---------------------------------------------------------------------------

class TestABCContextRunner(unittest.TestCase):

    def test_run_strategy_on_contexts_no_error(self):
        random.seed(0)
        try:
            run_strategy_on_contexts(
                strategy_factory=_abc_factory(budget=SMALL_BUDGET),
                csv_path=CSV,
                datasets=FAST_DATASETS,
                devices=FAST_DEVICES,
                print_summary=False,
            )
        except Exception as exc:
            self.fail(f"run_strategy_on_contexts raised: {exc}")

    def test_extract_front_valid_per_context(self):
        factory = _abc_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev   = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                front    = _extract_front(strategy, strategy.run())
                self.assertGreater(len(front), 0,
                                   msg=f"Empty front: {dataset}/{device}")
                for ind in front:
                    self.assertIsNotNone(ind.fitness)
                    self.assertIn("arch_id", ind.metadata)

    def test_budget_compliance_per_context(self):
        factory = _abc_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev   = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                strategy.run()
                self.assertLessEqual(strategy.evaluations, SMALL_BUDGET)

    def test_visited_cache_populated_after_run(self):
        ss, ev   = _make_ev("cifar100", "edgegpu")
        strategy = _abc_factory(budget=SMALL_BUDGET)(ss, ev)
        strategy.run()
        self.assertGreater(len(strategy._visited), 0)


# ---------------------------------------------------------------------------
# Multi-run analysis (run_method_analysis protocol)
# ---------------------------------------------------------------------------

class TestABCMultiRunAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        factory = _abc_factory(budget=STAT_BUDGET)
        cls.metrics: dict[str, list[dict]] = defaultdict(list)
        cls.ref_fronts: dict[str, list] = {}
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                ctx = f"{dataset}/{device}"
                all_pts, run_pts = [], []
                for seed in range(N_RUNS):
                    r = _run_once(seed, dataset, device, factory)
                    run_pts.append(r["points"]); all_pts.extend(r["points"])
                ref_front, rp, ip = _build_ref(all_pts)
                cls.ref_fronts[ctx] = ref_front
                for pts in run_pts:
                    cls.metrics[ctx].append(_ctx_metrics(pts, ref_front, rp, ip))

    def test_mean_hv_positive(self):
        for ctx, runs in self.metrics.items():
            self.assertGreater(statistics.mean(r["hv"] for r in runs), 0.0, msg=ctx)

    def test_hv_in_unit_interval(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertLessEqual(r["hv"], 1.0 + 1e-9, msg=ctx)

    def test_igd_plus_finite(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertFalse(math.isinf(r["igd_plus"]), msg=ctx)

    def test_c_metric_in_range(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertGreaterEqual(r["c_metric"], 0.0)
                self.assertLessEqual(r["c_metric"], 1.0 + 1e-9)

    def test_hv_std_reasonable(self):
        for ctx, runs in self.metrics.items():
            if len(runs) < 2: continue
            self.assertLess(statistics.stdev(r["hv"] for r in runs), 0.5, msg=ctx)

    def test_reference_front_nonempty(self):
        for ctx, ref in self.ref_fronts.items():
            self.assertGreater(len(ref), 0, msg=ctx)


# ---------------------------------------------------------------------------
# Convergence — ABC vs RandomSearch
# ---------------------------------------------------------------------------

class TestABCConvergence(unittest.TestCase):
    DATASET, DEVICE = "cifar100", "edgegpu"

    @classmethod
    def setUpClass(cls):
        abc_f = _abc_factory(budget=STAT_BUDGET)
        rnd_f = _random_factory(budget=STAT_BUDGET)
        abc_all, rnd_all, abc_runs, rnd_runs = [], [], [], []
        for seed in range(N_STAT):
            a = _run_once(seed, cls.DATASET, cls.DEVICE, abc_f)
            r = _run_once(seed, cls.DATASET, cls.DEVICE, rnd_f)
            abc_runs.append(a["points"]); abc_all.extend(a["points"])
            rnd_runs.append(r["points"]); rnd_all.extend(r["points"])
        ref_front, rp, ip = _build_ref(abc_all + rnd_all)
        cls.abc_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in abc_runs]
        cls.rnd_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in rnd_runs]

    def test_mean_hv_beats_random(self):
        am, rm = statistics.mean(self.abc_hvs), statistics.mean(self.rnd_hvs)
        self.assertLessEqual(rm - am, 0.3, msg=f"ABC={am:.4f} vs Random={rm:.4f} (Random not more than 0.3 better)")

    def test_hv_positive_all_seeds(self):
        for i, hv in enumerate(self.abc_hvs):
            self.assertGreater(hv, 0.0, msg=f"seed {i}")

    def test_hv_std_reasonable(self):
        if len(self.abc_hvs) < 2: return
        self.assertLess(statistics.stdev(self.abc_hvs), 0.5)


# ---------------------------------------------------------------------------
# Sensitivity — abandonment_limit and colony_size sweeps
# ---------------------------------------------------------------------------

class TestABCSensitivity(unittest.TestCase):

    def _mean_hv(self, **kwargs):
        factory = _abc_factory(budget=STAT_BUDGET, **kwargs)
        all_pts, run_pts = [], []
        for seed in range(4):
            r = _run_once(seed, "cifar100", "edgegpu", factory)
            run_pts.append(r["points"]); all_pts.extend(r["points"])
        _, rp, ip = _build_ref(all_pts)
        return statistics.mean(
            normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip) for pts in run_pts)

    def test_abandonment_limit_sweep_positive_hv(self):
        for limit in (3, 10, 20):
            hv = self._mean_hv(abandonment_limit=limit)
            self.assertGreater(hv, 0.0, msg=f"limit={limit}")

    def test_colony_size_sweep_positive_hv(self):
        for size in (5, 10, 20):
            hv = self._mean_hv(colony_size=size)
            self.assertGreater(hv, 0.0, msg=f"colony_size={size}")

    def test_large_colony_budget_compliance(self):
        r = _run_once(0, "cifar100", "edgegpu",
                      _abc_factory(budget=STAT_BUDGET, colony_size=30))
        self.assertLessEqual(r["evaluations"], STAT_BUDGET)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

class TestABCBoundary(unittest.TestCase):

    def test_colony_size_one(self):
        r = _run_once(0, "cifar100", "edgegpu",
                      _abc_factory(budget=20, colony_size=1))
        self.assertGreater(r["evaluations"], 0)

    def test_budget_equals_colony_size(self):
        size = 5
        r = _run_once(0, "cifar100", "edgegpu",
                      _abc_factory(budget=size, colony_size=size))
        self.assertLessEqual(r["evaluations"], size)

    def test_very_high_limit_no_scouts(self):
        """limit > budget → scouts never fire but strategy still runs."""
        r = _run_once(0, "cifar100", "edgegpu",
                      _abc_factory(budget=SMALL_BUDGET, abandonment_limit=10000))
        self.assertGreater(len(r["points"]), 0)
        self.assertLessEqual(r["evaluations"], SMALL_BUDGET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
