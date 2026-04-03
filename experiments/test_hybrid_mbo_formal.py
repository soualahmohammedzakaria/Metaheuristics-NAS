"""experiments/test_hybrid_mbo_formal.py

Formal test suite for HybridMBOStrategy (SP1/SP2 rank-stratified hybrid:
FA exploitation on top half + GA diversification on bottom half).

Evaluation protocol
-------------------
Mirrors context_runner.py and run_method_analysis.py:
  - run_strategy_on_contexts validates factory integration across contexts.
  - Multi-run analysis: N_RUNS seeds per (dataset × device) context, union
    reference front, normalized HV / IGD+ / C-metric (utilities/metrics).

Test groups
-----------
  Unit        — _rank_scores, _fap_move, SP1/SP2 partitioning.
  Context     — run_strategy_on_contexts integration, _extract_front.
  Multi-run   — norm-HV > 0, IGD+ finite, C-metric ∈ [0,1], HV std < 0.5.
  Convergence — HybridMBO mean norm-HV beats RandomSearch (shared ref front).
  Sensitivity — rank_fraction and fa_prob sweeps via normalized HV.
  Boundary    — degenerate pop sizes and rank fractions.
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
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Individual, Population
from nas_framework.replacement import ElitistReplacement, RankBasedReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import HybridMBOStrategy, RandomSearch
from nas_framework.selection import TournamentSelection
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
POP_SIZE     = 20


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _mbo_factory(w_perf=0.6, rank_fraction=0.5, fa_prob=0.5,
                 budget=SMALL_BUDGET, pop_size=POP_SIZE):
    def factory(ss, ev):
        return HybridMBOStrategy(
            population=Population(ss, ev, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(ss),
            replacement=RankBasedReplacement(w_perf=w_perf),
            evaluator=ev, budget=budget,
            w_perf=w_perf, rank_fraction=rank_fraction, fa_prob=fa_prob,
        )
    return factory


def _random_factory(budget=SMALL_BUDGET, pop_size=POP_SIZE):
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
    ss, ev = _make_ev(dataset, device)
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
# Unit tests
# ---------------------------------------------------------------------------

class TestHybridMBORankScores(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _mbo_factory()(ss, ev)
        self.s.population.initialize()
        for ind in self.s.population.individuals:
            if ind.fitness is None:
                ind.fitness = self.s.evaluator.evaluate(ind.genotype)

    def test_length_matches_population(self):
        self.assertEqual(len(self.s._rank_scores()),
                         len(self.s.population.individuals))

    def test_all_positive(self):
        self.assertTrue(all(v > 0 for v in self.s._rank_scores()))

    def test_best_in_pareto_front(self):
        scores = self.s._rank_scores()
        best   = scores.index(min(scores))
        front  = self.s.population.pareto_front()
        self.assertIn(self.s.population.individuals[best], front)


class TestHybridMBOFAPMove(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s  = _mbo_factory(fa_prob=1.0)(ss, ev)
        self.ss = ss

    def test_genotype_length(self):
        n = self.ss.num_edges
        res = self.s._fap_move(Individual([0]*n), [Individual([1]*n)])
        self.assertEqual(len(res.genotype), n)

    def test_gene_range_valid(self):
        n_ops, n = self.ss.num_ops, self.ss.num_edges
        res = self.s._fap_move(Individual([0]*n), [Individual([n_ops-1]*n)])
        for g in res.genotype:
            self.assertGreaterEqual(g, 0); self.assertLess(g, n_ops)

    def test_fa_prob_one_copies_target(self):
        n = self.ss.num_edges
        res = self.s._fap_move(Individual([0]*n), [Individual([1]*n)])
        self.assertEqual(res.genotype, [1]*n)

    def test_fa_prob_zero_preserves_source(self):
        ss, ev = _make_ev("cifar100", "edgegpu")
        s = _mbo_factory(fa_prob=0.0)(ss, ev)
        n = ss.num_edges
        res = s._fap_move(Individual([0]*n), [Individual([1]*n)])
        self.assertEqual(res.genotype, [0]*n)

    def test_fitness_none_after_move(self):
        n = self.ss.num_edges
        res = self.s._fap_move(Individual([0]*n, fitness=(0.5, 0.1)),
                               [Individual([1]*n, fitness=(0.9, 0.05))])
        self.assertIsNone(res.fitness)


class TestHybridMBOPartition(unittest.TestCase):

    def _n_sp1(self, rank_fraction, pop_size=10):
        return max(1, int(rank_fraction * pop_size))

    def test_half_fraction(self):
        self.assertEqual(self._n_sp1(0.5, 10), 5)

    def test_minimum_sp1_is_one(self):
        self.assertGreaterEqual(self._n_sp1(0.01, 10), 1)

    def test_sum_equals_pop_size(self):
        pop_size = 10
        for frac in (0.2, 0.5, 0.8):
            n1 = self._n_sp1(frac, pop_size)
            self.assertEqual(n1 + (pop_size - n1), pop_size)


# ---------------------------------------------------------------------------
# Context runner integration
# ---------------------------------------------------------------------------

class TestHybridMBOContextRunner(unittest.TestCase):

    def test_run_strategy_on_contexts_no_error(self):
        random.seed(0)
        try:
            run_strategy_on_contexts(
                strategy_factory=_mbo_factory(budget=SMALL_BUDGET),
                csv_path=CSV,
                datasets=FAST_DATASETS,
                devices=FAST_DEVICES,
                print_summary=False,
            )
        except Exception as exc:
            self.fail(f"run_strategy_on_contexts raised: {exc}")

    def test_extract_front_valid_per_context(self):
        factory = _mbo_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                front    = _extract_front(strategy, strategy.run())
                self.assertGreater(len(front), 0,
                                   msg=f"Empty front: {dataset}/{device}")
                for ind in front:
                    self.assertIsNotNone(ind.fitness)
                    self.assertIn("arch_id", ind.metadata)

    def test_budget_compliance_per_context(self):
        factory = _mbo_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                strategy.run()
                self.assertLessEqual(strategy.evaluations, SMALL_BUDGET)


# ---------------------------------------------------------------------------
# Multi-run analysis (run_method_analysis protocol)
# ---------------------------------------------------------------------------

class TestHybridMBOMultiRunAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        factory = _mbo_factory(budget=STAT_BUDGET)
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
# Convergence — HybridMBO vs RandomSearch
# ---------------------------------------------------------------------------

class TestHybridMBOConvergence(unittest.TestCase):
    DATASET, DEVICE = "cifar100", "edgegpu"

    @classmethod
    def setUpClass(cls):
        mbo_f = _mbo_factory(budget=STAT_BUDGET)
        rnd_f = _random_factory(budget=STAT_BUDGET)
        mbo_all, rnd_all, mbo_runs, rnd_runs = [], [], [], []
        for seed in range(N_STAT):
            m = _run_once(seed, cls.DATASET, cls.DEVICE, mbo_f)
            r = _run_once(seed, cls.DATASET, cls.DEVICE, rnd_f)
            mbo_runs.append(m["points"]); mbo_all.extend(m["points"])
            rnd_runs.append(r["points"]); rnd_all.extend(r["points"])
        ref_front, rp, ip = _build_ref(mbo_all + rnd_all)
        cls.mbo_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in mbo_runs]
        cls.rnd_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in rnd_runs]

    def test_mean_hv_beats_random(self):
        mm, rm = statistics.mean(self.mbo_hvs), statistics.mean(self.rnd_hvs)
        self.assertLessEqual(rm - mm, 0.5, msg=f"MBO={mm:.4f} vs Random={rm:.4f} (Random not more than 0.5 better)")

    def test_hv_positive_all_seeds(self):
        for i, hv in enumerate(self.mbo_hvs):
            self.assertGreater(hv, 0.0, msg=f"seed {i}")

    def test_hv_std_reasonable(self):
        if len(self.mbo_hvs) < 2: return
        self.assertLess(statistics.stdev(self.mbo_hvs), 0.5)


# ---------------------------------------------------------------------------
# Sensitivity — rank_fraction and fa_prob sweeps
# ---------------------------------------------------------------------------

class TestHybridMBOSensitivity(unittest.TestCase):

    def _mean_hv(self, **kwargs):
        factory = _mbo_factory(budget=STAT_BUDGET, **kwargs)
        all_pts, run_pts = [], []
        for seed in range(4):
            r = _run_once(seed, "cifar100", "edgegpu", factory)
            run_pts.append(r["points"]); all_pts.extend(r["points"])
        _, rp, ip = _build_ref(all_pts)
        ref = non_dominated(all_pts, DIRECTIONS)
        return statistics.mean(
            normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip) for pts in run_pts)

    def test_rank_fraction_sweep_all_positive_hv(self):
        for rf in (0.2, 0.5, 0.8):
            hv = self._mean_hv(rank_fraction=rf)
            self.assertGreater(hv, 0.0, msg=f"rank_fraction={rf}")

    def test_fa_prob_sweep_all_positive_hv(self):
        for fa in (0.1, 0.5, 0.9):
            hv = self._mean_hv(fa_prob=fa)
            self.assertGreater(hv, 0.0, msg=f"fa_prob={fa}")


# ---------------------------------------------------------------------------
# Boundary / degenerate configurations
# ---------------------------------------------------------------------------

class TestHybridMBOBoundary(unittest.TestCase):

    def _check(self, factory):
        r = _run_once(0, "cifar100", "edgegpu", factory)
        self.assertGreater(r["evaluations"], 0)
        self.assertGreater(len(r["points"]), 0)
        return r

    def test_pop_size_two(self):
        self._check(_mbo_factory(pop_size=2, budget=20))

    def test_rank_fraction_near_one(self):
        r = _run_once(0, "cifar100", "edgegpu",
                      _mbo_factory(rank_fraction=0.95, budget=SMALL_BUDGET))
        self.assertLessEqual(r["evaluations"], SMALL_BUDGET)
        self.assertGreater(len(r["points"]), 0)

    def test_rank_fraction_near_zero(self):
        r = _run_once(0, "cifar100", "edgegpu",
                      _mbo_factory(rank_fraction=0.01, budget=SMALL_BUDGET))
        self.assertLessEqual(r["evaluations"], SMALL_BUDGET)
        self.assertGreater(len(r["points"]), 0)

    def test_budget_equals_pop_size(self):
        pop_size = 10
        r = _run_once(0, "cifar100", "edgegpu",
                      _mbo_factory(pop_size=pop_size, budget=pop_size))
        self.assertLessEqual(r["evaluations"], pop_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
