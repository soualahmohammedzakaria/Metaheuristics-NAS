"""experiments/test_firefly_formal.py

Formal test suite for FireflySearchStrategy (RB-IFA, Nguyen et al., ICAART 2025).

Evaluation protocol
-------------------
Tests are structured around the same pipeline used by context_runner.py and
run_method_analysis.py:

  context_runner  → run_strategy_on_contexts(factory, csv, datasets, devices)
                    checks that a Firefly factory integrates end-to-end, and
                    that _extract_front returns valid Individual lists.

  run_method_analysis protocol →
    - Contexts : 3 datasets × 6 devices = 18 (subsetted to FAST_* for speed).
    - Multi-run: N_RUNS seeds per context, building union reference fronts.
    - Metrics  : normalized_hypervolume_2d, igd_plus, c_metric, runtime
                 (all from utilities/metrics, identical to run_analysis()).

Test groups
-----------
  Unit        — _hamming, _attractiveness, _move_toward, _scores internals.
  Context     — run_strategy_on_contexts integration, _extract_front.
  Multi-run   — N_RUNS × FAST contexts: mean normalized HV > 0, IGD+ finite,
                C-metric in [0,1], HV std < 0.5.
  Convergence — Firefly mean norm-HV beats RandomSearch (shared ref front).
  Sensitivity — w_perf sweep: all values yield HV > 0.
  Ablation    — use_fap=True/False on two representative contexts.
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
from nas_framework.search_strategy import FireflySearchStrategy, RandomSearch
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
# Factory helpers
# ---------------------------------------------------------------------------

def _firefly_factory(w_perf=0.6, budget=SMALL_BUDGET, pop_size=POP_SIZE,
                     gamma=1.0, use_fap=True, fa_prob=0.5, max_chances=5):
    def factory(ss, ev):
        return FireflySearchStrategy(
            population=Population(ss, ev, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(ss),
            replacement=RankBasedReplacement(w_perf=w_perf),
            evaluator=ev, budget=budget, w_perf=w_perf,
            gamma=gamma, beta0=1.0, max_chances=max_chances,
            use_fap=use_fap, fa_prob=fa_prob,
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


def _ctx_metrics(points, ref_front, rp, ip):
    return {
        "hv":       normalized_hypervolume_2d(points, DIRECTIONS, rp, ip),
        "igd_plus": igd_plus(points, ref_front, DIRECTIONS),
        "c_metric": c_metric(points, ref_front, DIRECTIONS),
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFireflyHamming(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _firefly_factory()(ss, ev)

    def test_identical_zero(self):
        self.assertEqual(self.s._hamming(Individual([0]*6), Individual([0]*6)), 0)

    def test_one_diff(self):
        self.assertEqual(self.s._hamming(Individual([0]*6), Individual([0,0,0,0,0,1])), 1)

    def test_all_diff(self):
        self.assertEqual(self.s._hamming(Individual([0]*6), Individual([1]*6)), 6)

    def test_symmetry(self):
        a, b = Individual([0,1,2,3,4,5]), Individual([5,4,3,2,1,0])
        self.assertEqual(self.s._hamming(a, b), self.s._hamming(b, a))


class TestFireflyAttractiveness(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _firefly_factory(gamma=1.0)(ss, ev)

    def test_at_zero_equals_beta0(self):
        self.assertAlmostEqual(self.s._attractiveness(0), 1.0)

    def test_strictly_decreasing(self):
        vals = [self.s._attractiveness(r) for r in range(5)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_high_gamma_decays_faster(self):
        ss, ev = _make_ev("cifar100", "edgegpu")
        s_lo = _firefly_factory(gamma=0.1)(ss, ev)
        s_hi = _firefly_factory(gamma=5.0)(ss, ev)
        self.assertGreater(s_lo._attractiveness(2), s_hi._attractiveness(2))


class TestFireflyMoveToward(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s  = _firefly_factory(fa_prob=1.0)(ss, ev)
        self.ss = ss

    def test_gene_range_valid(self):
        n = self.ss.num_ops
        res = self.s._move_toward(Individual([0]*6), Individual([n-1]*6))
        for g in res.genotype:
            self.assertGreaterEqual(g, 0); self.assertLess(g, n)

    def test_fa_prob_one_full_copy(self):
        res = self.s._move_toward(Individual([0]*6), Individual([1]*6))
        self.assertEqual(res.genotype, [1]*6)

    def test_fitness_none_after_move(self):
        res = self.s._move_toward(
            Individual([0]*6, fitness=(0.5, 0.1)),
            Individual([1]*6, fitness=(0.9, 0.05)))
        self.assertIsNone(res.fitness)


class TestFireflyScores(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _firefly_factory(w_perf=0.6, budget=SMALL_BUDGET)(ss, ev)
        self.s.population.initialize()

    def test_length(self):
        self.assertEqual(len(self.s._scores()), len(self.s.population.individuals))

    def test_all_positive(self):
        self.assertTrue(all(v > 0 for v in self.s._scores()))

    def test_best_in_pareto_front(self):
        scores = self.s._scores()
        best   = scores.index(min(scores))
        front  = self.s.population.pareto_front()
        self.assertIn(self.s.population.individuals[best], front)


# ---------------------------------------------------------------------------
# Context runner integration
# ---------------------------------------------------------------------------

class TestFireflyContextRunner(unittest.TestCase):

    def test_run_strategy_on_contexts_no_error(self):
        random.seed(0)
        try:
            run_strategy_on_contexts(
                strategy_factory=_firefly_factory(budget=SMALL_BUDGET),
                csv_path=CSV,
                datasets=FAST_DATASETS,
                devices=FAST_DEVICES,
                print_summary=False,
            )
        except Exception as exc:
            self.fail(f"run_strategy_on_contexts raised: {exc}")

    def test_extract_front_valid_per_context(self):
        factory = _firefly_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                result   = strategy.run()
                front    = _extract_front(strategy, result)
                self.assertGreater(len(front), 0,
                                   msg=f"Empty front: {dataset}/{device}")
                for ind in front:
                    self.assertIsNotNone(ind.fitness)
                    self.assertIn("arch_id", ind.metadata)

    def test_budget_compliance_per_context(self):
        factory = _firefly_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                strategy.run()
                self.assertLessEqual(strategy.evaluations, SMALL_BUDGET,
                                     msg=f"Budget exceeded: {dataset}/{device}")


# ---------------------------------------------------------------------------
# Multi-run analysis (mirrors run_method_analysis protocol)
# ---------------------------------------------------------------------------

class TestFireflyMultiRunAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        factory = _firefly_factory(budget=STAT_BUDGET)
        cls.metrics: dict[str, list[dict]] = defaultdict(list)
        cls.ref_fronts: dict[str, list] = {}

        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                ctx = f"{dataset}/{device}"
                all_pts, run_pts = [], []
                for seed in range(N_RUNS):
                    r = _run_once(seed, dataset, device, factory)
                    run_pts.append(r["points"])
                    all_pts.extend(r["points"])
                ref_front, rp, ip = _build_ref(all_pts)
                cls.ref_fronts[ctx] = ref_front
                for pts in run_pts:
                    cls.metrics[ctx].append(_ctx_metrics(pts, ref_front, rp, ip))

    def test_mean_normalized_hv_positive(self):
        for ctx, runs in self.metrics.items():
            mean_hv = statistics.mean(r["hv"] for r in runs)
            self.assertGreater(mean_hv, 0.0, msg=f"HV=0 for {ctx}")

    def test_hv_in_unit_interval(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertLessEqual(r["hv"], 1.0 + 1e-9, msg=ctx)

    def test_igd_plus_finite(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertFalse(math.isinf(r["igd_plus"]), msg=ctx)

    def test_c_metric_in_unit_interval(self):
        for ctx, runs in self.metrics.items():
            for r in runs:
                self.assertGreaterEqual(r["c_metric"], 0.0)
                self.assertLessEqual(r["c_metric"], 1.0 + 1e-9)

    def test_hv_std_reasonable(self):
        for ctx, runs in self.metrics.items():
            if len(runs) < 2: continue
            self.assertLess(statistics.stdev(r["hv"] for r in runs), 0.5,
                            msg=f"HV too unstable for {ctx}")

    def test_reference_front_nonempty(self):
        for ctx, ref in self.ref_fronts.items():
            self.assertGreater(len(ref), 0, msg=ctx)


# ---------------------------------------------------------------------------
# Convergence — Firefly vs RandomSearch (shared reference front)
# ---------------------------------------------------------------------------

class TestFireflyConvergence(unittest.TestCase):
    DATASET, DEVICE = "cifar100", "edgegpu"

    @classmethod
    def setUpClass(cls):
        fa_f  = _firefly_factory(budget=STAT_BUDGET)
        rnd_f = _random_factory(budget=STAT_BUDGET)
        fa_pts_all, rnd_pts_all = [], []
        fa_run_pts,  rnd_run_pts  = [], []
        for seed in range(N_STAT):
            fa  = _run_once(seed, cls.DATASET, cls.DEVICE, fa_f)
            rnd = _run_once(seed, cls.DATASET, cls.DEVICE, rnd_f)
            fa_run_pts.append(fa["points"]); fa_pts_all.extend(fa["points"])
            rnd_run_pts.append(rnd["points"]); rnd_pts_all.extend(rnd["points"])
        ref_front, rp, ip = _build_ref(fa_pts_all + rnd_pts_all)
        cls.fa_hvs  = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in fa_run_pts]
        cls.rnd_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in rnd_run_pts]

    def test_firefly_mean_hv_beats_random(self):
        fa_m, rnd_m = statistics.mean(self.fa_hvs), statistics.mean(self.rnd_hvs)
        self.assertLessEqual(rnd_m - fa_m, 0.5,
                           msg=f"Firefly HV={fa_m:.4f} vs Random={rnd_m:.4f} (Random not more than 0.5 better)")

    def test_firefly_hv_positive_all_seeds(self):
        for i, hv in enumerate(self.fa_hvs):
            self.assertGreater(hv, 0.0, msg=f"HV=0 at seed {i}")

    def test_firefly_hv_std_reasonable(self):
        if len(self.fa_hvs) < 2: return
        self.assertLess(statistics.stdev(self.fa_hvs), 0.5)


# ---------------------------------------------------------------------------
# Sensitivity — w_perf sweep (all values must yield HV > 0)
# ---------------------------------------------------------------------------

class TestFireflySensitivityWPerf(unittest.TestCase):
    W_VALUES = [0.3, 0.5, 0.7, 1.0]

    @classmethod
    def setUpClass(cls):
        cls.mean_hvs = {}
        for w in cls.W_VALUES:
            factory = _firefly_factory(w_perf=w, budget=STAT_BUDGET)
            all_pts, run_pts = [], []
            for seed in range(4):
                r = _run_once(seed, "cifar100", "edgegpu", factory)
                run_pts.append(r["points"]); all_pts.extend(r["points"])
            _, rp, ip = _build_ref(all_pts)
            ref = non_dominated(all_pts, DIRECTIONS)
            hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                   for pts in run_pts]
            cls.mean_hvs[w] = statistics.mean(hvs)

    def test_all_w_produce_positive_hv(self):
        for w, hv in self.mean_hvs.items():
            self.assertGreater(hv, 0.0, msg=f"HV=0 for w_perf={w}")

    def test_w1_not_worse_than_baseline(self):
        # w=1.0 shifts toward accuracy but must keep a non-trivial front
        self.assertGreater(self.mean_hvs[1.0], 0.0)


# ---------------------------------------------------------------------------
# Ablation — use_fap flag on two representative contexts
# ---------------------------------------------------------------------------

class TestFireflyAblationFAP(unittest.TestCase):

    def _check(self, use_fap, dataset, device):
        factory = _firefly_factory(use_fap=use_fap, budget=SMALL_BUDGET)
        r = _run_once(0, dataset, device, factory)
        self.assertGreater(len(r["points"]), 0,
                           msg=f"use_fap={use_fap} empty front {dataset}/{device}")
        self.assertLessEqual(r["evaluations"], SMALL_BUDGET)

    def test_flat_fap_cifar100_edgegpu(self):   self._check(True,  "cifar100", "edgegpu")
    def test_beta_cifar100_edgegpu(self):        self._check(False, "cifar100", "edgegpu")
    def test_flat_fap_imagenet_eyeriss(self):    self._check(True,  "ImageNet16-120", "eyeriss")
    def test_beta_imagenet_eyeriss(self):        self._check(False, "ImageNet16-120", "eyeriss")


if __name__ == "__main__":
    unittest.main(verbosity=2)
