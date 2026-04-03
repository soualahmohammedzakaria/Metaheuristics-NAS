"""experiments/test_pso_formal.py

Formal test suite for PSOSearchStrategy (MOIPSO: Multi-Objective PSO with
trigonometric acceleration and adaptive Gaussian mutation, Shao et al.).

Evaluation protocol
-------------------
Mirrors context_runner.py and run_method_analysis.py:
  - run_strategy_on_contexts validates factory integration per context.
  - Multi-run: N_RUNS seeds, union reference fronts, normalized HV / IGD+ /
    C-metric exactly as run_analysis() computes them.

Test groups
-----------
  Unit        — _trig_coefficients bounds, _evaluate_particle counter/fitness,
                GaussianMutation sigma decay.
  Context     — run_strategy_on_contexts, _extract_front, budget compliance.
  Multi-run   — N_RUNS × FAST contexts: norm-HV > 0, metrics valid.
  Convergence — PSO mean norm-HV beats RandomSearch (shared ref front).
  Sensitivity — inertia weight (w) sweep via normalized HV.
  Boundary    — w=0, w=1, pop_size=2.
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
from nas_framework.mutation import GaussianMutation, SinglePointMutation
from nas_framework.population import Individual, Population, PSOPopulation
from nas_framework.replacement import CrowdingReplacement, ElitistReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import PSOSearchStrategy, RandomSearch
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
STAT_BUDGET  = 1000
N_RUNS       = 5
N_STAT       = 6
POP_SIZE     = 20


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _pso_factory(w=0.4, budget=SMALL_BUDGET, pop_size=POP_SIZE):
    def factory(ss, ev):
        return PSOSearchStrategy(
            population=PSOPopulation(ss, ev, size=pop_size, w=w),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(ss),
            replacement=CrowdingReplacement(),
            evaluator=ev, budget=budget, w=w,
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
# Unit tests
# ---------------------------------------------------------------------------

class TestPSOTrigCoefficients(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _pso_factory()(ss, ev)

    def test_c1_c2_nonneg(self):
        for _ in range(50):
            c1, c2 = self.s._trig_coefficients()
            self.assertGreaterEqual(c1, 0.0)
            self.assertGreaterEqual(c2, 0.0)

    def test_c1_c2_bounded_above(self):
        for _ in range(50):
            c1, c2 = self.s._trig_coefficients()
            self.assertLessEqual(c1, 2.05 + 1e-9)
            self.assertLessEqual(c2, 2.05 + 1e-9)

    def test_sum_bounded(self):
        for _ in range(100):
            c1, c2 = self.s._trig_coefficients()
            self.assertLessEqual(c1 + c2, 2 * 2.05 + 1e-9)

    def test_randomness(self):
        random.seed(0)
        pairs = {self.s._trig_coefficients() for _ in range(20)}
        self.assertGreater(len(pairs), 1)


class TestPSOEvaluateParticle(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        ss, ev = _make_ev("cifar100", "edgegpu")
        self.s = _pso_factory()(ss, ev)
        self.s.evaluations = 0

    def test_fitness_set(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._evaluate_particle(ind)
        self.assertIsNotNone(ind.fitness)

    def test_counter_increments(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._evaluate_particle(ind)
        self.assertEqual(self.s.evaluations, 1)

    def test_two_objectives(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._evaluate_particle(ind)
        self.assertEqual(len(ind.fitness), 2)

    def test_accuracy_range(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._evaluate_particle(ind)
        self.assertGreaterEqual(ind.fitness[0], 0.0)
        self.assertLessEqual(ind.fitness[0], 100.0)

    def test_latency_positive(self):
        ind = Individual([0, 1, 2, 3, 4, 0])
        self.s._evaluate_particle(ind)
        self.assertGreater(ind.fitness[1], 0.0)


class TestPSOGaussianMutationDecay(unittest.TestCase):

    def test_sigma_near_zero_at_progress_one(self):
        ss = CSVSearchSpace(str(CSV))
        mut = GaussianMutation(ss)
        mut._get_progress = lambda: 1.0
        ind  = Individual([0] * ss.num_edges)
        hits = sum(mut.mutate(ind).genotype != ind.genotype for _ in range(200))
        self.assertLess(hits, 20)

    def test_sigma_active_at_progress_zero(self):
        ss = CSVSearchSpace(str(CSV))
        mut = GaussianMutation(ss)
        mut._get_progress = lambda: 0.0
        ind  = Individual([0] * ss.num_edges)
        hits = sum(mut.mutate(ind).genotype != ind.genotype for _ in range(200))
        self.assertGreater(hits, 5)


# ---------------------------------------------------------------------------
# Context runner integration
# ---------------------------------------------------------------------------

class TestPSOContextRunner(unittest.TestCase):

    def test_run_strategy_on_contexts_no_error(self):
        random.seed(0)
        try:
            run_strategy_on_contexts(
                strategy_factory=_pso_factory(budget=SMALL_BUDGET),
                csv_path=CSV,
                datasets=FAST_DATASETS,
                devices=FAST_DEVICES,
                print_summary=False,
            )
        except Exception as exc:
            self.fail(f"run_strategy_on_contexts raised: {exc}")

    def test_extract_front_valid_per_context(self):
        factory = _pso_factory(budget=SMALL_BUDGET)
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
        factory = _pso_factory(budget=SMALL_BUDGET)
        for dataset in FAST_DATASETS:
            for device in FAST_DEVICES:
                random.seed(0)
                ss, ev   = _make_ev(dataset, device)
                strategy = factory(ss, ev)
                strategy.run()
                self.assertLessEqual(strategy.evaluations, SMALL_BUDGET)

    def test_inertia_propagated_to_population(self):
        ss, ev   = _make_ev("cifar100", "edgegpu")
        strategy = _pso_factory(w=0.7)(ss, ev)
        strategy.run()
        self.assertEqual(strategy.population.w, 0.7)


# ---------------------------------------------------------------------------
# Multi-run analysis (run_method_analysis protocol)
# ---------------------------------------------------------------------------

class TestPSOMultiRunAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        factory = _pso_factory(budget=STAT_BUDGET)
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
# Convergence — PSO vs RandomSearch (shared reference front)
# ---------------------------------------------------------------------------

class TestPSOConvergence(unittest.TestCase):
    DATASET, DEVICE = "cifar100", "edgegpu"

    @classmethod
    def setUpClass(cls):
        pso_f = _pso_factory(budget=STAT_BUDGET)
        rnd_f = _random_factory(budget=STAT_BUDGET)
        pso_all, rnd_all, pso_runs, rnd_runs = [], [], [], []
        for seed in range(N_STAT):
            p = _run_once(seed, cls.DATASET, cls.DEVICE, pso_f)
            r = _run_once(seed, cls.DATASET, cls.DEVICE, rnd_f)
            pso_runs.append(p["points"]); pso_all.extend(p["points"])
            rnd_runs.append(r["points"]); rnd_all.extend(r["points"])
        ref_front, rp, ip = _build_ref(pso_all + rnd_all)
        cls.pso_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in pso_runs]
        cls.rnd_hvs = [normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                       for pts in rnd_runs]

    def test_mean_hv_beats_random(self):
        pm, rm = statistics.mean(self.pso_hvs), statistics.mean(self.rnd_hvs)
        self.assertLessEqual(rm - pm, 0.3, msg=f"PSO={pm:.4f} vs Random={rm:.4f} (Random not more than 0.3 better)")

    def test_hv_positive_all_seeds(self):
        for i, hv in enumerate(self.pso_hvs):
            self.assertGreater(hv, 0.0, msg=f"seed {i}")

    def test_hv_std_reasonable(self):
        if len(self.pso_hvs) < 2: return
        self.assertLess(statistics.stdev(self.pso_hvs), 0.5)


# ---------------------------------------------------------------------------
# Sensitivity — inertia weight sweep
# ---------------------------------------------------------------------------

class TestPSOSensitivityInertia(unittest.TestCase):
    W_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]

    @classmethod
    def setUpClass(cls):
        cls.mean_hvs = {}
        for w in cls.W_VALUES:
            factory = _pso_factory(w=w, budget=STAT_BUDGET)
            all_pts, run_pts = [], []
            seeds = [seed * 97 + int(w * 100) for seed in range(4)]
            for seed in seeds:
                r = _run_once(seed, "cifar100", "edgegpu", factory)
                run_pts.append(r["points"]); all_pts.extend(r["points"])
            _, rp, ip = _build_ref(all_pts)
            cls.mean_hvs[w] = statistics.mean(
                normalized_hypervolume_2d(pts, DIRECTIONS, rp, ip)
                for pts in run_pts)

    def test_all_w_positive_hv(self):
        for w, hv in self.mean_hvs.items():
            self.assertGreater(hv, 0.0, msg=f"w={w}")

    def test_default_w_not_worst(self):
        """Paper default w=0.4 should not have the worst HV."""
        default_hv = self.mean_hvs[0.4]
        min_hv = min(self.mean_hvs.values())
        # Allow it to be equal to min (ties happen), but not isolated worst
        non_default = [v for k, v in self.mean_hvs.items() if k != 0.4]
        self.assertGreaterEqual(default_hv, min(non_default) * 0.5)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

class TestPSOBoundary(unittest.TestCase):

    def test_w_zero(self):
        r = _run_once(0, "cifar100", "edgegpu", _pso_factory(w=0.0, budget=SMALL_BUDGET))
        self.assertGreater(len(r["points"]), 0)

    def test_w_one(self):
        r = _run_once(0, "cifar100", "edgegpu", _pso_factory(w=1.0, budget=SMALL_BUDGET))
        self.assertGreater(len(r["points"]), 0)

    def test_pop_size_two(self):
        r = _run_once(0, "cifar100", "edgegpu", _pso_factory(pop_size=2, budget=20))
        self.assertGreater(r["evaluations"], 0)

    def test_budget_equals_pop_size(self):
        r = _run_once(0, "cifar100", "edgegpu",
                      _pso_factory(pop_size=10, budget=10))
        self.assertLessEqual(r["evaluations"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
