"""
tests/conftest.py
=================
Shared pytest fixtures used across all test modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "nas_benchmarks" / "datasets" / "nas_hw_search_space_bench.csv"

# ── Lightweight shared objects (session-scoped → built once per test run) ──

@pytest.fixture(scope="session")
def csv_path() -> Path:
    if not CSV_PATH.exists():
        pytest.skip("Benchmark CSV not found – skipping integration tests.")
    return CSV_PATH


@pytest.fixture(scope="session")
def search_space(csv_path):
    from nas_framework.search_space import CSVSearchSpace
    return CSVSearchSpace(str(csv_path))


@pytest.fixture(scope="session")
def benchmark(csv_path):
    from nas_framework.benchmark_api import CSVBenchmarkAPI
    return CSVBenchmarkAPI(str(csv_path))


@pytest.fixture(scope="session")
def evaluator(benchmark):
    from nas_framework.evaluator import Evaluator
    return Evaluator(benchmark, dataset="cifar100", device="edgegpu")


@pytest.fixture
def small_population(search_space, evaluator):
    """Fresh population of size 10 for fast tests."""
    from nas_framework.population import Population
    return Population(search_space, evaluator, size=10)


@pytest.fixture
def tiny_population(search_space, evaluator):
    """Population of size 4 – minimum viable for DE (needs at least 4)."""
    from nas_framework.population import Population
    return Population(search_space, evaluator, size=4)
