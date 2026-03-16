from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Population
from nas_framework.replacement import ElitistReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.selection import TournamentSelection
from nas_framework.variation import CrossoverMutationVariation
from nas_framework.pdns_descriptor import PDNSDescriptorExtractor
from nas_framework.pdns_strategy import MTF_PDNS


CSV = "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"

random.seed(42)

search_space = CSVSearchSpace(CSV)
benchmark    = CSVBenchmarkAPI(CSV)
evaluator    = Evaluator(benchmark, dataset="cifar100", device="edgegpu")
population   = Population(search_space, evaluator, size=20)

descriptor_extractor = PDNSDescriptorExtractor(
    benchmark=benchmark,
    dataset="cifar100",
    latency_devices=("edgegpu", "raspi4", "fpga"),
)

variation = CrossoverMutationVariation(UniformCrossover(), SinglePointMutation(search_space))

strategy = MTF_PDNS(
    population=population,
    selection=TournamentSelection(k=3),
    variation=variation,
    replacement=ElitistReplacement(),
    evaluator=evaluator,
    descriptor_extractor=descriptor_extractor,
    budget=100,
)

final_pop = strategy.run()
archive   = strategy.archive
inds      = archive.individuals()

print("RUN_OK")
print(f"evaluations:  {strategy.evaluations}")
print(f"generations:  {strategy.generations}")
print(f"archive_size: {len(archive)}")

if inds:
    fits = [i.fitness for i in inds if i.fitness]
    best_acc = max(f[0] for f in fits)
    best_lat = min(f[1] for f in fits)
    print(f"best_accuracy: {best_acc:.4f}")
    print(f"best_latency:  {best_lat:.4f}")

entries = strategy.history.entries
print(f"history_len:  {len(entries)}")
for e in entries[::max(1, len(entries)//5)]:
    print(f"  gen={e.generation} evals={e.evaluations} archive={e.pareto_front_size}")
