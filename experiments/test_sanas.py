"""
End-to-end test for SA-NAS on the merged CSV benchmark.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.evaluator import Evaluator
from nas_framework.search_space import CSVSearchSpace
from nas_framework.sa_strategy import SANAS


def _resolve_csv(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    for fb in [ROOT / "nas_benchmarks" / "datasets" / p.name,
               ROOT / "nas_benchmarks" / p.name]:
        if fb.exists():
            return fb
    raise FileNotFoundError(f"CSV not found: {raw}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SA-NAS on the NAS/HW CSV benchmark.")
    p.add_argument("--csv",           default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv")
    p.add_argument("--dataset",       default="cifar100")
    p.add_argument("--device",        default="edgegpu")
    p.add_argument("--budget",        type=int,   default=500)
    p.add_argument("--N",             type=int,   default=1,    help="Neighbours per iteration")
    p.add_argument("--T",             type=float, default=1e5,  help="Initial temperature")
    p.add_argument("--b",             type=float, default=1.0,  help="Boltzmann constant scale")
    p.add_argument("--c",             type=float, default=0.98, help="Cooling rate")
    p.add_argument("--gamma",         type=float, default=0.0,  help="Latency weight in loss")
    p.add_argument("--history-steps", type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    csv_path     = _resolve_csv(args.csv)
    search_space = CSVSearchSpace(str(csv_path))
    benchmark    = CSVBenchmarkAPI(str(csv_path))
    evaluator    = Evaluator(benchmark, dataset=args.dataset, device=args.device)

    strategy = SANAS(
        search_space=search_space,
        evaluator=evaluator,
        budget=args.budget,
        N=args.N,
        T=args.T,
        b=args.b,
        c=args.c,
        gamma=args.gamma,
    )

    pareto = strategy.run()

    print("RUN_OK")
    print(f"csv_path    : {csv_path}")
    print(f"dataset     : {args.dataset}")
    print(f"evaluations : {strategy.evaluations}")
    print(f"generations : {strategy.generations}")
    print(f"pareto_size : {len(pareto)}")

    if pareto:
        fits     = [i.fitness for i in pareto if i.fitness]
        best_acc = max(f[0] for f in fits)
        best_lat = min(f[1] for f in fits)
        top      = max(pareto, key=lambda i: i.fitness[0] if i.fitness else -1)
        print(f"best_acc    : {best_acc:.4f}")
        print(f"best_lat    : {best_lat:.4f}")
        print(f"top_geno    : {top.genotype}")
        print(f"top_fitness : {top.fitness}")
        print(f"top_arch_id : {top.metadata.get('arch_id')}")

    entries = strategy.history.entries
    archive = strategy.history.pareto_archive
    n    = max(1, min(args.history_steps, len(entries)))
    idxs = (
        [0] if n == 1
        else sorted({int(i * (len(entries) - 1) / (n - 1)) for i in range(n)})
    )
    print(f"history ({len(entries)} entries):")
    for i in idxs:
        e     = entries[i]
        front = archive[i] if i < len(archive) else []
        fits  = [ind.fitness for ind in front if ind.fitness]
        if fits:
            metric = (f"best_acc={max(f[0] for f in fits):.4f}, "
                      f"best_lat={min(f[1] for f in fits):.4f}")
        else:
            metric = "best_acc=nan, best_lat=nan"
        print(f"  step[{i}] gen={e.generation}, evals={e.evaluations}, "
              f"pareto={e.pareto_front_size}, {metric}")


if __name__ == "__main__":
    main()
