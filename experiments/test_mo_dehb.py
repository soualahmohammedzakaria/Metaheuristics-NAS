"""
End-to-end test for MO-DEHB (both NSGA-II and EpsNet variants)
on the merged CSV benchmark.

Usage
-----
    python experiments/test_mo_dehb.py [options] --variant   nsga2 | epsnet | both   (default: both)

    EX: python experiments/test_mo_dehb.py --csv nas_benchmarks/datasets/nas_hw_search_space_bench.csv --dataset cifar100 --device edgegpu --pop-size 20 --budget 200 --seed 42 --variant both
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
from nas_framework.dehb_strategy import MODEHBNsgaII, MODEHBEpsNet, MODEHBBase


def _resolve_csv(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    for fb in [
        ROOT / "nas_benchmarks" / "datasets" / p.name,
        ROOT / "nas_benchmarks" / p.name,
    ]:
        if fb.exists():
            return fb
    raise FileNotFoundError(f"CSV not found: {raw}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MO-DEHB on the NAS/HW merged CSV benchmark."
    )
    p.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
    )
    p.add_argument("--dataset",      default="cifar100")
    p.add_argument("--device",       default="edgegpu")
    p.add_argument("--pop-size",     type=int,   default=20)
    p.add_argument("--budget",       type=int,   default=200)
    p.add_argument("--eta",          type=int,   default=3)
    p.add_argument("--min-fidelity", type=float, default=0.1)
    p.add_argument("--max-fidelity", type=float, default=1.0)
    p.add_argument("--F",            type=float, default=0.5,
                   help="DE mutation scale factor")
    p.add_argument("--CR",           type=float, default=0.5,
                   help="DE crossover rate")
    p.add_argument("--variant",      default="both",
                   choices=["nsga2", "epsnet", "both"])
    p.add_argument("--history-steps", type=int,  default=5)
    p.add_argument("--seed",          type=int,  default=None)
    return p


def run_variant(
    cls: type[MODEHBBase],
    name: str,
    csv_path: Path,
    dataset: str,
    device: str,
    pop_size: int,
    budget: int,
    eta: int,
    min_fidelity: float,
    max_fidelity: float,
    F: float,
    CR: float,
    history_steps: int,
) -> None:
    search_space = CSVSearchSpace(str(csv_path))
    benchmark    = CSVBenchmarkAPI(str(csv_path))
    evaluator    = Evaluator(benchmark, dataset=dataset, device=device)

    strategy = cls(
        search_space=search_space,
        evaluator=evaluator,
        pop_size=pop_size,
        budget=budget,
        eta=eta,
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        F=F,
        CR=CR,
    )

    pareto_front = strategy.run()

    print(f"\n{'='*55}")
    print(f"  MO-DEHB variant : {name}")
    print(f"{'='*55}")
    print(f"  evaluations     : {strategy.evaluations}")
    print(f"  generations     : {strategy.generations}")
    print(f"  pareto_size     : {len(pareto_front)}")

    if pareto_front:
        fits     = [ind.fitness for ind in pareto_front if ind.fitness]
        best_acc = max(f[0] for f in fits)
        best_lat = min(f[1] for f in fits)
        top_ind  = max(pareto_front, key=lambda i: i.fitness[0] if i.fitness else -1)
        print(f"  best_accuracy   : {best_acc:.4f}")
        print(f"  best_latency    : {best_lat:.4f}")
        print(f"  top_acc_geno    : {top_ind.genotype}")
        print(f"  top_acc_fitness : {top_ind.fitness}")
        print(f"  top_acc_arch_id : {top_ind.metadata.get('arch_id')}")

    entries = strategy.history.entries
    archive = strategy.history.pareto_archive
    if entries:
        n = max(1, min(history_steps, len(entries)))
        idxs = (
            [0] if n == 1
            else sorted({int(i * (len(entries) - 1) / (n - 1)) for i in range(n)})
        )
        print(f"  history ({len(entries)} entries):")
        for i in idxs:
            e     = entries[i]
            front = archive[i] if i < len(archive) else []
            fits  = [ind.fitness for ind in front if ind.fitness]
            if fits:
                ba = max(f[0] for f in fits)
                bl = min(f[1] for f in fits)
                metric = f"best_acc={ba:.4f}, best_lat={bl:.4f}"
            else:
                metric = "best_acc=nan, best_lat=nan"
            print(
                f"    step[{i}] gen={e.generation}, evals={e.evaluations}, "
                f"pareto={e.pareto_front_size}, {metric}"
            )


def main() -> None:
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    csv_path = _resolve_csv(args.csv)

    variants: list[tuple[type[MODEHBBase], str]] = []
    if args.variant in ("nsga2", "both"):
        variants.append((MODEHBNsgaII, "NSGA-II"))
    if args.variant in ("epsnet", "both"):
        variants.append((MODEHBEpsNet, "EpsNet"))

    print(f"RUN_OK  csv={csv_path}  dataset={args.dataset}  budget={args.budget}")

    for cls, name in variants:
        if args.seed is not None:
            random.seed(args.seed)  
        run_variant(
            cls=cls, name=name,
            csv_path=csv_path,
            dataset=args.dataset,
            device=args.device,
            pop_size=args.pop_size,
            budget=args.budget,
            eta=args.eta,
            min_fidelity=args.min_fidelity,
            max_fidelity=args.max_fidelity,
            F=args.F,
            CR=args.CR,
            history_steps=args.history_steps,
        )


if __name__ == "__main__":
    main()
