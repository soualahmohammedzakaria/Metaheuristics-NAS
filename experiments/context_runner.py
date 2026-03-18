from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

# Ensure imports work when running: python experiments/context_runner.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.evaluator import Evaluator
from nas_framework.population import Individual, Population
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import SkylineSearch

DEFAULT_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
DEFAULT_DEVICES = ("edgegpu", "edgetpu", "eyeriss", "fpga", "pixel3", "raspi4")

StrategyFactory = Callable[[CSVSearchSpace, Evaluator], Any]


def _resolve_csv_path(raw_csv: str) -> Path:
    candidate = Path(raw_csv)
    if candidate.exists():
        return candidate

    fallback_candidates = [
        ROOT / "nas_benchmarks" / "datasets" / candidate.name,
        ROOT / "nas_benchmarks" / candidate.name,
        ROOT / "datasets" / candidate.name,
    ]
    for path in fallback_candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "CSV file not found. Tried: "
        f"{candidate}, {fallback_candidates[0]}, {fallback_candidates[1]}"
    )


def _extract_front(strategy: Any, result: Any) -> list[Individual]:
    if isinstance(result, Population):
        return result.pareto_front()
    if isinstance(result, list) and (not result or isinstance(result[0], Individual)):
        return result

    history = getattr(strategy, "history", None)
    if history and getattr(history, "pareto_archive", None):
        archive = history.pareto_archive
        if archive:
            return archive[-1]

    raise TypeError(
        "Unable to get Pareto front from strategy result. "
        "Expected Population, list[Individual], or strategy.history.pareto_archive."
    )


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def format_row(cells: Iterable[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    output = [line, format_row(headers), line]
    output.extend(format_row(row) for row in rows)
    output.append(line)
    return "\n".join(output)


def _write_rows_csv(output_csv: Path, headers: list[str], rows: list[list[str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def run_strategy_on_contexts(
    strategy_factory: StrategyFactory,
    csv_path: Path,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    devices: tuple[str, ...] = DEFAULT_DEVICES,
    output_csv: Path | None = None,
    print_summary: bool = True,
) -> None:
    pareto_rows_by_context: dict[tuple[str, str], list[list[str]]] = {}
    pareto_rows_flat: list[list[str]] = []
    summary_rows: list[list[str]] = []
    context_id_by_key: dict[tuple[str, str], int] = {}

    context_counter = 0
    for dataset in datasets:
        for device in devices:
            context_counter += 1
            context_id_by_key[(dataset, device)] = context_counter

    for dataset in datasets:
        for device in devices:
            search_space = CSVSearchSpace(str(csv_path))
            benchmark = CSVBenchmarkAPI(str(csv_path))
            evaluator = Evaluator(benchmark, dataset=dataset, device=device)
            strategy = strategy_factory(search_space, evaluator)

            run_result = strategy.run()
            front = _extract_front(strategy, run_result)
            context_key = (dataset, device)
            pareto_rows_by_context.setdefault(context_key, [])

            if not front:
                summary_rows.append([dataset, device, "0", "nan", "nan"])
                continue

            fits = [ind.fitness for ind in front if ind.fitness is not None]
            best_acc = max(f[0] for f in fits)
            best_lat = min(f[1] for f in fits)
            summary_rows.append([
                dataset,
                device,
                str(len(front)),
                f"{best_acc:.6f}",
                f"{best_lat:.6f}",
            ])

            for ind in front:
                if ind.fitness is None:
                    continue
                arch_id = ind.metadata.get("arch_id") if ind.metadata else None
                row = [
                    str(context_id_by_key[context_key]),
                    str(arch_id),
                    device,
                    dataset,
                    f"{ind.fitness[0]:.6f}",
                    f"{ind.fitness[1]:.6f}",
                ]
                pareto_rows_by_context[context_key].append(row)
                pareto_rows_flat.append(row)

    any_rows = False
    for dataset in datasets:
        for device in devices:
            context_rows = pareto_rows_by_context.get((dataset, device), [])
            if not context_rows:
                continue

            any_rows = True
            context_rows.sort(key=lambda r: (-float(r[4]), float(r[5])))
            print(f"Pareto Front Rows for dataset={dataset}, device={device}")
            print(
                _format_table(
                    ["context_id", "archid", "device", "dataset", "accuracy", "latency"],
                    context_rows,
                )
            )
            print()

    if not any_rows:
        print("No Pareto rows found.")

    if print_summary:
        summary_rows.sort(key=lambda r: (r[0], r[1]))
        print("\nContext Summary")
        print(
            _format_table(
                ["dataset", "device", "pareto_size", "best_acc", "best_latency"],
                summary_rows,
            )
        )

    if output_csv is not None:
        pareto_rows_flat.sort(key=lambda r: (int(r[0]), -float(r[4]), float(r[5])))
        _write_rows_csv(
            output_csv,
            ["context_id", "archid", "device", "dataset", "accuracy", "latency"],
            pareto_rows_flat,
        )
        print(f"\nSaved Pareto rows CSV: {output_csv}")
        print(f"Total contexts configured: {len(datasets) * len(devices)}")


def skyline_factory() -> StrategyFactory:
    def _factory(search_space: CSVSearchSpace, evaluator: Evaluator) -> SkylineSearch:
        return SkylineSearch(
            search_space=search_space,
            evaluator=evaluator,
        )

    return _factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a search strategy over all dataset x device contexts and print Pareto tables."
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Path to CSV benchmark/search-space file.",
    )
    parser.add_argument(
        "--output-csv",
            default="experiments/results/context_runner_pareto.csv",
        help="Path to output CSV for Pareto rows.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable context summary table.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    csv_path = _resolve_csv_path(args.csv)
    output_csv = Path(args.output_csv)
    factory = skyline_factory()

    run_strategy_on_contexts(
        strategy_factory=factory,
        csv_path=csv_path,
        output_csv=output_csv,
        print_summary=not args.no_summary,
    )


if __name__ == "__main__":
    main()
