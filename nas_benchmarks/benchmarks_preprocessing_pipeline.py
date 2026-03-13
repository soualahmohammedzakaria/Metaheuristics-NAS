import argparse
from pathlib import Path

from export_hw_nas_bench import export_hw_csv
from export_nas_bench_201 import export_csv
from merge_and_preprocess_benchmarks import merge_and_preprocess


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[1]
    alt = root / path_like
    if alt.exists():
        return alt
    return p


def run_pipeline(
    nas_path: Path,
    hw_path: Path,
    nas_output: Path,
    hw_output: Path,
    merged_output: Path,
    max_arch: int,
) -> None:
    print("[1/3] Export NAS-Bench-201 metrics...")
    export_csv(nas_path=nas_path, output_path=nas_output)

    print("[2/3] Export HW-NAS-Bench metrics...")
    export_hw_csv(hw_path=hw_path, output_path=hw_output, max_arch=max_arch)

    print("[3/3] Merge and preprocess (numeric conversion, latency -1 -> NaN, op/gene columns)...")
    merge_and_preprocess(nas_csv=nas_output, hw_csv=hw_output, output_csv=merged_output)

    print("Pipeline completed successfully.")
    print(f"NAS export: {nas_output}")
    print(f"HW export: {hw_output}")
    print(f"Merged preprocessed: {merged_output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full NAS/HW benchmark preprocessing pipeline: "
            "export_nas_bench_201 + export_hw_nas_bench + merge_and_preprocess_benchmarks"
        )
    )
    parser.add_argument(
        "--nas",
        default="nas_benchmarks/datasets/NAS-Bench-201-v1_0-e61699.pth",
        help="NAS-Bench-201 .pth path",
    )
    parser.add_argument(
        "--hw",
        default="nas_benchmarks/datasets/HW-NAS-Bench-v1_0.pickle",
        help="HW-NAS-Bench .pickle path",
    )
    parser.add_argument("--nas-output", default="results/nas_bench_201_export.csv", help="NAS export csv")
    parser.add_argument("--hw-output", default="results/hw_nas_bench_export.csv", help="HW export csv")
    parser.add_argument(
        "--merged-output",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Merged output csv",
    )
    parser.add_argument("--max-arch", type=int, default=15625, help="Upper bound while probing HW arch ids")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    nas_path = _resolve_path(args.nas)
    hw_path = _resolve_path(args.hw)
    if not nas_path.exists():
        raise FileNotFoundError(f"NAS-Bench-201 file not found: {nas_path}")
    if not hw_path.exists():
        raise FileNotFoundError(f"HW-NAS-Bench file not found: {hw_path}")

    run_pipeline(
        nas_path=nas_path,
        hw_path=hw_path,
        nas_output=_resolve_path(args.nas_output),
        hw_output=_resolve_path(args.hw_output),
        merged_output=_resolve_path(args.merged_output),
        max_arch=args.max_arch,
    )