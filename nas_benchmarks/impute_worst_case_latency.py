import argparse
import csv
from pathlib import Path


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[1]
    alt = root / path_like
    if alt.exists():
        return alt
    return p


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    v = value.strip().lower()
    return v in {"", "nan", "none", "null", "na", "n/a"}


def _parse_float(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        x = float(value)
        if x != x:  # NaN check without math.isnan
            return None
        return x
    except ValueError:
        return None


def _latency_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.endswith("_latency")]


def impute_worst_case_latency(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        rows = list(reader)
        fieldnames = reader.fieldnames

    latency_cols = _latency_columns(fieldnames)
    if not latency_cols:
        raise ValueError("No latency columns found (expected columns ending with '_latency')")

    worst_by_col: dict[str, float] = {}
    for col in latency_cols:
        observed = []
        for row in rows:
            x = _parse_float(row.get(col))
            if x is not None:
                observed.append(x)
        if not observed:
            raise ValueError(f"Column has no valid numeric values to infer worst-case latency: {col}")
        worst_by_col[col] = max(observed)

    replacements_by_col: dict[str, int] = {col: 0 for col in latency_cols}
    for row in rows:
        for col in latency_cols:
            if _parse_float(row.get(col)) is None:
                row[col] = f"{worst_by_col[col]:.10g}"
                replacements_by_col[col] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_replacements = sum(replacements_by_col.values())
    print("Imputation completed successfully.")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Latency columns: {len(latency_cols)}")
    print(f"Total replacements: {total_replacements}")
    for col in latency_cols:
        print(
            f"  {col}: replaced={replacements_by_col[col]}, worst_case={worst_by_col[col]:.10g}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace missing/NaN latency values with worst-case latency (column max) "
            "for conservative Pareto analysis."
        )
    )
    parser.add_argument(
        "--input",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input merged benchmark CSV",
    )
    parser.add_argument(
        "--output",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench_worst_latency.csv",
        help="Output CSV with imputed worst-case latency values",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input CSV directly (output argument will be ignored)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_csv = _resolve_path(args.input)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = input_csv if args.in_place else _resolve_path(args.output)
    impute_worst_case_latency(input_csv=input_csv, output_csv=output_csv)
