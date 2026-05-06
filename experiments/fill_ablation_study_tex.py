from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class RunAggregate:
    hv: float
    igd_plus: float
    spacing: float
    runtime_sec: float


@dataclass(frozen=True)
class Summary:
    hv_median: float
    hv_iqr: float
    igd_median: float
    igd_iqr: float
    spacing_median: float
    spacing_iqr: float
    runtime_median: float
    runtime_iqr: float


def _tukey_iqr(values: list[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if n < 4:
        return 0.0
    half = n // 2
    lower = xs[:half]
    upper = xs[-half:]
    return float(median(upper) - median(lower))


def _load_metrics_by_run(csv_path: Path) -> dict[int, list[RunAggregate]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing per-run metrics CSV: {csv_path}")

    by_run: dict[int, list[RunAggregate]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"run_id", "hv", "igd_plus", "spacing", "runtime_sec"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} missing columns {sorted(missing)}; got {reader.fieldnames}"
            )

        for row in reader:
            try:
                run_id = int(row["run_id"])
                hv = float(row["hv"])
                igd = float(row["igd_plus"])
                sp = float(row["spacing"])
                rt = float(row["runtime_sec"])
            except (ValueError, KeyError):
                continue

            by_run.setdefault(run_id, []).append(
                RunAggregate(hv=hv, igd_plus=igd, spacing=sp, runtime_sec=rt)
            )

    if not by_run:
        raise ValueError(f"No rows parsed from {csv_path}")
    return by_run


def _aggregate_across_contexts(by_run: dict[int, list[RunAggregate]]) -> dict[int, RunAggregate]:
    aggregated: dict[int, RunAggregate] = {}
    for run_id, rows in by_run.items():
        if not rows:
            continue
        aggregated[run_id] = RunAggregate(
            hv=float(mean(r.hv for r in rows)),
            igd_plus=float(mean(r.igd_plus for r in rows)),
            spacing=float(mean(r.spacing for r in rows)),
            runtime_sec=float(mean(r.runtime_sec for r in rows)),
        )

    if not aggregated:
        raise ValueError("No aggregated run metrics computed")
    return aggregated


def _summary(aggregated: dict[int, RunAggregate]) -> Summary:
    run_ids = sorted(aggregated.keys())
    hv = [aggregated[r].hv for r in run_ids]
    igd = [aggregated[r].igd_plus for r in run_ids]
    sp = [aggregated[r].spacing for r in run_ids]
    rt = [aggregated[r].runtime_sec for r in run_ids]

    return Summary(
        hv_median=float(median(hv)),
        hv_iqr=_tukey_iqr(hv),
        igd_median=float(median(igd)),
        igd_iqr=_tukey_iqr(igd),
        spacing_median=float(median(sp)),
        spacing_iqr=_tukey_iqr(sp),
        runtime_median=float(median(rt)),
        runtime_iqr=_tukey_iqr(rt),
    )


def _fmt(x: float, ndigits: int) -> str:
    return f"{x:.{ndigits}f}"


def _fmt_signed(x: float, ndigits: int) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.{ndigits}f}"


def _replace_row_in_table(
    tex: str,
    *,
    table_label: str,
    row_startswith: str,
    new_line: str,
) -> str:
    label_token = f"\\label{{{table_label}}}"
    label_idx = tex.find(label_token)
    if label_idx < 0:
        raise ValueError(f"Could not find table label {label_token!r}")

    end_idx = tex.find("\\end{table}", label_idx)
    if end_idx < 0:
        raise ValueError(f"Could not find \\end{{table}} after label {table_label}")

    before = tex[:label_idx]
    table_block = tex[label_idx:end_idx]
    after = tex[end_idx:]

    lines = table_block.splitlines(keepends=False)
    out_lines: list[str] = []
    replaced = 0
    for line in lines:
        if line.lstrip().startswith(row_startswith):
            out_lines.append(new_line)
            replaced += 1
        else:
            out_lines.append(line)

    if replaced != 1:
        raise ValueError(
            f"Expected exactly 1 row starting with {row_startswith!r} in table {table_label}, found {replaced}"
        )

    new_table_block = "\n".join(out_lines)
    return before + new_table_block + after


def _replace_first(tex: str, *, old: str, new: str) -> str:
    """Replace the first occurrence of `old` with `new`.

    This is intentionally *optional* (no-op if `old` is not present) so the
    script can be re-run after it has already patched the file once.
    """

    if old not in tex:
        return tex
    return tex.replace(old, new, 1)


def _fill_tex(tex_path: Path, full: Summary, base: Summary, mowso: Summary) -> None:
    tex = tex_path.read_text(encoding="utf-8")

    # Title page note (keep it honest).
    tex = _replace_first(
        tex,
        old="{\\small\\color{gray} Draft --- values are randomly simulated placeholders}",
        new="{\\small\\color{gray} Draft --- baseline rows filled from experiments; remaining values are placeholders pending ablation runs}",
    )

    # Results intro note.
    tex = _replace_first(
        tex,
        old="Values are \\emph{simulated placeholders} for structural review.",
        new="Baseline rows are filled from experiments; remaining values are placeholders pending ablation runs.",
    )

    # --- Table tab:res_oat (median performance) ---
    tex = _replace_row_in_table(
        tex,
        table_label="tab:res_oat",
        row_startswith="\\mosho{} (full)",
        new_line=(
            f"\\mosho{{}} (full) & {_fmt(full.igd_median, 4)} & {_fmt(full.hv_median, 4)} & "
            f"{_fmt(full.spacing_median, 4)} & {_fmt(full.runtime_median, 2)} \\\\"  # noqa: W605
        ),
    )
    tex = _replace_row_in_table(
        tex,
        table_label="tab:res_oat",
        row_startswith="MOSHO-Base",
        new_line=(
            f"MOSHO-Base      & {_fmt(base.igd_median, 4)} & {_fmt(base.hv_median, 4)} & "
            f"{_fmt(base.spacing_median, 4)} & {_fmt(base.runtime_median, 2)} \\\\"  # noqa: W605
        ),
    )
    tex = _replace_row_in_table(
        tex,
        table_label="tab:res_oat",
        row_startswith="\\mowso{}",
        new_line=(
            f"\\mowso{{}}        & {_fmt(mowso.igd_median, 4)} & {_fmt(mowso.hv_median, 4)} & "
            f"{_fmt(mowso.spacing_median, 4)} & {_fmt(mowso.runtime_median, 2)} \\\\"  # noqa: W605
        ),
    )

    # --- Table tab:iqr (robustness) ---
    tex = _replace_row_in_table(
        tex,
        table_label="tab:iqr",
        row_startswith="\\mosho{} (full) &",
        new_line=(
            f"\\mosho{{}} (full) & {_fmt(full.igd_median, 4)} & {_fmt(full.igd_iqr, 4)} & "
            f"{_fmt(full.hv_median, 4)} & {_fmt(full.hv_iqr, 4)} \\\\"  # noqa: W605
        ),
    )
    tex = _replace_row_in_table(
        tex,
        table_label="tab:iqr",
        row_startswith="MOSHO-Base      &",
        new_line=(
            f"MOSHO-Base      & {_fmt(base.igd_median, 4)} & {_fmt(base.igd_iqr, 4)} & "
            f"{_fmt(base.hv_median, 4)} & {_fmt(base.hv_iqr, 4)} \\\\"  # noqa: W605
        ),
    )
    tex = _replace_row_in_table(
        tex,
        table_label="tab:iqr",
        row_startswith="\\mowso{}        &",
        new_line=(
            f"\\mowso{{}}        & {_fmt(mowso.igd_median, 4)} & {_fmt(mowso.igd_iqr, 4)} & "
            f"{_fmt(mowso.hv_median, 4)} & {_fmt(mowso.hv_iqr, 4)} \\\\"  # noqa: W605
        ),
    )

    # --- Table tab:delta (absolute deltas) ---
    # Conventions used in the caption:
    #  - For IGD+, SP, Time (lower is better): Delta = variant - full (positive => variant worse)
    #  - For nHV (higher is better):          Delta = full - variant (positive => full better)
    base_delta_igd = base.igd_median - full.igd_median
    base_delta_hv = full.hv_median - base.hv_median
    base_delta_sp = base.spacing_median - full.spacing_median
    base_delta_t = base.runtime_median - full.runtime_median

    mowso_delta_igd = mowso.igd_median - full.igd_median
    mowso_delta_hv = full.hv_median - mowso.hv_median
    mowso_delta_sp = mowso.spacing_median - full.spacing_median
    mowso_delta_t = mowso.runtime_median - full.runtime_median

    tex = _replace_row_in_table(
        tex,
        table_label="tab:delta",
        row_startswith="MOSHO-Base      &",
        new_line=(
            f"MOSHO-Base      & {_fmt_signed(base_delta_igd, 4)} & {_fmt_signed(base_delta_hv, 4)} & "
            f"{_fmt_signed(base_delta_sp, 4)} & {_fmt_signed(base_delta_t, 2)} \\\\"  # noqa: W605
        ),
    )
    tex = _replace_row_in_table(
        tex,
        table_label="tab:delta",
        row_startswith="\\mowso{}",
        new_line=(
            f"\\mowso{{}}        & {_fmt_signed(mowso_delta_igd, 4)} & {_fmt_signed(mowso_delta_hv, 4)} & "
            f"{_fmt_signed(mowso_delta_sp, 4)} & {_fmt_signed(mowso_delta_t, 2)} \\\\"  # noqa: W605
        ),
    )

    tex_path.write_text(tex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fill baseline rows in ablation_study.tex from experiment CSVs. "
            "Uses MOSHO full = mosho_enhanced, MOSHO-Base = mosho, baseline = mowso."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results_ablation_B10000_N50_allContexts"),
        help="Root folder containing per-method result subfolders.",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("ablation_study.tex"),
        help="Path to the LaTeX file to update (in-place).",
    )
    args = parser.parse_args()

    results_root: Path = args.results_root
    tex_path: Path = args.tex

    full_csv = results_root / "mosho_enhanced" / "mosho_enhanced_metrics_by_run.csv"
    base_csv = results_root / "mosho" / "mosho_metrics_by_run.csv"
    mowso_csv = results_root / "mowso" / "mowso_metrics_by_run.csv"

    full = _summary(_aggregate_across_contexts(_load_metrics_by_run(full_csv)))
    base = _summary(_aggregate_across_contexts(_load_metrics_by_run(base_csv)))
    mowso = _summary(_aggregate_across_contexts(_load_metrics_by_run(mowso_csv)))

    _fill_tex(tex_path, full=full, base=base, mowso=mowso)

    print("Updated:", tex_path)
    print("Full (mosho_enhanced):", full)
    print("Base (mosho):", base)
    print("Baseline (mowso):", mowso)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
