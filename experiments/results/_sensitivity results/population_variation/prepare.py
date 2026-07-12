from pathlib import Path
import pandas as pd

# ==========================================
# CHANGE THIS
# ==========================================
ROOT = Path(".")        # folder containing budget_xxx_pop_xxx folders
OUTPUT = "population_sensitivity_summary.csv"
# ==========================================

rows = []

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir():
        continue

    csv_path = folder / "mosho_context_metrics.csv"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    # --------------------------------------
    # extract budget and population
    # --------------------------------------
    parts = folder.name.split("_")

    budget = int(parts[1])
    population = int(parts[3])

    row = {
        "budget": budget,
        "population": population,
    }

    # --------------------------------------
    # metrics to summarize
    # --------------------------------------

    metrics = [
        "hv_median",
        "igd_plus_median",
        "spacing_median",
        "runtime_median",
        "best_accuracy_mean",
        "best_latency_mean",
    ]

    for metric in metrics:
        row[metric] = df[metric].median()

    rows.append(row)

summary = pd.DataFrame(rows)

summary = summary.sort_values("population")

summary.to_csv(OUTPUT, index=False)

print(summary)
print(f"\nSaved to {OUTPUT}")