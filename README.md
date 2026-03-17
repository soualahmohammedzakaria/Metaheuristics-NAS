# Dvolver on ORAMA NAS Framework

This repository now contains a Dvolver-style multi-objective NAS implementation integrated into the existing modular NAS framework.

## What Is Implemented

### 1 Dvolver Search Space
Implemented in [nas_framework/search_space.py](nas_framework/search_space.py).

- `NASSearchSpace`
  - NASNet-like two-cell architecture representation (`normal_cell`, `reduction_cell`)
  - 5 blocks per cell, each block as `(I1, O1, I2, O2)`
  - Extra skip connections per cell as binary-mask-backed source list
  - Full `encode()` / `decode()` pipeline for evolutionary operators
  - `random_value_for_gene()` for bounded per-gene mutation
  - `num_possible_architectures()` returns an approximate `1e20`

- `CSVGenotypeDvolverSearchSpace`
  - Dvolver-compatible search over NAS-Bench-201 genotype vectors loaded from CSV
  - Produces architecture dicts with `benchmark_genotype`, `arch_id`, and `architecture_details`
  - Lets Dvolver query benchmark metrics directly

### 2 Population and Individual Model
Implemented in [nas_framework/population.py](nas_framework/population.py).

- `Individual` now supports both legacy and Dvolver styles:
  - legacy fields: `genotype`, `fitness`
  - Dvolver aliases: `architecture`, `objectives`
- Added NSGA-II bookkeeping fields:
  - `rank`, `crowding_distance`, `dominated_by`, `dominates`
- `Population` remains backward-compatible and can initialize from architecture lists

### 3 Multi-Objective Core (NSGA-II / Dvolver)
Implemented in [nas_framework/mo_utils.py](nas_framework/mo_utils.py).

- Pareto dominance
- Fast non-dominated sorting
- Crowding distance assignment
- Crowded comparison operator
- 2D hypervolume computation (maximize-maximize)
- Pareto extraction utilities

### 4 Evolutionary Operators
Implemented/extended in:
- [nas_framework/crossover.py](nas_framework/crossover.py)
- [nas_framework/mutation.py](nas_framework/mutation.py)
- [nas_framework/variation.py](nas_framework/variation.py)
- [nas_framework/selection.py](nas_framework/selection.py)
- [nas_framework/replacement.py](nas_framework/replacement.py)

Added Dvolver-specific classes:
- `DvolverUniformCrossover` (uniform gene swap)
- `UniformMutation` (per-gene random valid-value mutation)
- `DvolverVariation` (pairing + crossover + mutation -> N offspring)
- `BinaryTournamentSelection` (crowded comparison)
- `DvolverReplacement` (Algorithm-1 style select N from 2N)

### 5 Evaluation Logic
Implemented/extended in [nas_framework/evaluator.py](nas_framework/evaluator.py).

- `DvolverEvaluator` returns two maximized objectives:
  - objective 1: accuracy
  - objective 2: speed = `2e9 / FLOPs`
- Two modes are supported:
  - proxy mode (no benchmark): analytic FLOPs + proxy accuracy
  - benchmark mode: query `CSVBenchmarkAPI` from benchmark CSV via `benchmark_genotype`

### 6 History, Archive, and Termination
Implemented/extended in:
- [nas_framework/history.py](nas_framework/history.py)
- [nas_framework/termination.py](nas_framework/termination.py)

- `History` now tracks:
  - all evaluated individuals
  - per-generation Pareto fronts
  - hypervolume history
  - convergence check by hypervolume patience
- `TerminationCriteria` supports stopping by:
  - max generations
  - max evaluations
  - no hypervolume improvement

### 7 Main Dvolver Strategy
Implemented in [nas_framework/search_strategy.py](nas_framework/search_strategy.py).

- Added `DvolverSearchStrategy`:
  1. initialize population of size `N`
  2. evaluate
  3. rank + crowding
  4. iterate selection -> variation -> evaluate offspring -> replacement
  5. update history and check termination

### 8 Public Exports
Updated [nas_framework/__init__.py](nas_framework/__init__.py) to export Dvolver classes while preserving old APIs.

### 9 Experiment Runner and Outputs
Implemented in [experiments/test.py](experiments/test.py).

- Generates:
  - Pareto scatter plot
  - hypervolume convergence plot
  - JSON/CSV of all evaluated architectures
  - top-3 knee-point candidates (`Dvolver-A/B/C`)
- Supports benchmark mode with:
  - `--benchmark-csv`
  - `--dataset`
  - `--device`

## How To Run

### A Proxy mode (no benchmark query)

```bash
python experiments/test.py --max-generations 20 --seed 42 --output-dir results/dvolver
```

### B Benchmark mode (queries NAS/HW CSV)

```bash
python experiments/test.py --max-generations 5 --seed 42 --benchmark-csv nas_benchmarks/datasets/nas_hw_search_space_bench.csv --dataset cifar100 --device edgegpu --output-dir results/dvolver_hw
```

## Results Explanation

## Proxy run (results/dvolver)
Observed from run:
- generations: `20`
- evaluations: `672`
- Pareto size: `61`
- last hypervolume: `10.906286`

Observed objective ranges from [results/dvolver/all_evaluated_architectures.json](results/dvolver/all_evaluated_architectures.json):
- accuracy: `0.773` to `1.000`
- speed: `1.885` to `11.439`

Top-3 knee candidates from [results/dvolver/top3_architectures.json](results/dvolver/top3_architectures.json):
- Dvolver-A: `(0.918, 8.5685)`
- Dvolver-B: `(0.931, 8.1385)`
- Dvolver-C: `(0.931, 8.1385)`

Interpretation:
- The Pareto front shows the expected trade-off between accuracy and speed.
- Hypervolume increased through generations, indicating progressive search improvement.
- B and C are identical in this run due to knee-point ranking ties.

## Benchmark run (results/dvolver_hw)
Observed from run:
- generations: `5`
- evaluations: `192`
- Pareto size: `49`
- last hypervolume: `9110.211828`
- benchmark mode: `True`

Observed objective ranges from [results/dvolver_hw/all_evaluated_architectures.json](results/dvolver_hw/all_evaluated_architectures.json):
- accuracy: `46.55` to `72.42`
- speed: `26.0207` to `136.8626`

Top-3 from [results/dvolver_hw/top3_architectures.json](results/dvolver_hw/top3_architectures.json):
- Dvolver-A:
  - `arch_id`: `13413`
  - genotype: `[1, 2, 0, 1, 3, 0]`
  - objectives: `(67.30666, 80.82258)`
- Dvolver-B:
  - `arch_id`: `13320`
  - genotype: `[0, 4, 2, 2, 0, 0]`
  - objectives: `(63.04, 106.33674)`
- Dvolver-C: same as B in this run

Interpretation:
- Dvolver is now querying the benchmark-backed objectives in this mode.
- Accuracy and speed are both optimized in maximize-maximize form.
- Duplicate B/C indicates the current knee-point postprocessing does not enforce diversity.

## Output Files

Proxy outputs:
- [results/dvolver/pareto_front.png](results/dvolver/pareto_front.png)
- [results/dvolver/hypervolume_curve.png](results/dvolver/hypervolume_curve.png)
- [results/dvolver/all_evaluated_architectures.json](results/dvolver/all_evaluated_architectures.json)
- [results/dvolver/all_evaluated_architectures.csv](results/dvolver/all_evaluated_architectures.csv)
- [results/dvolver/top3_architectures.json](results/dvolver/top3_architectures.json)

Benchmark outputs:
- [results/dvolver_hw/pareto_front.png](results/dvolver_hw/pareto_front.png)
- [results/dvolver_hw/hypervolume_curve.png](results/dvolver_hw/hypervolume_curve.png)
- [results/dvolver_hw/all_evaluated_architectures.json](results/dvolver_hw/all_evaluated_architectures.json)
- [results/dvolver_hw/all_evaluated_architectures.csv](results/dvolver_hw/all_evaluated_architectures.csv)
- [results/dvolver_hw/top3_architectures.json](results/dvolver_hw/top3_architectures.json)

## Notes

- Legacy framework components (RandomSearch/GA/ES) are preserved.
- Dvolver integration was added in a backward-compatible way.
- For strict Dvolver-paper cell encoding experiments, use `NASSearchSpace` (proxy mode). For direct NAS/HW benchmark querying in this codebase, use CSV benchmark mode.
