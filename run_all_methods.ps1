# Run all methods + MOSHO ablation suite with B=10000, N=50, R=30, seed schedule.
$resultsRoot = "results_final"

$commonArgs = @(
  "--budget", "10000",
  "--pop-size", "50",
  "--runs", "10",
  "--seed", "42",
  "--seed-step", "7",
  "--results-root", $resultsRoot
)

$methods = @(
  "mowso",
  "mosho",
  "mosho_enhanced",
  "pso",
  "abc",
  "firefly",
  "nsga2"
)

foreach ($m in $methods) {
  Write-Host "=== Running method: $m ==="
  $args = @("experiments/run_method_analysis.py", "--method", $m) + $commonArgs
  python @args
  if ($LASTEXITCODE -ne 0) {
    Write-Error "Method $m failed with exit code $LASTEXITCODE. Aborting."
    exit $LASTEXITCODE
  }
}

Write-Host "=== Running MOSHO ablation suite ==="
$args = @("experiments/run_method_analysis.py", "--suite", "mosho_ablation") + $commonArgs
python @args
if ($LASTEXITCODE -ne 0) {
  Write-Error "Ablation suite failed with exit code $LASTEXITCODE."
  exit $LASTEXITCODE
}

Write-Host "All runs completed. Results are in $resultsRoot"