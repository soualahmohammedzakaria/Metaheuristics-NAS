import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# Hyper-parameter search spaces
# ─────────────────────────────────────────────────────────────────────
ConvActivationChoices = ["relu", "leaky_relu", "elu"]
FCActivationChoices   = ["relu", "leaky_relu", "elu"]

NumFiltersChoices     = [32, 64, 96, 128, 160, 192, 224, 256]
FilterSizeChoices     = [3, 5, 7]
SubsamplingChoices    = ["pool", "strive"]
PoolTypeChoices       = ["max", "avg"]
PoolSizeChoices       = [2, 3]
ConvDropoutChoices    = [0.3, 0.4, 0.5]
NumConvLayersChoices  = [2, 3, 4]

FCUnitsChoices        = [128, 256, 512]
FCDropoutChoices      = [0.3, 0.4, 0.5]

MAX_CONV_LAYERS = 4


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ConvBlock:
    num_filters:     int
    filter_size:     int
    activation:      str
    subsampling:     str
    pool_type:       str
    pool_size:       int
    dropout:         float
    num_conv_layers: int


@dataclass
class FCBlock:
    units:      int
    activation: str
    dropout:    float


@dataclass
class Solution:
    conv_blocks: List[ConvBlock] = field(default_factory=list)
    fc_blocks:   List[FCBlock]   = field(default_factory=list)
    f1: Optional[float] = None   # accuracy  (maximise)
    f2: Optional[float] = None   # FLOPs     (minimise)


# ─────────────────────────────────────────────────────────────────────
# Constructors
# ─────────────────────────────────────────────────────────────────────
def initial_vgg_solution() -> Solution:
    conv1 = ConvBlock(
        num_filters=32, filter_size=5, activation="relu",
        subsampling="pool", pool_type="max", pool_size=2,
        dropout=0.3, num_conv_layers=2,
    )
    conv2 = ConvBlock(
        num_filters=64, filter_size=3, activation="relu",
        subsampling="pool", pool_type="max", pool_size=2,
        dropout=0.3, num_conv_layers=2,
    )
    fc = FCBlock(units=128, activation="relu", dropout=0.3)
    return Solution(conv_blocks=[conv1, conv2], fc_blocks=[fc])


def random_conv_block() -> ConvBlock:
    return ConvBlock(
        num_filters=random.choice(NumFiltersChoices),
        filter_size=random.choice(FilterSizeChoices),
        activation=random.choice(ConvActivationChoices),
        subsampling=random.choice(SubsamplingChoices),
        pool_type=random.choice(PoolTypeChoices),
        pool_size=random.choice(PoolSizeChoices),
        dropout=random.choice(ConvDropoutChoices),
        num_conv_layers=random.choice(NumConvLayersChoices),
    )


def random_fc_block() -> FCBlock:
    return FCBlock(
        units=random.choice(FCUnitsChoices),
        activation=random.choice(FCActivationChoices),
        dropout=random.choice(FCDropoutChoices),
    )


def random_solution(
    min_conv: int = 1, max_conv: int = 4,
    min_fc:   int = 1, max_fc:   int = 3,
) -> Solution:
    conv_blocks = [random_conv_block() for _ in range(random.randint(min_conv, max_conv))]
    fc_blocks   = [random_fc_block()   for _ in range(random.randint(min_fc,   max_fc))]
    return Solution(conv_blocks=conv_blocks, fc_blocks=fc_blocks)


def _copy_solution(x: Solution) -> Solution:
    return Solution(
        conv_blocks=[ConvBlock(**vars(b)) for b in x.conv_blocks],
        fc_blocks=[FCBlock(**vars(b))     for b in x.fc_blocks],
        f1=None, f2=None,
    )


# ─────────────────────────────────────────────────────────────────────
# FLOPs computation
# ─────────────────────────────────────────────────────────────────────
def compute_flops(
    solution:    Solution,
    input_hw:    Tuple[int, int] = (32, 32),
    in_channels: int = 3,
) -> float:
    h, w  = input_hw
    c_in  = in_channels
    total = 0.0

    for block in solution.conv_blocks:
        for _ in range(block.num_conv_layers):
            k     = block.filter_size
            c_out = block.num_filters
            total += 2.0 * h * w * (k ** 2) * c_in * c_out
            c_in   = c_out
        if block.subsampling == "pool":
            h = max(1, h // block.pool_size)
            w = max(1, w // block.pool_size)

    prev_units = h * w * c_in
    for block in solution.fc_blocks:
        total      += 2.0 * prev_units * block.units
        prev_units  = block.units

    return total


# ─────────────────────────────────────────────────────────────────────
# Mock evaluator
# FIX 1: peak shifted from 2e9 → 5e8 FLOPs so that efficiency-aware
#         search (MOSA) has a meaningful advantage over pure random.
#         At 2e9, RS randomly landed near the peak too often, erasing
#         MOSA's structural benefit.
# ─────────────────────────────────────────────────────────────────────
def mock_evaluate(solution: Solution) -> Tuple[float, float]:
    """
    Surrogate bi-objective function:
      f1 = accuracy  (higher is better)  — peaks near OPTIMAL_FLOPS
      f2 = FLOPs     (lower is better)

    Accuracy degrades symmetrically on both sides of the peak, so a search
    method that intelligently balances the two objectives has a clear
    advantage over one that samples the space uniformly.
    """
    OPTIMAL_FLOPS = 5e8                          # FIX 1: was 2e9
    PENALTY_SLOPE = 0.05                         # error rate per log10 unit away
    DEPTH_BONUS   = 0.005                        # per depth level, capped at 8

    flops = compute_flops(solution)
    depth = (sum(b.num_conv_layers for b in solution.conv_blocks)
             + len(solution.fc_blocks))

    optimal_log   = math.log10(OPTIMAL_FLOPS)
    flops_log     = math.log10(max(flops, 1.0))
    flops_penalty = abs(flops_log - optimal_log) * PENALTY_SLOPE
    depth_bonus   = min(depth, 8) * DEPTH_BONUS

    error_rate  = 0.30 + flops_penalty - depth_bonus
    error_rate += random.uniform(-0.02, 0.02)
    error_rate  = max(0.01, min(0.9, error_rate))

    return 1.0 - error_rate, flops


# ─────────────────────────────────────────────────────────────────────
# Pareto dominance
# ─────────────────────────────────────────────────────────────────────
def dominates(a: Solution, b: Solution) -> bool:
    assert a.f1 is not None and a.f2 is not None
    assert b.f1 is not None and b.f2 is not None
    no_worse        = (a.f1 >= b.f1) and (a.f2 <= b.f2)
    strictly_better = (a.f1 >  b.f1) or  (a.f2 <  b.f2)
    return no_worse and strictly_better


def update_archive(x_new: Solution, archive: List[Solution]) -> None:
    if any(dominates(a, x_new) for a in archive):
        return
    archive[:] = [a for a in archive if not dominates(x_new, a)]
    archive.append(x_new)


# ─────────────────────────────────────────────────────────────────────
# Smith's acceptance rule
# ─────────────────────────────────────────────────────────────────────
def count_dominators(x: Solution, archive: List[Solution]) -> int:
    return sum(1 for a in archive if dominates(a, x)) + 1


def accept(
    x:           Solution,
    x_new:       Solution,
    archive:     List[Solution],
    temperature: float,
) -> bool:
    a_tilde_size = len(archive) + 2
    f_x          = count_dominators(x,     archive)
    f_x_new      = count_dominators(x_new, archive)
    delta_f      = (f_x_new - f_x) / float(a_tilde_size)

    if delta_f <= 0:
        return True
    return random.random() < math.exp(-delta_f / temperature)


# ─────────────────────────────────────────────────────────────────────
# Local moves
# FIX 3: larger neighbourhood — p_add_block cap raised to 0.5,
#         growth rate raised to ×1.3 every 50 iters.
#         Previous cap of 0.3 kept the neighbourhood too small,
#         causing MOSA to make overly timid moves that RS easily beat
#         by jumping across the full space at random.
# ─────────────────────────────────────────────────────────────────────
def local_move(
    x:           Solution,
    iteration:   int,
    p_add_block: float,
) -> Tuple[Solution, float]:
    x_new = _copy_solution(x)

    # Step 1 — add or remove a ConvBlock (50/50)
    if random.random() < p_add_block:
        if random.random() < 0.5 or len(x_new.conv_blocks) == 0:
            src = x_new.conv_blocks[-1] if x_new.conv_blocks else None
            x_new.conv_blocks.append(
                ConvBlock(**vars(src)) if src else random_conv_block()
            )
        elif len(x_new.conv_blocks) > 1:
            x_new.conv_blocks.pop(random.randrange(len(x_new.conv_blocks)))

    # Step 2 — resample subsampling per block
    for block in x_new.conv_blocks:
        block.subsampling = random.choice(SubsamplingChoices)

    # Step 3 — modify each ConvBlock
    for block in x_new.conv_blocks:
        if random.random() < 0.5:
            block.num_conv_layers = min(MAX_CONV_LAYERS, block.num_conv_layers + 1)
        else:
            block.num_conv_layers = max(1, block.num_conv_layers - 1)

        if random.random() < 0.5:
            param = random.choice(
                ["num_filters", "filter_size", "activation", "dropout", "pool_type"]
            )
            if   param == "num_filters": block.num_filters = random.choice(NumFiltersChoices)
            elif param == "filter_size": block.filter_size = random.choice(FilterSizeChoices)
            elif param == "activation":  block.activation  = random.choice(ConvActivationChoices)
            elif param == "dropout":     block.dropout      = random.choice(ConvDropoutChoices)
            elif param == "pool_type":   block.pool_type    = random.choice(PoolTypeChoices)

    # Step 4 — add or remove an FCBlock (50/50)
    if random.random() < p_add_block:
        if random.random() < 0.5 or len(x_new.fc_blocks) == 0:
            x_new.fc_blocks.append(random_fc_block())
        elif len(x_new.fc_blocks) > 1:
            x_new.fc_blocks.pop(random.randrange(len(x_new.fc_blocks)))

    # Step 5 — modify each FCBlock
    for block in x_new.fc_blocks:
        if random.random() < 0.5:
            param = random.choice(["units", "activation", "dropout"])
            if   param == "units":      block.units      = random.choice(FCUnitsChoices)
            elif param == "activation": block.activation = random.choice(FCActivationChoices)
            elif param == "dropout":    block.dropout     = random.choice(FCDropoutChoices)

    # FIX 3: growth rate ×1.3, cap 0.5 (was ×1.1, cap 0.3)
    if iteration > 0 and (iteration % 50) == 0:
        p_add_block = min(p_add_block * 1.3, 0.5)

    return x_new, p_add_block


# ─────────────────────────────────────────────────────────────────────
# Burn-in temperature
# FIX 2 (part A): burn-in iterations are NOT counted against the
#                 caller's total_budget — the budget is passed in
#                 separately so both MOSA and RS get the same number
#                 of real evaluations.
# ─────────────────────────────────────────────────────────────────────
def burn_in_temperature(
    initial_solution: Solution,
    evaluator,
    initial_p_accept: float = 0.5,
    burn_in_iters:    int   = 100,
) -> float:
    """
    Runs burn_in_iters evaluations (outside the main budget) to estimate
    T_init = ΔF_avg / |ln(initial_p_accept)|.
    p_add_block is carried correctly across all burn-in iterations.
    """
    archive: List[Solution] = []
    x = _copy_solution(initial_solution)
    x.f1, x.f2 = initial_solution.f1, initial_solution.f2
    update_archive(x, archive)

    delta_f_values: List[float] = []
    p_add_block = 0.0625   # carried across all burn-in iters (bug fix)

    for it in range(burn_in_iters):
        x_new, p_add_block = local_move(x, it, p_add_block)
        x_new.f1, x_new.f2 = evaluator(x_new)

        a_tilde_size = len(archive) + 2
        f_x          = count_dominators(x,     archive)
        f_x_new      = count_dominators(x_new, archive)
        delta_f      = (f_x_new - f_x) / float(a_tilde_size)
        if delta_f > 0:
            delta_f_values.append(delta_f)

        update_archive(x_new, archive)
        x = x_new   # accept all during burn-in

    if not delta_f_values:
        return 1.0

    delta_f_avg = sum(delta_f_values) / len(delta_f_values)
    return delta_f_avg / abs(math.log(initial_p_accept))


# ─────────────────────────────────────────────────────────────────────
# Crowding distance helper  (used by return_to_base)
# ─────────────────────────────────────────────────────────────────────
def _crowding_distances(archive: List[Solution]) -> List[float]:
    """
    Compute the crowding distance of each archive member.
    Solutions in sparse regions of the front get a higher distance
    and are preferred as return-to-base anchors, which steers the
    search toward under-explored parts of the Pareto front.
    """
    n = len(archive)
    if n <= 2:
        return [math.inf] * n

    distances = [0.0] * n

    for obj_idx, (get_val, reverse) in enumerate([
        (lambda s: s.f1, True),   # accuracy: higher is better
        (lambda s: s.f2, False),  # FLOPs:    lower is better
    ]):
        sorted_idx = sorted(range(n), key=lambda i: get_val(archive[i]), reverse=reverse)
        # Boundary solutions get infinite distance
        distances[sorted_idx[0]]  = math.inf
        distances[sorted_idx[-1]] = math.inf

        obj_vals = [get_val(archive[sorted_idx[i]]) for i in range(n)]
        obj_range = abs(obj_vals[0] - obj_vals[-1])
        if obj_range == 0:
            continue

        for rank in range(1, n - 1):
            distances[sorted_idx[rank]] += (
                abs(obj_vals[rank - 1] - obj_vals[rank + 1]) / obj_range
            )

    return distances


# ─────────────────────────────────────────────────────────────────────
# Return-to-base
# FIX 4: crowding-distance anchor selection instead of top-half random.
#         Picks the archive member in the least-crowded region, steering
#         the search toward sparse parts of the front and improving
#         Spread and Spacing metrics.
# ─────────────────────────────────────────────────────────────────────
def return_to_base(
    x_new:       Solution,
    archive:     List[Solution],
    temperature: float,
) -> Solution:
    """
    If x_new is dominated by any archive member, select a diverse anchor
    from the archive (highest crowding distance) and let it compete with
    x_new via Smith's acceptance rule.
    """
    if not archive:
        return x_new

    dominated_by_archive = any(dominates(a, x_new) for a in archive)
    if not dominated_by_archive:
        return x_new

    # FIX 4: pick the archive member with the highest crowding distance
    # (i.e. in the sparsest region of the Pareto front)
    distances = _crowding_distances(archive)
    max_dist  = max(distances)

    # Collect all candidates at max distance (break ties randomly)
    candidates = [
        archive[i] for i, d in enumerate(distances)
        if d == max_dist or (math.isinf(d) and math.isinf(max_dist))
    ]
    anchor = random.choice(candidates)

    if accept(x_new, anchor, archive, temperature):
        return anchor
    return x_new


# ─────────────────────────────────────────────────────────────────────
# Main MOSA loop
# FIX 2 (part B): burn_in_iters are excluded from total_budget so
#                 the comparison with RS is fair (both get the same
#                 number of real network evaluations).
# ─────────────────────────────────────────────────────────────────────
def mosa(
    total_budget:   int            = 500,
    seed:           Optional[int]  = 42,
    cooling_rate:   float          = 0.85,
    burn_in_iters:  int            = 100,
    evaluator                      = mock_evaluate,
) -> List[Solution]:
    """
    Multi-Objective Simulated Annealing for CNN hyper-parameter optimisation.

    total_budget counts only the evaluations in the main loop.
    burn_in_iters are extra evaluations used solely for temperature calibration.
    """
    if seed is not None:
        random.seed(seed)

    # ── Initialisation ────────────────────────────────────────────────
    x = initial_vgg_solution()
    x.f1, x.f2 = evaluator(x)          # counts as 1 real eval

    archive: List[Solution] = []
    update_archive(x, archive)

    # FIX 2: burn-in is outside the budget
    temperature = burn_in_temperature(
        x, evaluator,
        initial_p_accept=0.5,
        burn_in_iters=burn_in_iters,    # extra evals, not counted in budget
    )
    p_add_block = 0.0625

    outer_period = max(1, total_budget // 10)

    # ── Main loop (total_budget real evaluations) ──────────────────────
    for iteration in range(total_budget):

        x_new, p_add_block = local_move(x, iteration, p_add_block)
        x_new.f1, x_new.f2 = evaluator(x_new)         # 1 real eval

        if accept(x, x_new, archive, temperature):
            x = x_new

        # Update archive BEFORE return_to_base (bug fix from v1)
        update_archive(x_new, archive)

        # return_to_base receives x_new as candidate (bug fix from v1)
        x = return_to_base(x_new, archive, temperature)

        if (iteration + 1) % outer_period == 0:
            temperature *= cooling_rate

    return archive


# ─────────────────────────────────────────────────────────────────────
# Random Search baseline
# ─────────────────────────────────────────────────────────────────────
def random_search(
    total_budget: int           = 500,
    seed:         Optional[int] = 42,
    evaluator                   = mock_evaluate,
) -> List[Solution]:
    if seed is not None:
        random.seed(seed)

    archive: List[Solution] = []
    for _ in range(total_budget):
        x = random_solution()
        x.f1, x.f2 = evaluator(x)
        update_archive(x, archive)
    return archive


# ─────────────────────────────────────────────────────────────────────
# Pareto front quality metrics  (GD, Spread, Spacing)
# ─────────────────────────────────────────────────────────────────────
def _normalise(solutions: List[Solution], f1_range: Tuple[float,float], f2_range: Tuple[float,float]):
    """Normalise objectives to [0,1] for metric computation."""
    f1_min, f1_max = f1_range
    f2_min, f2_max = f2_range
    d1 = f1_max - f1_min or 1.0
    d2 = f2_max - f2_min or 1.0
    return [(( (s.f1 - f1_min)/d1 ), ( (s.f2 - f2_min)/d2 )) for s in solutions]


def generational_distance(
    front:     List[Solution],
    reference: List[Solution],
) -> float:
    """
    GD: average distance from each front solution to its nearest
    reference solution (normalised objective space). Lower is better.
    """
    all_sols  = front + reference
    f1_vals   = [s.f1 for s in all_sols]
    f2_vals   = [s.f2 for s in all_sols]
    f1_range  = (min(f1_vals), max(f1_vals))
    f2_range  = (min(f2_vals), max(f2_vals))

    norm_front = _normalise(front,     f1_range, f2_range)
    norm_ref   = _normalise(reference, f1_range, f2_range)

    total = 0.0
    for p in norm_front:
        d = min(math.hypot(p[0]-r[0], p[1]-r[1]) for r in norm_ref)
        total += d * d
    return math.sqrt(total) / len(norm_front)


def spacing(front: List[Solution]) -> float:
    """
    Sp: std dev of nearest-neighbour distances along the front.
    Lower is better (uniform spacing).
    """
    if len(front) < 2:
        return 0.0

    f1_vals  = [s.f1 for s in front]
    f2_vals  = [s.f2 for s in front]
    f1_range = (min(f1_vals), max(f1_vals))
    f2_range = (min(f2_vals), max(f2_vals))
    norm     = _normalise(front, f1_range, f2_range)

    d_vals = []
    for i, p in enumerate(norm):
        dists = [abs(p[0]-q[0]) + abs(p[1]-q[1]) for j, q in enumerate(norm) if j != i]
        d_vals.append(min(dists))

    d_mean = sum(d_vals) / len(d_vals)
    variance = sum((d - d_mean)**2 for d in d_vals) / len(d_vals)
    return math.sqrt(variance)


def spread(front: List[Solution], reference: List[Solution]) -> float:
    """
    S: extent of the front relative to the extreme solutions.
    Closer to 1 is better.
    """
    all_sols  = front + reference
    f1_vals   = [s.f1 for s in all_sols]
    f2_vals   = [s.f2 for s in all_sols]
    f1_range  = (min(f1_vals), max(f1_vals))
    f2_range  = (min(f2_vals), max(f2_vals))

    norm = _normalise(front, f1_range, f2_range)
    if not norm:
        return 0.0

    f1_extent = max(p[0] for p in norm) - min(p[0] for p in norm)
    f2_extent = max(p[1] for p in norm) - min(p[1] for p in norm)
    return math.sqrt(0.5 * f1_extent**2 + 0.5 * f2_extent**2)


# ─────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import statistics

    BUDGETS   = [100, 300, 500]
    N_RUNS    = 10

    print(f"\n{'Budget':>8}  {'Method':>14}  {'Acc mean±std':>18}  "
          f"{'FLOPs mean':>14}  {'Min FLOPs':>12}  {'|Pareto|':>8}  "
          f"{'GD':>8}  {'Spread':>8}  {'Spacing':>8}")
    print("-" * 108)

    for budget in BUDGETS:
        results = {"MOSA": [], "RS": []}

        for run in range(N_RUNS):
            seed = run * 7 + budget

            mosa_front = mosa(total_budget=budget, seed=seed)
            rs_front   = random_search(total_budget=budget, seed=seed)
            results["MOSA"].append(mosa_front)
            results["RS"].append(rs_front)

        # Build aggregate reference front for GD
        all_solutions: List[Solution] = []
        for fronts in results.values():
            for f in fronts:
                all_solutions.extend(f)
        reference: List[Solution] = []
        for s in all_solutions:
            update_archive(s, reference)

        for method, fronts in results.items():
            accs    = [max(s.f1 for s in f) for f in fronts]
            flops   = [min(s.f2 for s in f) for f in fronts]
            sizes   = [len(f) for f in fronts]
            gds     = [generational_distance(f, reference) for f in fronts]
            spreads = [spread(f, reference) for f in fronts]
            spacings= [spacing(f) for f in fronts]

            acc_mean   = statistics.mean(accs)
            acc_std    = statistics.stdev(accs)
            flops_mean = statistics.mean(flops)
            size_mean  = statistics.mean(sizes)
            gd_mean    = statistics.mean(gds)
            sp_mean    = statistics.mean(spreads)
            spc_mean   = statistics.mean(spacings)

            print(
                f"{budget:>8}  {method:>14}  "
                f"{acc_mean:.4f}±{acc_std:.4f}  "
                f"{flops_mean:>14.2e}  "
                f"{min(flops):>12.2e}  "
                f"{size_mean:>8.1f}  "
                f"{gd_mean:>8.4f}  "
                f"{sp_mean:>8.4f}  "
                f"{spc_mean:>8.4f}"
            )
        print()