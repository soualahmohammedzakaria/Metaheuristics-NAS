from __future__ import annotations

from typing import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nas_framework.population import Individual


def _as_maximization_vector(fitness: tuple[float, ...],
                            directions: tuple[int, ...]) -> tuple[float, ...]:
    # direction: +1 means maximize objective, -1 means minimize objective
    return tuple(value * direction for value, direction in zip(fitness, directions))


def dominates(a: Individual, b: Individual,
              directions: tuple[int, ...]) -> bool:
    af = _as_maximization_vector(a.fitness, directions)
    bf = _as_maximization_vector(b.fitness, directions)
    no_worse = all(x >= y for x, y in zip(af, bf))
    strictly_better = any(x > y for x, y in zip(af, bf))
    return no_worse and strictly_better


def fast_non_dominated_sort(individuals: list[Individual],
                            directions: tuple[int, ...]) -> list[list[Individual]]:
    fronts: list[list[Individual]] = []
    domination_count: dict[int, int] = {}
    dominates_set: dict[int, list[int]] = {}

    for i, p in enumerate(individuals):
        domination_count[i] = 0
        dominates_set[i] = []
        for j, q in enumerate(individuals):
            if i == j:
                continue
            if dominates(p, q, directions):
                dominates_set[i].append(j)
            elif dominates(q, p, directions):
                domination_count[i] += 1

        if domination_count[i] == 0:
            p.rank = 0

    current_front = [i for i in range(len(individuals)) if domination_count[i] == 0]
    front_idx = 0

    while current_front:
        fronts.append([individuals[i] for i in current_front])
        next_front: list[int] = []
        for i in current_front:
            for j in dominates_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    individuals[j].rank = front_idx + 1
                    next_front.append(j)
        front_idx += 1
        current_front = next_front

    return fronts


def crowding_distance(front: list[Individual],
                      directions: tuple[int, ...]) -> None:
    if not front:
        return
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    for ind in front:
        ind.crowding_distance = 0.0

    n_obj = len(front[0].fitness)
    for obj in range(n_obj):
        front.sort(key=lambda ind: ind.fitness[obj] * directions[obj])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        min_v = front[0].fitness[obj] * directions[obj]
        max_v = front[-1].fitness[obj] * directions[obj]
        if max_v == min_v:
            continue

        for i in range(1, len(front) - 1):
            prev_v = front[i - 1].fitness[obj] * directions[obj]
            next_v = front[i + 1].fitness[obj] * directions[obj]
            front[i].crowding_distance += (next_v - prev_v) / (max_v - min_v)


def assign_rank_and_crowding(individuals: list[Individual],
                             directions: tuple[int, ...]) -> list[list[Individual]]:
    fronts = fast_non_dominated_sort(individuals, directions)
    for front in fronts:
        crowding_distance(front, directions)
    return fronts


def pareto_sort_key(ind: Individual) -> tuple[float, float]:
    # lower rank is better, higher crowding is better
    return (float(ind.rank), -float(ind.crowding_distance))


def take_pareto_best(individuals: list[Individual], n: int,
                     directions: tuple[int, ...]) -> list[Individual]:
    if not individuals:
        return []
    fronts = assign_rank_and_crowding(individuals, directions)
    selected: list[Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
            continue
        front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
        selected.extend(front[: n - len(selected)])
        break
    return selected


def pareto_front(individuals: Iterable[Individual],
                 directions: tuple[int, ...]) -> list[Individual]:
    inds = list(individuals)
    if not inds:
        return []
    fronts = assign_rank_and_crowding(inds, directions)
    return fronts[0] if fronts else []


def exact_pareto_front_2d(individuals: Iterable[Individual],
                          directions: tuple[int, int]) -> list[Individual]:
    """Exact first Pareto front for 2 objectives using sort+sweep.

    This is equivalent to pairwise dominance for finite values and runs in
    O(N log N) due to sorting.
    """
    inds = [ind for ind in individuals if ind.fitness is not None]
    if not inds:
        return []

    valid: list[tuple[float, float, Individual]] = []
    nan_like: list[Individual] = []
    for ind in inds:
        x = ind.fitness[0] * directions[0]
        y = ind.fitness[1] * directions[1]
        if x != x or y != y:
            # Keep behavior compatible with pairwise dominance semantics where
            # NaN values are effectively non-dominated.
            nan_like.append(ind)
            continue
        valid.append((x, y, ind))

    if not valid:
        return nan_like

    valid.sort(key=lambda row: (-row[0], -row[1]))
    front: list[Individual] = []
    best_y_seen = float("-inf")
    i = 0
    while i < len(valid):
        x_value = valid[i][0]
        group_start = i
        group_best_y = valid[i][1]
        while i < len(valid) and valid[i][0] == x_value:
            if valid[i][1] > group_best_y:
                group_best_y = valid[i][1]
            i += 1

        if group_best_y > best_y_seen:
            for j in range(group_start, i):
                _, y_value, ind = valid[j]
                if y_value == group_best_y:
                    ind.rank = 0
                    front.append(ind)
            best_y_seen = group_best_y

    front.extend(nan_like)
    crowding_distance(front, directions)
    return front


# ---------------------------------------------------------------------------
# Rank-based scalar scoring  (RB-IFA, Nguyen et al., ICAART 2025)
# ---------------------------------------------------------------------------

def rank_based_score(individuals: list["Individual"],
                     directions: tuple[int, ...],
                     w_perf: float = 0.5) -> list[float]:
    """Compute a scalar rank score for each individual (Eq. 6 of RB-IFA).

    Instead of building a Pareto front, each objective is ranked
    independently and the ranks are combined into a single weighted sum:

        R_solution = mean(R_perf_objectives) * w_perf
                   + mean(R_cost_objectives) * w_cost

    where w_perf + w_cost = 1.

    Convention used here (consistent with the paper):
      - Objectives with direction +1 (maximise, e.g. accuracy) are
        performance objectives.  Rank 1 = highest value = best.
      - Objectives with direction -1 (minimise, e.g. latency) are cost
        objectives.  Rank 1 = lowest value = best.

    The lower the returned score, the better the individual.

    Parameters
    ----------
    individuals : list of Individual with non-None fitness.
    directions  : per-objective direction (+1 maximise, -1 minimise).
    w_perf      : weight given to performance objectives (default 0.5).
                  Cost weight is derived as 1 - w_perf.

    Returns
    -------
    List of float scores in the same order as *individuals*.
    Lower score = better individual.
    """
    w_cost = 1.0 - w_perf
    eligible = [ind for ind in individuals if ind.fitness is not None]
    if not eligible:
        return [0.0] * len(individuals)

    n = len(eligible)
    n_obj = len(eligible[0].fitness)

    perf_cols = [o for o in range(n_obj) if o < len(directions) and directions[o] == 1]
    cost_cols = [o for o in range(n_obj) if o < len(directions) and directions[o] == -1]

    # Fallback: treat all as performance if no cost objective defined.
    if not perf_cols:
        perf_cols = list(range(n_obj))
    if not cost_cols:
        w_cost = 0.0

    # Rank each objective independently (rank 1 = best).
    obj_ranks: list[list[float]] = []
    for o in range(n_obj):
        direction = directions[o] if o < len(directions) else 1
        # Sort descending for perf (+1), ascending for cost (-1).
        sorted_inds = sorted(
            range(n), key=lambda i: eligible[i].fitness[o] * direction, reverse=True
        )
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_inds):
            ranks[idx] = float(rank + 1)
        obj_ranks.append(ranks)

    scores: list[float] = []
    for i in range(n):
        perf_rank = (sum(obj_ranks[o][i] for o in perf_cols) / len(perf_cols)
                     if perf_cols else 0.0)
        cost_rank = (sum(obj_ranks[o][i] for o in cost_cols) / len(cost_cols)
                     if cost_cols else 0.0)
        scores.append(perf_rank * w_perf + cost_rank * w_cost)

    # Map back to the original individuals list (non-eligible get worst score).
    worst = float(n + 1)
    eligible_set = set(id(ind) for ind in eligible)
    eligible_scores = iter(scores)
    result = []
    for ind in individuals:
        if id(ind) in eligible_set:
            result.append(next(eligible_scores))
        else:
            result.append(worst)
    return result