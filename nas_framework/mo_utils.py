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


def compute_crowding_distance(front: list[Individual]) -> None:
    """Dvolver/NSGA-II crowding distance for maximization objectives."""
    if not front:
        return
    directions = (1,) * len(front[0].fitness)
    crowding_distance(front, directions)


def crowded_comparison(ind_a: Individual, ind_b: Individual) -> bool:
    """True if ind_a is preferred over ind_b by rank then crowding distance."""
    if ind_a.rank != ind_b.rank:
        return ind_a.rank < ind_b.rank
    return ind_a.crowding_distance > ind_b.crowding_distance


def fast_non_dominated_sort_max(population: list[Individual]) -> list[list[Individual]]:
    """NSGA-II non-dominated sorting with all objectives maximized."""
    directions = (1,) * len(population[0].fitness) if population else (1,)
    return fast_non_dominated_sort(population, directions)


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


def _point_dominates(p: tuple[float, float], q: tuple[float, float]) -> bool:
    return (p[0] >= q[0] and p[1] >= q[1]) and (p[0] > q[0] or p[1] > q[1])


def _non_dominated_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    nd: list[tuple[float, float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if _point_dominates(q, p):
                dominated = True
                break
        if not dominated:
            nd.append(p)

    return sorted(set(nd), key=lambda x: (x[0], x[1]))


def compute_hypervolume(pareto_front: list[Individual],
                        reference_point: tuple[float, float] = (0.0, 0.0)) -> float:
    """2D hypervolume for maximize-maximize objectives using an x-sweep."""
    if not pareto_front:
        return 0.0

    rx, ry = reference_point
    points: list[tuple[float, float]] = []
    for ind in pareto_front:
        if ind.fitness is None:
            continue
        x = float(ind.fitness[0])
        y = float(ind.fitness[1])
        if x > rx and y > ry:
            points.append((x, y))

    if not points:
        return 0.0

    nd_points = _non_dominated_points(points)
    hv = 0.0
    prev_x = rx
    for x, y in nd_points:
        width = max(0.0, x - prev_x)
        height = max(0.0, y - ry)
        hv += width * height
        prev_x = x
    return hv

