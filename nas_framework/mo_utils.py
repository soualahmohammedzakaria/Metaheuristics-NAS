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

