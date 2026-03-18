"""
Non-dominated sorting utilities for MO-DEHB.

Delegates NDS and crowding-distance to the existing mo_utils module.
Only adds what is new:
  - rank_by_crowding  : NSGA-II intra-front selection using mo_utils.crowding_distance
  - rank_by_epsnet    : EpsNet intra-front selection (max-min distance)
  - select_top_mo     : unified top-k selector (NDS + chosen intra-front strategy)
"""
from __future__ import annotations

import math
import random
from typing import Literal

from nas_framework.population import Individual
from nas_framework.mo_utils import (
    fast_non_dominated_sort,   
    crowding_distance,         
)


# ──────────────────────────────────────────────────────────────────────────────
# NSGA-II intra-front selection
# ──────────────────────────────────────────────────────────────────────────────

def rank_by_crowding(
    front: list[Individual],
    directions: tuple[int, ...],
    k: int,
) -> list[Individual]:
    """Select *k* individuals from *front* with the highest crowding distance.
    """
    crowding_distance(front, directions)
    ranked = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
    return ranked[:k]


# ──────────────────────────────────────────────────────────────────────────────
# EpsNet intra-front selection  
# ──────────────────────────────────────────────────────────────────────────────

def _euclidean_fitness(a: Individual, b: Individual) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.fitness, b.fitness)))


def rank_by_epsnet(front: list[Individual], k: int) -> list[Individual]:
    """Iteratively select *k* individuals maximising min-distance to chosen set.

    EpsNet strategy: start with a random individual, then greedily add
    the one furthest (in objective space) from the already-chosen set.
    """
    if k >= len(front):
        return list(front)

    remaining = list(front)
    first = random.choice(remaining)
    ranked = [first]
    remaining.remove(first)

    while len(ranked) < k and remaining:
        best = max(
            remaining,
            key=lambda ind: min(_euclidean_fitness(ind, r) for r in ranked),
        )
        ranked.append(best)
        remaining.remove(best)

    return ranked


# ──────────────────────────────────────────────────────────────────────────────
# Unified top-k selector
# ──────────────────────────────────────────────────────────────────────────────

Strategy = Literal["nsga2", "epsnet"]


def select_top_mo(
    individuals: list[Individual],
    k: int,
    directions: tuple[int, ...],
    strategy: Strategy = "nsga2",
) -> list[Individual]:
    """Select *k* individuals using NDS + intra-front strategy.
    Uses ``mo_utils.fast_non_dominated_sort`` for front partitioning,
    then applies the chosen intra-front strategy for tie-breaking.
    """
    if not individuals:
        return []
    k = min(k, len(individuals))

    fronts = fast_non_dominated_sort(individuals, directions)
    selected: list[Individual] = []

    for front in fronts:
        remaining_slots = k - len(selected)
        if remaining_slots <= 0:
            break
        if len(front) <= remaining_slots:
            selected.extend(front)
        else:
            if strategy == "nsga2":
                selected.extend(rank_by_crowding(front, directions, remaining_slots))
            else:
                selected.extend(rank_by_epsnet(front, remaining_slots))

    return selected
