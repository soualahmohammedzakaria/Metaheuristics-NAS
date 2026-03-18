"""
Differential Evolution operators for MO-DEHB.

Implements:
  - rand1_mutation      : DE/rand/1 mutation over a parent pool.
  - binomial_crossover  : standard DE binomial crossover.
  - mo_de_selection     : Algorithm 1 — MO-aware selection using NDS
                          front ranks and dominated HV contribution.

All NDS logic delegates to mo_utils.fast_non_dominated_sort.
Crowding distance delegates to mo_utils.crowding_distance.
"""
from __future__ import annotations

import math
import random

from nas_framework.population import Individual
from nas_framework.mo_utils import (
    fast_non_dominated_sort,   
    crowding_distance,         
)


# ──────────────────────────────────────────────────────────────────────────────
# DE mutation  (rand/1)
# ──────────────────────────────────────────────────────────────────────────────

def rand1_mutation(
    parent_pool: list[Individual],
    n_edges: int,
    n_ops: int,
    F: float = 0.5,
) -> list[int]:
    """DE/rand/1 mutation adapted for integer genotypes.

    Selects three distinct individuals from *parent_pool* and computes:
        mutant = x1 + F * (x2 - x3)   (rounded and clipped to [0, n_ops-1])

    Falls back to random genotype when pool is too small.
    """
    if len(parent_pool) < 3:
        return [random.randint(0, n_ops - 1) for _ in range(n_edges)]

    x1, x2, x3 = random.sample(parent_pool, 3)
    mutant: list[int] = []
    for g1, g2, g3 in zip(x1.genotype, x2.genotype, x3.genotype):
        val = int(round(g1 + F * (g2 - g3)))
        mutant.append(max(0, min(n_ops - 1, val)))
    return mutant


# ──────────────────────────────────────────────────────────────────────────────
# DE crossover  (binomial)
# ──────────────────────────────────────────────────────────────────────────────

def binomial_crossover(
    target: Individual,
    mutant: list[int],
    CR: float = 0.5,
) -> Individual:
    """Binomial crossover between *target* genotype and *mutant* vector.

    Each gene is taken from *mutant* with probability CR, otherwise from
    *target*. At least one gene is always from *mutant* (j_rand).
    """
    n = len(target.genotype)
    j_rand = random.randrange(n)
    child_geno = [
        mutant[i] if (random.random() < CR or i == j_rand) else target.genotype[i]
        for i in range(n)
    ]
    return Individual(child_geno)


# ──────────────────────────────────────────────────────────────────────────────
# HV contribution helper
# ──────────────────────────────────────────────────────────────────────────────

def _hypervolume_contribution(
    individual: Individual,
    front: list[Individual],
    directions: tuple[int, ...],
) -> float:
    """Approximate HV contribution of *individual* within *front*.

    Uses exact 2-D sweep for bi-objective case.
    Falls back to crowding distance (via mo_utils) for >2 objectives.
    """
    if len(front) <= 1:
        return 0.0

    def to_max(ind: Individual) -> tuple[float, ...]:
        return tuple(v * d for v, d in zip(ind.fitness, directions))

    max_vecs = [to_max(ind) for ind in front]
    n_obj = len(directions)

    ref_point = tuple(min(v[i] for v in max_vecs) - 1.0 for i in range(n_obj))

    def hv_2d(points: list[tuple[float, ...]]) -> float:
        pts = sorted(points, key=lambda p: p[0], reverse=True)
        hv, prev_y = 0.0, ref_point[1]
        for p in pts:
            if p[1] > prev_y:
                hv += (p[0] - ref_point[0]) * (p[1] - prev_y)
                prev_y = p[1]
        return hv

    my_vec = to_max(individual)
    others  = [v for v in max_vecs if v != my_vec]

    if n_obj == 2:
        return hv_2d(max_vecs) - (hv_2d(others) if others else 0.0)

    crowding_distance(front, directions)
    return individual.crowding_distance


# ──────────────────────────────────────────────────────────────────────────────
# MO-DE selection 
# ──────────────────────────────────────────────────────────────────────────────

def mo_de_selection(
    target: Individual,
    offspring: Individual,
    global_pop: list[Individual],
    directions: tuple[int, ...],
) -> Individual:
    """Decide which of *target* or *offspring* survives.

    Steps
    -----
    1. NDS on global_pop (includes both target and offspring).
    2. Compare front ranks of target vs offspring.
    3. Better-ranked one wins.
    4. Tie: keep the individual with greater HV contribution;
       replace the least HV contributor.
    """
    if not global_pop:
        return offspring

    pool = list(global_pop)
    ids_in_pool = {id(ind) for ind in pool}
    if id(target) not in ids_in_pool:
        pool.append(target)
    if id(offspring) not in ids_in_pool:
        pool.append(offspring)

    fronts = fast_non_dominated_sort(pool, directions)

    rank_map: dict[int, int] = {
        id(ind): fi
        for fi, front in enumerate(fronts)
        for ind in front
    }

    r_target   = rank_map.get(id(target),   len(fronts))
    r_offspring = rank_map.get(id(offspring), len(fronts))

    if r_offspring < r_target:
        return offspring
    if r_target < r_offspring:
        return target

    same_front = fronts[r_target] if r_target < len(fronts) else [target, offspring]
    hv_t = _hypervolume_contribution(target,   same_front, directions)
    hv_o = _hypervolume_contribution(offspring, same_front, directions)
    return offspring if hv_o >= hv_t else target
