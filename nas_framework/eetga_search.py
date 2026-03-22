"""
EE-TGA: Exploitation-Enhanced Tree Growth Algorithm
====================================================
Population-based optimizer for CNN NAS using the project's existing
IP byte-pair encoding (ip_layer.py) and IPPSOEvaluator.

Population structure (N=8):
  N1 = 3  best trees        (exploitation)
  N2 = 3  light seekers     (exploration via neighbours)
  N3 = 2  worst trees       (replaced with random)
  N4 = 1  offspring         (mask crossover with best)

Each solution is a list[int] of length MAX_LENGTH*2 (= 18 bytes).
Fitness = acc component from evaluator tuple (maximize).

FIX 1 — N1 bottom-60% update formula:
    The original code applied x/theta + r*x which for theta=0.2 gives
    5x + r*x — a massive upward explosion of the solution values on every
    iteration, immediately pushing all coordinates to 255 (clipped).
    The paper's formula is x_{i}^{j+1} = x_i^j / theta + r * x_i^j,
    meaning the NEW solution replaces the current one only if it is better
    (greedy), but the formula itself must keep values in range.
    Correct interpretation: scale the deviation from best, not the raw
    value. Fixed to:  new = best + (old - best) * r * theta
    which is a convex interpolation that keeps the result near the best
    solution and within reasonable range.

FIX 2 — Sigma floor:
    sigma = sigma_0 * (1 - t/t_max) reaches exactly 0 at the final
    iteration. A hard floor of 0.05 is applied so the scatter never
    fully collapses.

FIX 3 — N4 mask operator:
    The original offspring was generated fully at random and then had
    some dimensions overwritten by best.  Corrected to start FROM best
    and randomly replace some dimensions with new random bytes, which
    better reflects the paper's "mask operator with respect to the best
    solution".
"""
from __future__ import annotations

import math
import random
import numpy as np
from typing import List, Tuple

from nas_framework.ip_layer import (
    MAX_LENGTH,
    resample_valid_for_slot,
    is_valid_for_slot,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _random_solution() -> List[int]:
    """Generate a random valid IP-encoded solution."""
    pos: List[int] = []
    for slot in range(MAX_LENGTH):
        b0, b1 = resample_valid_for_slot(slot, pos)
        pos.extend([b0, b1])
    return pos


def _clip_solution(sol: List[int]) -> List[int]:
    """Clip each byte to [0,255] and integer-round, then enforce slot validity."""
    clipped = [max(0, min(255, round(v))) for v in sol]
    for slot in range(MAX_LENGTH):
        idx = slot * 2
        if not is_valid_for_slot(slot, clipped[idx], clipped):
            b0, b1 = resample_valid_for_slot(slot, clipped)
            clipped[idx] = b0
            clipped[idx + 1] = b1
    return clipped


def _euclidean(a: List[int], b: List[int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ── main class ────────────────────────────────────────────────────────────────

class EETGASearch:
    """
    Exploitation-Enhanced Tree Growth Algorithm (EE-TGA).

    Parameters
    ----------
    evaluator : IPPSOEvaluator
        Shared evaluator (mock or real CIFAR-10).
    N : int
        Total population size (default 8).
    t_max : int
        Number of iterations (default 20).
    sigma_0 : float
        Initial scatter std-deviation for N1 top-40% update (default 3).
    sigma_floor : float
        Minimum std so the normal distribution never degenerates (default 0.05).
    theta : float
        Scaling factor for N1 bottom-60% update (default 0.2).
    lam : float
        Lambda for N2 linear combination of two nearest neighbours (default 0.5).
    """

    def __init__(self, evaluator, N: int = 8, t_max: int = 20,
                 sigma_0: float = 3.0, sigma_floor: float = 0.05,
                 theta: float = 0.2, lam: float = 0.5):
        self.evaluator   = evaluator
        self.N           = N
        self.t_max       = t_max
        self.sigma_0     = sigma_0
        self.sigma_floor = sigma_floor
        self.theta       = theta
        self.lam         = lam

        # Sub-group sizes as per paper
        self.n1 = 3   # best
        self.n2 = 3   # light seekers
        self.n3 = 2   # worst (discarded)
        self.n4 = 1   # offspring

        self.evaluations = 0

    # ── fitness helper ────────────────────────────────────────────────────────

    def _fitness(self, sol: List[int]) -> float:
        """Evaluate solution; return scalar accuracy (maximize)."""
        result = self.evaluator.evaluate(sol)
        self.evaluations += 1
        return result[0]

    # ── population initialisation ─────────────────────────────────────────────

    def _init_population(self) -> Tuple[List[List[int]], List[float]]:
        solutions = [_random_solution() for _ in range(self.N)]
        fitnesses = [self._fitness(s) for s in solutions]
        return solutions, fitnesses

    # ── per-iteration update ──────────────────────────────────────────────────

    def _update(self, solutions: List[List[int]], fitnesses: List[float],
                t: int) -> Tuple[List[List[int]], List[float]]:

        # Sort descending → best first
        order     = sorted(range(self.N), key=lambda i: fitnesses[i], reverse=True)
        solutions = [solutions[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        best_sol = solutions[0]

        # FIX 2: sigma decays linearly but never reaches 0
        raw_sigma = self.sigma_0 * (1.0 - t / self.t_max)
        sigma     = max(raw_sigma, self.sigma_floor)

        # ── Step 1: N1 group (indices 0 .. n1-1) ─────────────────────────────
        n1_top = math.ceil(self.n1 * 0.4)   # top 40% of N1

        for i in range(self.n1):
            old_sol = solutions[i]
            old_fit = fitnesses[i]

            if i < n1_top:
                # Top 40%: scatter around best using normal distribution
                noise   = np.random.normal(loc=0.0, scale=sigma,
                                           size=len(best_sol))
                new_sol = [b + n for b, n in zip(best_sol, noise)]

            else:
                # FIX 1: bottom 60% — interpolate between old and best
                # new = best + (old - best) * r * theta
                # This is a contraction toward best, stays in-range,
                # matches the paper's intent of "reduction rate theta".
                r       = random.random()
                new_sol = [
                    b + (o - b) * r * self.theta
                    for b, o in zip(best_sol, old_sol)
                ]

            new_sol = _clip_solution(new_sol)
            new_fit = self._fitness(new_sol)

            # Greedy selection
            if new_fit > old_fit:
                solutions[i] = new_sol
                fitnesses[i] = new_fit

        # ── Step 2: N2 group (indices n1 .. n1+n2-1) ─────────────────────────
        for i in range(self.n1, self.n1 + self.n2):
            old_sol = solutions[i]
            old_fit = fitnesses[i]

            # Find 2 nearest neighbours by Euclidean distance among ALL solutions
            dists = [
                (j, _euclidean(old_sol, solutions[j]))
                for j in range(self.N) if j != i
            ]
            dists.sort(key=lambda x: x[1])
            x1 = solutions[dists[0][0]]
            x2 = solutions[dists[1][0]]

            # Linear combination of two nearest neighbours
            y       = [self.lam * a + (1.0 - self.lam) * b for a, b in zip(x1, x2)]
            alpha_i = random.random()
            new_sol = [o + alpha_i * yi for o, yi in zip(old_sol, y)]
            new_sol = _clip_solution(new_sol)
            new_fit = self._fitness(new_sol)

            if new_fit > old_fit:
                solutions[i] = new_sol
                fitnesses[i] = new_fit

        # ── Step 3: N3 group — replace worst 2 with random solutions ──────────
        for i in range(self.n1 + self.n2, self.n1 + self.n2 + self.n3):
            new_sol        = _random_solution()
            solutions[i]   = new_sol
            fitnesses[i]   = self._fitness(new_sol)

        # ── Step 4: N4 — mask crossover offspring then merge ─────────────────
        # FIX 3: start FROM best, randomly replace some dims with new random bytes
        offspring = list(best_sol)
        random_sol = _random_solution()
        for d in range(len(offspring)):
            if random.random() < 0.5:
                offspring[d] = random_sol[d]
        offspring     = _clip_solution(offspring)
        offspring_fit = self._fitness(offspring)

        # Merge offspring into population, then keep best N
        solutions.append(offspring)
        fitnesses.append(offspring_fit)

        # ── Step 5: Sort all, keep best N ────────────────────────────────────
        order     = sorted(range(len(solutions)),
                           key=lambda i: fitnesses[i], reverse=True)
        solutions = [solutions[i] for i in order[:self.N]]
        fitnesses = [fitnesses[i] for i in order[:self.N]]

        return solutions, fitnesses

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self) -> Tuple[List[int], float, List[float]]:
        """
        Run EE-TGA for t_max iterations.

        Returns
        -------
        best_solution : list[int]
        best_fitness  : float
        history       : list[float]  best fitness at end of each iteration
        """
        solutions, fitnesses = self._init_population()
        history: List[float] = [max(fitnesses)]

        for t in range(1, self.t_max + 1):
            solutions, fitnesses = self._update(solutions, fitnesses, t)
            history.append(max(fitnesses))

        best_idx = fitnesses.index(max(fitnesses))
        return solutions[best_idx], fitnesses[best_idx], history