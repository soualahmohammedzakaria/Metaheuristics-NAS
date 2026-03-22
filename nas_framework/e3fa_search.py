"""
E³-FA: Exploitation and Exploration Enhanced Firefly Algorithm
=============================================================
Population of "fireflies" optimizing CNN architectures encoded as the
project's existing IP byte-pair representation (ip_layer.py).

Enhancements over standard FA:
  1. Probabilistic scatter around best solution (exploitation, p > 0.5)
  2. Replace worst 20% with random solutions each iteration (exploration)

Parameters: alpha=0.5, gamma=1.0, beta_0=0.2, sigma_0=3, t_max=20, N=8
Fitness = acc component of IPPSOEvaluator output (maximize).

FIX 1 — Evaluation count:
    The original nested i×j loop called _fitness() inside the inner loop,
    producing up to N² evaluations per iteration instead of the paper's
    intended ~N.  The corrected version makes ONE movement attempt per
    firefly per iteration: for each firefly i, find the single brightest
    firefly j that is better than i, perform the move, evaluate once.

FIX 2 — Sigma floor:
    sigma = sigma_0 * (1 - t/t_max) reaches exactly 0 at the final
    iteration, collapsing the normal distribution.  A hard floor of 0.05
    is applied so the last iterations still produce meaningful scatter.

FIX 3 — Dynamic alpha decay:
    The original formula produced alpha → 0 almost immediately.  Replaced
    with the standard FA decay: alpha(t) = alpha_0 * delta^t where
    delta = (1e-4 / 0.9) ^ (1 / t_max), which keeps alpha positive and
    meaningful throughout the run.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

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
    """Clip bytes to [0,255], round to int, enforce per-slot validity."""
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

class E3FASearch:
    """
    Exploitation and Exploration Enhanced Firefly Algorithm (E³-FA).

    Parameters
    ----------
    evaluator : IPPSOEvaluator
        Shared evaluator (mock or real CIFAR-10).
    N : int
        Population size (default 8).
    t_max : int
        Number of iterations (default 20).
    alpha : float
        Initial randomization step size (default 0.5).
    gamma : float
        Light absorption coefficient (default 1.0).
    beta_0 : float
        Attractiveness at distance 0 (default 0.2).
    sigma_0 : float
        Initial std for exploitation scatter (default 3.0).
    sigma_floor : float
        Minimum std so the normal distribution never degenerates (default 0.05).
    """

    def __init__(self, evaluator, N: int = 8, t_max: int = 20,
                 alpha: float = 0.5, gamma: float = 1.0,
                 beta_0: float = 0.2, sigma_0: float = 3.0,
                 sigma_floor: float = 0.05):
        self.evaluator    = evaluator
        self.N            = N
        self.t_max        = t_max
        self.alpha0       = alpha
        self.gamma        = gamma
        self.beta_0       = beta_0
        self.sigma_0      = sigma_0
        self.sigma_floor  = sigma_floor

        self.evaluations  = 0

    # ── fitness ───────────────────────────────────────────────────────────────

    def _fitness(self, sol: List[int]) -> float:
        result = self.evaluator.evaluate(sol)
        self.evaluations += 1
        return result[0]  # acc

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_population(self) -> Tuple[List[List[int]], List[float]]:
        solutions = [_random_solution() for _ in range(self.N)]
        fitnesses = [self._fitness(s) for s in solutions]
        return solutions, fitnesses

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self) -> Tuple[List[int], float, List[float]]:
        """
        Run E³-FA for t_max iterations.

        Returns
        -------
        best_solution : list[int]
        best_fitness  : float
        history       : list[float]  best fitness at end of each iteration
        """
        solutions, fitnesses = self._init_population()

        # Sort descending: brightest (highest fitness) first
        order     = sorted(range(self.N), key=lambda i: fitnesses[i], reverse=True)
        solutions = [solutions[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        best_sol  = list(solutions[0])
        best_fit  = fitnesses[0]
        history: List[float] = [best_fit]

        dim   = MAX_LENGTH * 2
        alpha = self.alpha0

        # FIX 3: correct decay factor — keeps alpha positive across all iters
        delta = (1e-4 / 0.9) ** (1.0 / self.t_max)

        for t in range(1, self.t_max + 1):

            # FIX 2: sigma decays linearly to sigma_floor, never to 0
            raw_sigma = self.sigma_0 * (1.0 - t / self.t_max)
            sigma     = max(raw_sigma, self.sigma_floor)

            # -- Step 1: one move per firefly (FIX 1) -------------------------
            # For each firefly i, find the single brightest firefly j > i,
            # perform one movement, evaluate once.  This matches the paper's
            # intended ~N evaluations per iteration instead of N².
            for i in range(self.N):

                # Find brightest firefly that is better than i
                best_j     = None
                best_j_fit = fitnesses[i]
                for j in range(self.N):
                    if j != i and fitnesses[j] > best_j_fit:
                        best_j_fit = fitnesses[j]
                        best_j     = j

                if best_j is None:
                    # No brighter firefly exists → no movement this step
                    continue

                p = random.random()

                if p > 0.5:
                    # Exploitation: scatter around global best
                    noise   = np.random.normal(0.0, sigma, size=dim)
                    new_sol = [b + n for b, n in zip(best_sol, noise)]
                else:
                    # Standard FA movement toward brightest neighbour
                    r_ij    = _euclidean(solutions[i], solutions[best_j])
                    beta    = self.beta_0 / (1.0 + self.gamma * r_ij ** 2)
                    rand_vec = [random.random() for _ in range(dim)]
                    new_sol  = [
                        solutions[i][d]
                        + beta * (solutions[best_j][d] - solutions[i][d])
                        + alpha * (rand_vec[d] - 0.5)
                        for d in range(dim)
                    ]

                new_sol = _clip_solution(new_sol)
                new_fit = self._fitness(new_sol)   # exactly ONE eval per firefly

                # Greedy selection
                if new_fit > fitnesses[i]:
                    solutions[i] = new_sol
                    fitnesses[i] = new_fit

            # -- Step 2: sort descending by fitness (brightest first) ----------
            order     = sorted(range(self.N), key=lambda i: fitnesses[i], reverse=True)
            solutions = [solutions[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # -- Step 3: replace worst 20% with random solutions ---------------
            n_replace = max(1, math.floor(0.2 * self.N))
            for k in range(self.N - n_replace, self.N):
                new_sol        = _random_solution()
                solutions[k]   = new_sol
                fitnesses[k]   = self._fitness(new_sol)

            # -- Step 4: track global best -------------------------------------
            if fitnesses[0] > best_fit:
                best_fit = fitnesses[0]
                best_sol = list(solutions[0])

            history.append(best_fit)

            # FIX 3: standard FA alpha decay
            alpha = self.alpha0 * (delta ** t)

        return best_sol, best_fit, history