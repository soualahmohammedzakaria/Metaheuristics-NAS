"""
SA-NAS — Simulated Annealing for Neural Architecture Search.

    Initialize K, N, T, b, c, k=1
    while k < K:
        for n in 1..N:
            sample neighbour α_n  (Eq. 3: α + perturbation)
            evaluate L(α) and L(α_n)
            if L(α_n) < L(α):          # neighbour is better
                α = α_n
            else:
                p = exp(-(L(α_n)-L(α)) / (b*T))
                if rand() < p: α = α_n
            T = c * T
        k += 1

Adaptation to our benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The paper uses validation loss as the objective (minimise).
Our benchmark provides accuracy (maximise) and latency (minimise).

We convert to a single scalar loss used only for the SA acceptance
decision:
    L(α) = −accuracy(α) + γ * latency(α)

where γ is a user-controlled trade-off weight (default 0, meaning
latency is ignored in the SA criterion and only accuracy drives the
search — matching the paper's single-objective framing).

The final result is the Pareto front of ALL accepted architectures
recorded throughout the run, using the existing mo_utils.pareto_front.

Neighbour generation reuses SinglePointMutation from mutation.py.
"""
from __future__ import annotations

import math
import random

from nas_framework.evaluator import Evaluator
from nas_framework.history import History
from nas_framework.mo_utils import pareto_front as compute_pareto_front
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Individual
from nas_framework.search_space import CSVSearchSpace
from nas_framework.termination import MaxEvaluationsTermination, Termination


class SANAS:
    """Simulated Annealing for NAS (Algorithm 1).

    Parameters
    ----------
    search_space  : CSVSearchSpace.
    evaluator     : Evaluator (accuracy + latency).
    budget        : total evaluation budget (overridden by termination).
    N             : number of neighbours sampled per iteration (paper: N=1).
    T             : initial temperature (paper: T=1e5).
    b             : Boltzmann constant scale (paper: b=1).
    c             : cooling rate, T ← c·T each step (paper: c=0.98).
    gamma         : latency weight in scalar loss L = −acc + γ·lat.
                    Default 0 → accuracy-only criterion (paper setting).
    termination   : custom termination; overrides budget.
    history       : History instance for recording statistics.
    """

    def __init__(
        self,
        search_space: CSVSearchSpace,
        evaluator: Evaluator,
        budget: int = 500,
        N: int = 1,
        T: float = 1e5,
        b: float = 1.0,
        c: float = 0.98,
        gamma: float = 0.0,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        self.search_space = search_space
        self.evaluator    = evaluator
        self.N            = N
        self.T            = T
        self.b            = b
        self.c            = c
        self.gamma        = gamma
        self.termination  = termination or MaxEvaluationsTermination(budget)
        self.history      = history or History()

        # Reuse existing SinglePointMutation for neighbour generation.
        self._mutator = SinglePointMutation(search_space)

        self.evaluations: int = 0
        self.generations: int = 0   # = outer iteration k

        # All accepted individuals (for final Pareto front).
        self._accepted: list[Individual] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[Individual]:
        """Execute SA-NAS. Returns the Pareto front of accepted architectures."""
        T = self.T

        # Initialise current architecture randomly.
        current = self._sample_and_eval()
        self._accepted.append(current)
        self._record([current])

        # Track unique genotypes seen to avoid duplicate Pareto computation.
        _seen: set[tuple[int, ...]] = {tuple(current.genotype)}

        while not self.termination.should_stop(self.evaluations, self.generations):
            for _ in range(self.N):
                if self.termination.should_stop(self.evaluations, self.generations):
                    break

                # Generate neighbour via SinglePointMutation 
                neighbour = self._mutate_and_eval(current)

                L_current   = self._loss(current)
                L_neighbour = self._loss(neighbour)

                if L_neighbour < L_current:
                    current = neighbour
                else:
                    delta = L_neighbour - L_current
                    p = math.exp(-delta / max(self.b * T, 1e-300))
                    if random.random() < p:
                        current = neighbour

                # Only store unique architectures to keep Pareto computation fast.
                key = tuple(current.genotype)
                if key not in _seen:
                    _seen.add(key)
                    self._accepted.append(current)

                T = self.c * T

            self.generations += 1
            # Record every 10 generations to avoid O(n²) NDS overhead.
            if self.generations % 10 == 0 or self.termination.should_stop(
                self.evaluations, self.generations
            ):
                self._record([current])

            if self.termination.should_stop(self.evaluations, self.generations):
                break

        return compute_pareto_front(
            self._accepted,
            self.evaluator.objective_directions,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _loss(self, ind: Individual) -> float:
        """Scalar loss for SA acceptance: L = −accuracy + γ·latency."""
        acc, lat = ind.fitness
        return -acc + self.gamma * lat

    def _sample_and_eval(self) -> Individual:
        """Sample a random architecture and evaluate it."""
        geno     = self.search_space.random_individual()
        fitness  = self.evaluator.evaluate(geno)
        metadata = self.search_space.metadata_from_genotype(geno)
        self.evaluations += 1
        return Individual(geno, fitness, metadata)

    def _mutate_and_eval(self, ind: Individual) -> Individual:
        """Generate a neighbour via SinglePointMutation and evaluate it."""
        neighbour = self._mutator.mutate(ind)
        neighbour.fitness  = self.evaluator.evaluate(neighbour.genotype)
        neighbour.metadata = self.search_space.metadata_from_genotype(
            neighbour.genotype
        )
        self.evaluations += 1
        return neighbour

    def _record(self, population: list[Individual]) -> None:
        pareto = compute_pareto_front(
            self._accepted,
            self.evaluator.objective_directions,
        )
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=population,
            pareto_front=pareto,
        )
