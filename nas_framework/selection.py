from abc import ABC, abstractmethod
import random
from nas_framework.population import Individual
from nas_framework.mo_utils import assign_rank_and_crowding, pareto_sort_key


class Selection(ABC):
    """Abstract selection operator."""

    @abstractmethod
    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        """Select *n* parents from the population."""
        ...


class TournamentSelection(Selection):
    """Tournament selection with configurable tournament size."""

    def __init__(self, k: int = 3):
        self.k = k

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        assign_rank_and_crowding(individuals, objective_directions)
        parents = []
        for _ in range(n):
            candidates = random.sample(individuals, min(self.k, len(individuals)))
            winner = min(candidates, key=pareto_sort_key)
            parents.append(winner)
        return parents


class RouletteWheelSelection(Selection):
    """Fitness-proportionate (roulette wheel) selection for the ABC onlooker phase.

    HiveNAS (Shahawy & Benkhelifa, arXiv:2211.10250v2, §3.1.3) uses the
    classical ABC fitness function to convert raw objective scores into
    probabilities:

        fit_m(x) = 1 / (1 + f_m(x))   if f_m >= 0      (Eq. 3)
                 = 1 + |f_m(x)|        if f_m <  0

        p_m = fit_m / sum(fit_j)                         (Eq. 4)

    For multi-objective problems we derive a scalar score via a weighted sum
    (value * direction) before applying the ABC fitness transform.  This
    preserves the original probabilistic selection behaviour while staying
    compatible with the rest of the framework.

    Falls back to uniform random selection when all scores are equal or zero.
    """

    @staticmethod
    def _abc_fitness(score: float) -> float:
        """ABC fitness transform (Eq. 3)."""
        if score >= 0:
            return 1.0 / (1.0 + score)
        return 1.0 + abs(score)

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        eligible = [ind for ind in individuals if ind.fitness is not None]
        if not eligible:
            return random.sample(individuals, min(n, len(individuals)))

        # Scalar score per individual (weighted sum across objectives).
        raw_scores = [
            sum(v * d for v, d in zip(ind.fitness, objective_directions))
            for ind in eligible
        ]

        # Apply ABC fitness transform (Eq. 3).
        fit_scores = [self._abc_fitness(s) for s in raw_scores]
        total = sum(fit_scores)

        if total == 0:
            return random.choices(eligible, k=n)

        # Build cumulative probability wheel and spin it n times (Eq. 4).
        probs = [f / total for f in fit_scores]
        return random.choices(eligible, weights=probs, k=n)


class NeighborSelection(Selection):
    """Neighbor strategy: restrict crossover partner to objective-space neighbors.

    For each parent, finds individuals within ±k/P of its objective values,
    then picks one uniformly at random. Falls back to the full pool if the
    neighborhood is empty (edge case for very small populations).

    Eq.  N(X) = V_X^l ± (k/P) * (V_max^l - V_min^l)
    """

    def __init__(self, k: int = 3, P: int = 13):
        self.k = k   # neighbor range coefficient
        self.P = P   # number of reference point segments (same as NSGA-III's H)

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        import math
        parents = []
        fitnesses = [ind.fitness for ind in individuals if ind.fitness is not None]
        if not fitnesses:
            return random.sample(individuals, min(n, len(individuals)))

        n_obj = len(fitnesses[0])
        # Compute per-objective min/max for normalizing the neighborhood radius
        obj_min = [min(f[o] for f in fitnesses) for o in range(n_obj)]
        obj_max = [max(f[o] for f in fitnesses) for o in range(n_obj)]

        for _ in range(n):
            anchor = random.choice(individuals)
            if anchor.fitness is None:
                parents.append(anchor)
                continue

            neighbors = []
            for ind in individuals:
                if ind is anchor or ind.fitness is None:
                    continue
                # Check if ind is within the neighborhood of anchor on ALL objectives
                in_neighborhood = True
                for o in range(n_obj):
                    rng = obj_max[o] - obj_min[o]
                    if rng == 0:
                        continue
                    radius = (self.k / self.P) * rng
                    if abs(ind.fitness[o] - anchor.fitness[o]) > radius:
                        in_neighborhood = False
                        break
                if in_neighborhood:
                    neighbors.append(ind)

            parents.append(random.choice(neighbors) if neighbors else random.choice(individuals))
        return parents


class GuidanceSelection(Selection):
    """Guidance strategy: use per-objective best individuals as crossover targets.

    With probability P_guide, selects
    the best individual on a randomly chosen objective as the crossover partner.
    Inspired by PSO's global-best mechanism.
    """

    def select(self, individuals: list[Individual], n: int,
               objective_directions: tuple[int, ...]) -> list[Individual]:
        fitnesses = [ind.fitness for ind in individuals if ind.fitness is not None]
        if not fitnesses:
            return random.sample(individuals, min(n, len(individuals)))

        n_obj = len(fitnesses[0])
        parents = []
        for _ in range(n):
            # Pick a random objective, find the best individual on it
            obj = random.randint(0, n_obj - 1)
            direction = objective_directions[obj] if obj < len(objective_directions) else 1
            best = max(
                (ind for ind in individuals if ind.fitness is not None),
                key=lambda ind: ind.fitness[obj] * direction
            )
            parents.append(best)
        return parents