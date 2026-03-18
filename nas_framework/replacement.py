from abc import ABC, abstractmethod
from nas_framework.population import Individual
from nas_framework.mo_utils import take_pareto_best


class Replacement(ABC):
    """Abstract replacement / survivor-selection operator."""

    @abstractmethod
    def replace(self, population: list[Individual],
            offspring: list[Individual], pop_size: int,
            objective_directions: tuple[int, ...]) -> list[Individual]:
        """Merge offspring into population and return the next generation."""
        ...


class ElitistReplacement(Replacement):
    """Keep the best *pop_size* Pareto-ranked individuals from parents + offspring."""

    def replace(self, population: list[Individual],
        offspring: list[Individual], pop_size: int,
        objective_directions: tuple[int, ...]) -> list[Individual]:
        combined = population + offspring
        return take_pareto_best(combined, pop_size, objective_directions)


class GenerationalReplacement(Replacement):
    """Replace population with offspring using Pareto elitism."""

    def __init__(self, elitism: bool = True):
        self.elitism = elitism

    def replace(self, population: list[Individual],
                offspring: list[Individual], pop_size: int,
                objective_directions: tuple[int, ...]) -> list[Individual]:
        if self.elitism:
            elite = take_pareto_best(population, 1, objective_directions)
            offspring.extend(elite)
        return take_pareto_best(offspring, pop_size, objective_directions)



class RankBasedReplacement(Replacement):
    """Rank-based survivor selection from RB-IFA (Nguyen et al., ICAART 2025).

    Instead of Pareto-front construction, individuals are scored by a
    weighted rank across objectives (via mo_utils.rank_based_score) and
    the *pop_size* lowest-scoring (best) individuals survive.

    This is a drop-in replacement for ElitistReplacement when a focused
    performance-efficiency tradeoff direction is preferred over full
    Pareto coverage.

    Parameters
    ----------
    w_perf : float
        Weight given to performance objectives (direction +1).
        Cost weight is 1 - w_perf.  Default 0.5 (balanced tradeoff).
    """

    def __init__(self, w_perf: float = 0.6):
        self.w_perf = w_perf

    def replace(self, population: list, offspring: list,
                pop_size: int,
                objective_directions: tuple[int, ...]) -> list:
        from nas_framework.mo_utils import rank_based_score
        combined = population + offspring
        scores = rank_based_score(combined, objective_directions, self.w_perf)
        ranked = sorted(zip(scores, combined), key=lambda x: x[0])
        return [ind for _, ind in ranked[:pop_size]]