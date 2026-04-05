from __future__ import annotations

from abc import ABC, abstractmethod
from nas_framework.history import History


class Termination(ABC):
    """Termination policy for search strategies."""

    @abstractmethod
    def should_stop(self, evaluations: int, generations: int) -> bool:
        ...


class MaxEvaluationsTermination(Termination):
    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations

    def should_stop(self, evaluations: int, generations: int) -> bool:
        return evaluations >= self.max_evaluations


class MaxGenerationsTermination(Termination):
    def __init__(self, max_generations: int):
        self.max_generations = max_generations

    def should_stop(self, evaluations: int, generations: int) -> bool:
        return generations >= self.max_generations


class CompositeTermination(Termination):
    """Stops when any of the supplied criteria is met."""

    def __init__(self, terms: list[Termination]):
        self.terms = terms

    def should_stop(self, evaluations: int, generations: int) -> bool:
        return any(t.should_stop(evaluations, generations) for t in self.terms)


class TerminationCriteria:
    """Dvolver stopping criteria: generations/evaluations/hypervolume convergence."""

    def __init__(self,
                 max_generations: int | None = None,
                 max_evaluations: int | None = None,
                 hypervolume_patience: int = 10):
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations
        self.hypervolume_patience = hypervolume_patience

    def should_terminate(self, history: History, generation: int) -> bool:
        if self.max_generations is not None and generation >= self.max_generations:
            return True

        if self.max_evaluations is not None and history.evaluation_count >= self.max_evaluations:
            return True

        if history.has_converged(self.hypervolume_patience):
            return True

        return False

