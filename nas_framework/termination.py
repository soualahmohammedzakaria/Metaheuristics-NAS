from __future__ import annotations

from abc import ABC, abstractmethod


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

