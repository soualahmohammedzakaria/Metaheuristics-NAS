"""
MTF-PDNS search strategy.
"""
from __future__ import annotations

import random

from nas_framework.evaluator import Evaluator
from nas_framework.history import History
from nas_framework.population import Individual, Population
from nas_framework.replacement import Replacement
from nas_framework.search_strategy import SearchStrategy
from nas_framework.selection import Selection
from nas_framework.termination import MaxEvaluationsTermination, Termination
from nas_framework.variation import Variation

from nas_framework.pdns_archive import ElitistArchive
from nas_framework.pdns_descriptor import PDNSDescriptorExtractor
from nas_framework.pdns_novelty import NoveltyScorer


class MTF_PDNS(SearchStrategy):
    """Pareto Dominance-based Novelty Search with multiple descriptor metrics.

    Parameters
    ----------
    population:
        Configured Population object (search space + evaluator + size).
    selection:
        Parent-selection operator (used only to pick parents for
        crossover/mutation; novelty drives *survivor* selection).
    variation:
        Crossover + mutation variation operator.
    replacement:
        Not used for survivor selection in PDNS — survivors are chosen
        by top-k novelty score.  This parameter is kept for API
        compatibility and is ignored internally.
    evaluator:
        Evaluator that queries accuracy + latency from the benchmark.
    descriptor_extractor:
        Pre-configured PDNSDescriptorExtractor tied to the same
        CSVBenchmarkAPI as the evaluator.
    budget:
        Maximum number of architecture evaluations.
    termination:
        Optional custom termination criterion; overrides *budget*.
    history:
        Optional History instance for recording run statistics.
    """

    def __init__(
        self,
        population: Population,
        selection: Selection,
        variation: Variation,
        replacement: Replacement,
        evaluator: Evaluator,
        descriptor_extractor: PDNSDescriptorExtractor,
        budget: int = 500,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        super().__init__(
            population=population,
            selection=selection,
            variation=variation,
            replacement=replacement,
            evaluator=evaluator,
            termination=termination or MaxEvaluationsTermination(budget),
            history=history or History(),
            budget=budget,
        )
        self.archive = ElitistArchive()
        self.scorer  = NoveltyScorer(descriptor_extractor, self.archive)

    # ------------------------------------------------------------------
    # Main loop 
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Execute MTF-PDNS and return the final population.

        The elitist archive is available via ``self.archive`` after the
        run completes and represents the true result of the search.
        """
        # ── Initialisation ──────────────────────────────────────────────
        self.population.initialize()
        self.evaluations = len(self.population)
        self.generations  = 0

        # Evaluate descriptors + update archive for initial population.
        self.scorer.score_population(self.population.individuals)
        self._record_history()

        # ── Main loop ───────────────────────────────────────────────────
        while not self.termination.should_stop(self.evaluations, self.generations):

            # Step 1: select parents (tournament on Pareto rank / crowding).
            parents = self.selection.select(
                self.population.individuals,
                self.population.size,
                self.evaluator.objective_directions,
            )

            # Step 2: crossover + mutation → offspring.
            offspring = self.variation.generate(parents, self.population.size)

            # Step 3: evaluate offspring fitness (accuracy + latency).
            self._evaluate_offspring(offspring)
            if not offspring:
                break

            # Step 4: update archive and compute novelty for P ∪ O.
            combined = self.population.individuals + offspring
            self.scorer.score_population(combined)

            # Step 5: survivor selection — keep top-N by novelty score.
            self.population.individuals = _select_by_novelty(
                combined, self.population.size
            )

            self.generations += 1
            self._record_history()

        return self.population

    # ------------------------------------------------------------------
    # History helper override
    # ------------------------------------------------------------------

    def _record_history(self) -> None:
        """Record a history entry using the archive as the Pareto front."""
        self.history.record(
            generation=self.generations,
            evaluations=self.evaluations,
            population=self.population.individuals,
            pareto_front=self.archive.individuals(),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _select_by_novelty(
    individuals: list[Individual],
    n: int,
) -> list[Individual]:
    """Return the *n* individuals with the highest novelty score.

    Ties are broken randomly to preserve diversity.
    """
    scored = [
        (ind.metadata.get("novelty_score", 0.0), random.random(), ind)
        for ind in individuals
    ]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [ind for _, _, ind in scored[:n]]
