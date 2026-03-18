"""
MO-DEHB search strategy — both NSGA-II and EpsNet variants.

MODEHBBase
  ├── MODEHBNsgaII   (intra-front ranking by crowding distance)
  └── MODEHBEpsNet   (intra-front ranking by EpsNet max-min distance)
"""
from __future__ import annotations

import random
from abc import abstractmethod

from nas_framework.evaluator import Evaluator
from nas_framework.history import History
from nas_framework.mo_utils import pareto_front as compute_pareto_front
from nas_framework.population import Individual
from nas_framework.search_space import CSVSearchSpace
from nas_framework.termination import MaxEvaluationsTermination, Termination

from nas_framework.dehb_de import (
    binomial_crossover,
    mo_de_selection,
    rand1_mutation,
)
from nas_framework.dehb_fidelity import (
    FidelitySearchSpace,
    GlobalPopulation,
    SHBracket,
)
from nas_framework.dehb_sorting import Strategy, select_top_mo


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class MODEHBBase:
    """Base MO-DEHB implementation shared by both variants.

    Parameters
    ----------
    search_space    : CSVSearchSpace with all architectures.
    evaluator       : Evaluator wrapping benchmark (dataset + device).
    pop_size        : Sub-population size at each fidelity rung.
    budget          : Total evaluation budget.
    eta             : SH halving factor (default 3).
    min_fidelity    : Fraction of CSV rows at lowest rung (default 0.1).
    max_fidelity    : Fraction at highest rung (default 1.0).
    F               : DE mutation scale factor (default 0.5).
    CR              : DE crossover rate (default 0.5).
    termination     : Custom termination criterion; overrides *budget*.
    history         : History instance for recording statistics.
    """

    def __init__(
        self,
        search_space: CSVSearchSpace,
        evaluator: Evaluator,
        pop_size: int = 20,
        budget: int = 500,
        eta: int = 3,
        min_fidelity: float = 0.1,
        max_fidelity: float = 1.0,
        F: float = 0.5,
        CR: float = 0.5,
        termination: Termination | None = None,
        history: History | None = None,
    ):
        self.search_space  = search_space
        self.evaluator     = evaluator
        self.pop_size      = pop_size
        self.eta           = eta
        self.min_fidelity  = min_fidelity
        self.max_fidelity  = max_fidelity
        self.F             = F
        self.CR            = CR
        self.termination   = termination or MaxEvaluationsTermination(budget)
        self.history       = history or History()

        self.evaluations: int = 0
        self.generations: int = 0

        self._fid_space  = FidelitySearchSpace(search_space)
        self._global_pop = GlobalPopulation()
        self._full_fidelity_inds: list[Individual] = []

    # ------------------------------------------------------------------
    # Abstract: intra-front strategy name
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def strategy(self) -> Strategy: ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[Individual]:
        """Execute MO-DEHB and return the final Pareto front."""
        while not self.termination.should_stop(self.evaluations, self.generations):
            bracket = SHBracket(
                n_configs=self.pop_size,
                eta=self.eta,
                min_fidelity=self.min_fidelity,
                max_fidelity=self.max_fidelity,
            )
            self._run_bracket(bracket)
            self.generations += 1

            all_inds = self._global_pop.all_individuals()
            pareto   = compute_pareto_front(
                self._full_fidelity_inds or all_inds,
                self.evaluator.objective_directions,
            )
            self.history.record(
                generation=self.generations,
                evaluations=self.evaluations,
                population=all_inds,
                pareto_front=pareto,
            )

            if self.termination.should_stop(self.evaluations, self.generations):
                break

        return compute_pareto_front(
            self._full_fidelity_inds or self._global_pop.all_individuals(),
            self.evaluator.objective_directions,
        )

    # ------------------------------------------------------------------
    # SH bracket execution
    # ------------------------------------------------------------------

    def _run_bracket(self, bracket: SHBracket) -> None:
        prev_population: list[Individual] = []

        for rung_idx, rung in enumerate(bracket.rungs):
            fidelity = rung.fidelity
            is_first = rung_idx == 0

            if is_first:
                sub_pop = self._sample_random(rung.n_configs, fidelity)
            else:
                parent_pool = select_top_mo(
                    prev_population,
                    rung.n_configs,
                    self.evaluator.objective_directions,
                    strategy=self.strategy,
                )
                sub_pop = self._evolve_subpop(parent_pool, fidelity)

            if not sub_pop:
                break

            self._global_pop.update(fidelity, sub_pop)

            if abs(fidelity - self.max_fidelity) < 1e-9:
                self._full_fidelity_inds.extend(sub_pop)

            prev_population = sub_pop

            if self.termination.should_stop(self.evaluations, self.generations):
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(self, genotype: list[int]) -> Individual:
        """Evaluate genotype using the existing Evaluator, increment counter."""
        fitness  = self.evaluator.evaluate(genotype)
        metadata = {}
        if hasattr(self.search_space, "metadata_from_genotype"):
            metadata = self.search_space.metadata_from_genotype(genotype)
        self.evaluations += 1
        return Individual(genotype, fitness, metadata)

    def _sample_random(self, n: int, fidelity: float) -> list[Individual]:
        result: list[Individual] = []
        for _ in range(n):
            if self.termination.should_stop(self.evaluations, self.generations):
                break
            geno = self._fid_space.sample(fidelity)
            result.append(self._evaluate(geno))
        return result

    def _evolve_subpop(
        self,
        parent_pool: list[Individual],
        fidelity: float,
    ) -> list[Individual]:
        """Apply DE (rand/1 + binomial crossover + MO-selection) to parent_pool."""
        global_inds = self._global_pop.all_individuals()
        evolved: list[Individual] = []

        for target in parent_pool:
            if self.termination.should_stop(self.evaluations, self.generations):
                break

            mutant_geno = rand1_mutation(
                parent_pool,
                self.search_space.num_edges,
                self.search_space.num_ops,
                F=self.F,
            )
            offspring = binomial_crossover(target, mutant_geno, CR=self.CR)
            offspring = self._evaluate(offspring.genotype)

            survivor = mo_de_selection(
                target=target,
                offspring=offspring,
                global_pop=global_inds + evolved,
                directions=self.evaluator.objective_directions,
            )
            evolved.append(survivor)

        return evolved


# ──────────────────────────────────────────────────────────────────────────────
# Concrete variants
# ──────────────────────────────────────────────────────────────────────────────

class MODEHBNsgaII(MODEHBBase):
    """MO-DEHB with NSGA-II crowding-distance intra-front ranking."""

    @property
    def strategy(self) -> Strategy:
        return "nsga2"


class MODEHBEpsNet(MODEHBBase):
    """MO-DEHB with EpsNet max-min-distance intra-front ranking."""

    @property
    def strategy(self) -> Strategy:
        return "epsnet"
