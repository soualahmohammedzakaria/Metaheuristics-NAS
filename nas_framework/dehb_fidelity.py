"""
Successive Halving (SH) bracket logic and fidelity simulation for MO-DEHB.

Fidelity simulation
~~~~~~~~~~~~~~~~~~~
Because our benchmark is a pre-computed CSV lookup (not real training),
there is no natural "epoch" axis.  We simulate fidelity by restricting
the *search space*: at fidelity fraction f ∈ (0, 1], only the first
⌈f × N⌉ rows of the CSV are visible to the evaluator.  Lower-fidelity
runs therefore sample from a smaller sub-space; the full space is only
accessible at maximum fidelity.

This mirrors the spirit of DEHB's multi-fidelity design:
  - cheap (low-fidelity) evaluations explore broadly,
  - expensive (high-fidelity) evaluations refine the best candidates.

SH bracket
~~~~~~~~~~
A Successive Halving bracket is parameterised by:
  - n_configs : number of configurations at the lowest fidelity rung.
  - eta        : halving rate (typically 3).
  - min_fidelity, max_fidelity : fractions of the CSV rows (0 < min ≤ max ≤ 1).

At each rung r (0-indexed from the bottom):
  - fidelity   = min_fidelity * eta^r   (capped at max_fidelity)
  - n_survive  = n_configs // eta^r
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from nas_framework.population import Individual
from nas_framework.search_space import CSVSearchSpace


# ──────────────────────────────────────────────────────────────────────────────
# Fidelity-aware search space wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FidelitySearchSpace:
    """Wraps a CSVSearchSpace and exposes only the first *fidelity* fraction."""

    def __init__(self, base: CSVSearchSpace):
        self._base = base
        self._all  = base._genotypes          # full ordered list

    def sample(self, fidelity: float) -> list[int]:
        """Random individual from the first ⌈fidelity × N⌉ architectures."""
        n = max(1, math.ceil(fidelity * len(self._all)))
        import random
        idx = random.randrange(n)
        return self._all[idx][:]

    def metadata_from_genotype(self, genotype: list[int]) -> dict:
        return self._base.metadata_from_genotype(genotype)

    @property
    def num_edges(self) -> int:
        return self._base.num_edges

    @property
    def num_ops(self) -> int:
        return self._base.num_ops


# ──────────────────────────────────────────────────────────────────────────────
# SH rung
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SHRung:
    fidelity:  float              # fraction of CSV rows
    n_configs: int                # number of configs at this rung
    population: list[Individual] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# SH bracket
# ──────────────────────────────────────────────────────────────────────────────

class SHBracket:
    """One Successive Halving bracket used inside a DEHB iteration.

    Parameters
    ----------
    n_configs     : number of configurations sampled at the lowest rung.
    eta           : halving factor.
    min_fidelity  : lowest fidelity fraction.
    max_fidelity  : highest fidelity fraction (1.0 = full CSV).
    """

    def __init__(
        self,
        n_configs: int,
        eta: int = 3,
        min_fidelity: float = 0.1,
        max_fidelity: float = 1.0,
    ):
        self.eta = eta
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity

        # Build rungs bottom-up.
        n_rungs = max(1, math.ceil(
            math.log(max_fidelity / min_fidelity, eta)
        ) + 1)

        self.rungs: list[SHRung] = []
        for r in range(n_rungs):
            fid = min(min_fidelity * (eta ** r), max_fidelity)
            n   = max(1, n_configs // (eta ** r))
            self.rungs.append(SHRung(fidelity=fid, n_configs=n))

    # lowest rung first
    @property
    def n_rungs(self) -> int:
        return len(self.rungs)

    def rung_fidelity(self, r: int) -> float:
        return self.rungs[r].fidelity

    def rung_n_configs(self, r: int) -> int:
        return self.rungs[r].n_configs

    def __repr__(self) -> str:
        desc = ", ".join(
            f"(f={rung.fidelity:.2f}, n={rung.n_configs})"
            for rung in self.rungs
        )
        return f"SHBracket([{desc}])"


# ──────────────────────────────────────────────────────────────────────────────
# Global population across fidelities (for MO-DE selection)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalPopulation:
    """Stores the most-recently-evolved sub-population for each fidelity rung.

    Used by the MO-DE selection step to assemble a global view of all
    currently known individuals across fidelities.
    """

    def __init__(self) -> None:
        self._by_fidelity: dict[float, list[Individual]] = {}

    def update(self, fidelity: float, individuals: list[Individual]) -> None:
        self._by_fidelity[fidelity] = list(individuals)

    def all_individuals(self) -> list[Individual]:
        result: list[Individual] = []
        for inds in self._by_fidelity.values():
            result.extend(inds)
        return result

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_fidelity.values())
