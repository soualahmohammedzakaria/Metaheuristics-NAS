"""
Elitist archive for MTF-PDNS.

The archive stores non-dominated architectures judged by their
*descriptor vectors*  (latency_edgegpu, latency_raspi4, latency_fpga,
accuracy).  All four values are treated as minimization objectives
internally — accuracy is stored **negated** so that higher accuracy
corresponds to a smaller (better) value.
"""
from __future__ import annotations

from copy import deepcopy

from nas_framework.population import Individual


def _dominates_descriptor(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Return True if descriptor *a* Pareto-dominates *b* (all-minimise)."""
    no_worse      = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y  for x, y in zip(a, b))
    return no_worse and strictly_better


def _is_valid(descriptor: tuple[float, ...]) -> bool:
    """Return False if any dimension is NaN / Inf (missing data)."""
    import math
    return all(math.isfinite(v) for v in descriptor)


class ElitistArchive:
    """Maintains the non-dominated set of descriptor vectors seen so far.
    """

    def __init__(self, descriptor_keys: list[str] | None = None):
        self.descriptor_keys: list[str] = descriptor_keys or [
            "edgegpu_latency",
            "raspi4_latency",
            "fpga_latency",
            "neg_accuracy",   # stored as –accuracy so all dims minimised
        ]
        self._entries: list[tuple[tuple[float, ...], Individual]] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, ind: Individual, descriptor: tuple[float, ...]) -> bool:
        """Attempt to add *ind* to the archive.

        Returns True  if *ind* was accepted (non-dominated).
        Returns False if *ind* was dominated by an existing member.
        Dominated members that the new entry beats are removed.
        """
        if not _is_valid(descriptor):
            return False

        # Reject if dominated by any existing archive member.
        for arc_desc, _ in self._entries:
            if _dominates_descriptor(arc_desc, descriptor):
                return False

        # Remove existing members that the new entry dominates.
        self._entries = [
            (d, a) for d, a in self._entries
            if not _dominates_descriptor(descriptor, d)
        ]

        snapshot = Individual(
            ind.genotype[:],
            ind.fitness,
            deepcopy(ind.metadata),
        )
        self._entries.append((descriptor, snapshot))
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def descriptors(self) -> list[tuple[float, ...]]:
        """All descriptor vectors currently in the archive."""
        return [d for d, _ in self._entries]

    def individuals(self) -> list[Individual]:
        """All Individual snapshots currently in the archive."""
        return [a for _, a in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        return f"ElitistArchive(size={len(self)})"
