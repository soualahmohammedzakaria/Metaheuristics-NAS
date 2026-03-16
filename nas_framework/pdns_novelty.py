"""
Novelty score computation for MTF-PDNS.

    η(x) = +mean_dist(φ(x), A)   if x ∈ A   (non-dominated → rewarded)
    η(x) = −mean_dist(φ(x), A)   if x ∉ A   (dominated     → penalised)

where A is the current elitist archive and distance is Euclidean over
the normalised descriptor space.
"""
from __future__ import annotations

import math

from nas_framework.population import Individual
from nas_framework.pdns_archive import ElitistArchive
from nas_framework.pdns_descriptor import PDNSDescriptorExtractor


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_distance(
    descriptor: tuple[float, ...],
    archive_descriptors: list[tuple[float, ...]],
) -> float:
    """Mean Euclidean distance from *descriptor* to all archive entries."""
    if not archive_descriptors:
        return 0.0
    total = sum(_euclidean(descriptor, d) for d in archive_descriptors)
    return total / len(archive_descriptors)


class NoveltyScorer:
    """Assigns signed novelty scores to a batch of individuals.

    Parameters
    ----------
    extractor:
        Descriptor extractor instance (PDNSDescriptorExtractor).
    archive:
        The shared elitist archive.  Must be updated *before* calling
        ``score_population`` so that membership flags are correct.
    """

    def __init__(
        self,
        extractor: PDNSDescriptorExtractor,
        archive: ElitistArchive,
    ):
        self.extractor = extractor
        self.archive = archive

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_descriptor(self, ind: Individual) -> tuple[float, ...]:
        """Compute and cache the normalised descriptor on *ind*.

        The descriptor is stored in ``ind.metadata["pdns_descriptor"]``
        so it is computed at most once per individual.
        """
        if "pdns_descriptor" not in ind.metadata:
            ind.metadata["pdns_descriptor"] = self.extractor.extract(
                ind.genotype
            )
        return ind.metadata["pdns_descriptor"]

    def score(
        self,
        ind: Individual,
        archive_descriptors: list[tuple[float, ...]],
        in_archive: bool,
    ) -> float:
        """Compute η(x) for a single individual.

        Parameters
        ----------
        ind:
            Individual to score.  Its descriptor must already be built
            (call ``build_descriptor`` first).
        archive_descriptors:
            Snapshot of archive descriptors at the moment of scoring.
        in_archive:
            Whether *ind* is currently in the archive.
        """
        desc = self.build_descriptor(ind)
        dist = _mean_distance(desc, archive_descriptors)
        return dist if in_archive else -dist

    def score_population(
        self,
        individuals: list[Individual],
    ) -> dict[int, float]:
        """Score every individual in *individuals*.

        Steps:
        1. Build descriptors for all individuals.
        2. Update the archive with each individual's descriptor.
        3. Take a snapshot of the archive.
        4. Assign signed novelty scores.

        Returns a mapping  id(ind) → novelty_score  for all individuals.
        The score is also stored in ``ind.metadata["novelty_score"]``.
        """
        # Step 1: build all descriptors first.
        for ind in individuals:
            self.build_descriptor(ind)

        # Step 2: update archive with every individual.
        archive_membership: dict[int, bool] = {}
        for ind in individuals:
            desc = ind.metadata["pdns_descriptor"]
            added = self.archive.update(ind, desc)
            archive_membership[id(ind)] = added

        # Step 3: snapshot archive descriptors (stable for scoring).
        arc_descs = self.archive.descriptors()

        # Step 4: assign signed scores.
        scores: dict[int, float] = {}
        for ind in individuals:
            s = self.score(ind, arc_descs, archive_membership[id(ind)])
            ind.metadata["novelty_score"] = s
            scores[id(ind)] = s

        return scores
