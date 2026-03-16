"""
Descriptor extractor for MTF-PDNS.

In the original paper the descriptor φ(x) is built from three
training-free proxy metrics (synflow, jacov, snip) plus a complexity
metric (FLOPs).  Because our CSV does not contain those values, we
substitute four dataset-aware columns that are available:

    φ(x) = ( edgegpu_latency,  raspi4_latency,  fpga_latency,  –accuracy )

Accuracy is negated so that every dimension is a *minimisation*
objective, which is required by the dominance check in ElitistArchive.

The extractor also normalises each dimension to [0, 1] using per-
dimension min/max statistics computed lazily over the whole search
space the first time a lookup is made.
"""
from __future__ import annotations

import math
from pathlib import Path

from nas_framework.benchmark_api import BenchmarkAPI, CSVBenchmarkAPI


class PDNSDescriptorExtractor:
    """Extracts and normalises a 4-D descriptor vector for an architecture.

    Parameters
    ----------
    benchmark:
        Must be a CSVBenchmarkAPI instance (or any subclass) so we can
        access raw latency columns and the full data table for
        normalisation statistics.
    dataset:
        Dataset name used to resolve latency column names
        (e.g. ``"cifar100"``, ``"cifar10"``, ``"ImageNet16-120"``).
    latency_devices:
        The three device names whose latencies form the first three
        descriptor dimensions.  Defaults to the three devices used in
        the paper's hardware experiments.
    """

    DEFAULT_DEVICES: tuple[str, str, str] = ("edgegpu", "raspi4", "fpga")

    def __init__(
        self,
        benchmark: CSVBenchmarkAPI,
        dataset: str = "cifar100",
        latency_devices: tuple[str, str, str] | None = None,
    ):
        if not isinstance(benchmark, CSVBenchmarkAPI):
            raise TypeError(
                "PDNSDescriptorExtractor requires a CSVBenchmarkAPI instance. "
                f"Got: {type(benchmark).__name__}"
            )
        self.benchmark = benchmark
        self.dataset = dataset
        self.devices: tuple[str, str, str] = latency_devices or self.DEFAULT_DEVICES

        # Per-dimension normalisation constants.
        self._min: tuple[float, ...] | None = None
        self._max: tuple[float, ...] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def descriptor_keys(self) -> list[str]:
        return [
            f"{self.dataset}_{self.devices[0]}_latency",
            f"{self.dataset}_{self.devices[1]}_latency",
            f"{self.dataset}_{self.devices[2]}_latency",
            f"neg_{self.dataset}_accuracy",
        ]

    def extract(self, arch: list[int]) -> tuple[float, ...]:
        """Return the *normalised* descriptor vector for *arch*. """
        raw = self._extract_raw(arch)
        return self._normalise(raw)

    def extract_raw(self, arch: list[int]) -> tuple[float, ...]:
        """Return the *unnormalised* descriptor vector (for inspection)."""
        return self._extract_raw(arch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, arch: list[int]) -> tuple[float, ...]:
        lat0 = self.benchmark.query_latency(arch, self.dataset, self.devices[0])
        lat1 = self.benchmark.query_latency(arch, self.dataset, self.devices[1])
        lat2 = self.benchmark.query_latency(arch, self.dataset, self.devices[2])
        acc  = self.benchmark.query_accuracy(arch, self.dataset)
        return (lat0, lat1, lat2, -acc)

    def _normalise(self, raw: tuple[float, ...]) -> tuple[float, ...]:
        if self._min is None or self._max is None:
            self._compute_stats()

        result: list[float] = []
        for v, lo, hi in zip(raw, self._min, self._max):  
            if not math.isfinite(v):
                result.append(v)
                continue
            denom = hi - lo
            if denom == 0.0:
                result.append(0.0)
            else:
                result.append((v - lo) / denom)
        return tuple(result)

    def _compute_stats(self) -> None:
        """Compute per-dimension min / max over the entire benchmark table."""
        rows = list(self.benchmark._by_genotype.values())

        mins: list[float] = [math.inf,  math.inf,  math.inf,  math.inf]
        maxs: list[float] = [-math.inf, -math.inf, -math.inf, -math.inf]

        acc_col  = self.benchmark._resolve_accuracy_col(rows[0], self.dataset)
        lat_cols = [
            self.benchmark._resolve_latency_col(rows[0], self.dataset, dev)
            for dev in self.devices
        ]

        for row in rows:
            values_raw: list[float] = []
            for col in lat_cols:
                values_raw.append(CSVBenchmarkAPI._as_float(row.get(col)))
            acc_val = CSVBenchmarkAPI._as_float(row.get(acc_col))
            values_raw.append(-acc_val) 

            for i, v in enumerate(values_raw):
                if math.isfinite(v):
                    if v < mins[i]:
                        mins[i] = v
                    if v > maxs[i]:
                        maxs[i] = v

        self._min = tuple(mins)
        self._max = tuple(maxs)
