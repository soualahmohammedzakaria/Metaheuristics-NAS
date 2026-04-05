from nas_framework.benchmark_api import BenchmarkAPI


class DvolverEvaluator:
    """Dvolver evaluator with objectives (accuracy, speed=2e9/FLOPs), both maximized."""

    OPS = [
        "identity",
        "avg_pool_3x3",
        "max_pool_3x3",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "sep_conv_7x7",
    ]

    def __init__(self,
                 benchmark: BenchmarkAPI | None = None,
                 dataset: str = "cifar10",
                 device: str = "edgegpu",
                 num_cells_N: int = 2,
                 num_filters_F: int = 32):
        self.benchmark = benchmark
        self.dataset = dataset
        self.device = device
        self.num_cells_N = num_cells_N
        self.num_filters_F = num_filters_F
        self.objective_directions: tuple[int, int] = (1, 1)
        self.evaluations: int = 0

    def query_benchmark(self, architecture_id) -> dict:
        if self.benchmark is None:
            return {}

        if isinstance(architecture_id, (list, tuple)):
            arch = list(architecture_id)
        else:
            raise KeyError("Benchmark query expects an architecture genotype sequence")

        acc = float(self.benchmark.query_accuracy(arch, self.dataset))
        latency = float(self.benchmark.query_latency(arch, self.dataset, self.device))
        flops_proxy = max(1.0, latency * 1e7)
        return {
            "val_accuracy": acc,
            "flops": flops_proxy,
            "latency": latency,
        }

    def _count_ops(self, architecture: dict, op_name: str) -> int:
        count = 0
        for cell_key in ("normal_cell", "reduction_cell"):
            cell = architecture.get(cell_key, {})
            for block in cell.get("blocks", []):
                _, o1, _, o2 = block
                if self.OPS[o1] == op_name:
                    count += 1
                if self.OPS[o2] == op_name:
                    count += 1
        return count

    def compute_flops(self, architecture: dict, N: int, F: int) -> int:
        sep3 = self._count_ops(architecture, "sep_conv_3x3")
        sep5 = self._count_ops(architecture, "sep_conv_5x5")
        sep7 = self._count_ops(architecture, "sep_conv_7x7")
        pools = self._count_ops(architecture, "avg_pool_3x3") + self._count_ops(architecture, "max_pool_3x3")
        ids = self._count_ops(architecture, "identity")

        base = float((F ** 2) * (N * 6 + 2) * 1024)
        op_cost = (
            sep3 * 9.0 +
            sep5 * 25.0 +
            sep7 * 49.0 +
            pools * 3.0 +
            ids * 1.0
        )
        extras = 0
        for cell_key in ("normal_cell", "reduction_cell"):
            extras += len(architecture.get(cell_key, {}).get("extra_connections", []))

        flops = base * max(1.0, 0.15 * op_cost + 0.03 * extras)
        return int(max(1.0, flops))

    def _proxy_accuracy(self, architecture: dict) -> float:
        sep_total = (
            self._count_ops(architecture, "sep_conv_3x3")
            + self._count_ops(architecture, "sep_conv_5x5")
            + self._count_ops(architecture, "sep_conv_7x7")
        )
        pools = self._count_ops(architecture, "avg_pool_3x3") + self._count_ops(architecture, "max_pool_3x3")
        skips = 0
        for cell_key in ("normal_cell", "reduction_cell"):
            skips += len(architecture.get(cell_key, {}).get("extra_connections", []))

        score = 0.74 + 0.015 * sep_total + 0.004 * skips - 0.006 * pools
        return max(0.0, min(1.0, score))

    def evaluate(self, architecture: dict | list[int]) -> tuple[float, float]:
        self.evaluations += 1

        if self.benchmark is not None:
            benchmark_genotype = None
            if isinstance(architecture, dict):
                benchmark_genotype = architecture.get("benchmark_genotype")
                if benchmark_genotype is None and architecture.get("benchmark_id") is not None:
                    benchmark_genotype = architecture.get("benchmark_id")
            elif isinstance(architecture, (list, tuple)):
                benchmark_genotype = list(architecture)

            if benchmark_genotype is not None:
                try:
                    row = self.query_benchmark(benchmark_genotype)
                    acc = float(row.get("val_accuracy", 0.0))
                    flops = float(row.get("flops", 1.0))
                    return acc, float(2e9 / max(1.0, flops))
                except KeyError:
                    return 0.0, 0.0

        if not isinstance(architecture, dict):
            return 0.0, 0.0

        acc = self._proxy_accuracy(architecture)
        flops = self.compute_flops(architecture, self.num_cells_N, self.num_filters_F)
        speed = float(2e9 / max(1.0, flops))
        return acc, speed


class Evaluator:
    """Evaluates an architecture on both objectives."""

    def __init__(self, benchmark: BenchmarkAPI, dataset: str = "cifar10",
                 device: str = "edgegpu"):
        self.benchmark = benchmark
        self.dataset = dataset
        self.device = device
        # +1: maximize, -1: minimize
        self.objective_directions: tuple[int, int] = (1, -1)

    def evaluate(self, arch: list[int]) -> tuple[float, float]:
        acc = self.benchmark.query_accuracy(arch, self.dataset)
        lat = self.benchmark.query_latency(arch, self.dataset, self.device)
        return acc, lat

