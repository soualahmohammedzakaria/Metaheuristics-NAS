from nas_framework.benchmark_api import BenchmarkAPI


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

