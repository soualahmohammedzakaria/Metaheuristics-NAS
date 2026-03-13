from abc import ABC, abstractmethod
import importlib
import sys
import csv
from pathlib import Path


class BenchmarkAPI(ABC):
    """Abstract benchmark query interface."""

    @abstractmethod
    def query_accuracy(self, arch: list[int], dataset: str) -> float: ...

    @abstractmethod
    def query_latency(self, arch: list[int], dataset: str, device: str) -> float: ...

    def get_metadata(self, arch: list[int]) -> dict:
        """Optional metadata (e.g., arch_id, ops) for a genotype."""
        return {}


class NASBench201BenchmarkAPI(BenchmarkAPI):
    """Concrete benchmark combining NAS-Bench-201 + HW-NAS-Bench."""

    OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

    def __init__(self, nas201_path: str, hwnas_path: str):
        NASBench201API, HWNASBenchAPI = self._load_api_classes()

        self.nas201_api = NASBench201API(nas201_path, verbose=False)
        self.hw_api = HWNASBenchAPI(hwnas_path, search_space="nasbench201")

    @staticmethod
    def _load_api_classes():
        try:
            nas_mod = importlib.import_module("nas_201_api")
            hw_mod = importlib.import_module("hw_nas_bench_api")
            return nas_mod.NASBench201API, hw_mod.HWNASBenchAPI
        except ModuleNotFoundError:
            root = Path(__file__).resolve().parents[1]
            fallback_paths = [
                root / "HW_NAS_Test" / "HW-NAS-Bench",
                root / "third_party" / "HW-NAS-Bench",
            ]
            for p in fallback_paths:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            nas_mod = importlib.import_module("nas_201_api")
            hw_mod = importlib.import_module("hw_nas_bench_api")
            return nas_mod.NASBench201API, hw_mod.HWNASBenchAPI

    @classmethod
    def arch_to_str(cls, arch: list[int]) -> str:
        ops = [cls.OPS[i] for i in arch]
        return (
            f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+"
            f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
        )

    def _arch_index(self, arch: list[int]) -> int:
        return self.nas201_api.query_index_by_arch(self.arch_to_str(arch))

    def query_accuracy(self, arch: list[int], dataset: str = "cifar10") -> float:
        idx = self._arch_index(arch)
        info = self.nas201_api.get_more_info(idx, dataset, hp="200", is_random=False)
        return info["test-accuracy"]

    def query_latency(self, arch: list[int], dataset: str = "cifar10",
                      device: str = "edgegpu") -> float:
        idx = self._arch_index(arch)
        hw = self.hw_api.query_by_index(idx, dataset)
        return hw[f"{device}_latency"]


class CSVBenchmarkAPI(BenchmarkAPI):
    """Benchmark API backed by a merged CSV containing genes, ops, and metrics."""

    def __init__(self, csv_path: str,
                 default_accuracy_col: str = "cifar100_test_accuracy",
                 default_latency_col: str = "cifar100_edgegpu_latency"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV benchmark file not found: {self.csv_path}")

        self.default_accuracy_col = default_accuracy_col
        self.default_latency_col = default_latency_col
        self._by_genotype: dict[tuple[int, ...], dict] = {}
        self._load_csv()

    def _load_csv(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = [f"gene_{i}" for i in range(6)]
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing required gene column: {col}")

            for row in reader:
                genotype = tuple(int(row[f"gene_{i}"]) for i in range(6))
                self._by_genotype[genotype] = row

        if not self._by_genotype:
            raise ValueError("CSV benchmark file has no rows")

    @staticmethod
    def _as_float(value: str | None) -> float:
        if value is None or value == "":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    def _row_for(self, arch: list[int]) -> dict:
        key = tuple(int(x) for x in arch)
        if key not in self._by_genotype:
            raise KeyError(f"Genotype not found in CSV benchmark: {key}")
        return self._by_genotype[key]

    @staticmethod
    def _normalize_dataset(dataset: str) -> list[str]:
        # Support both ImageNet16-120 and ImageNet16_120 naming styles.
        variants = {dataset, dataset.replace("-", "_")}
        return list(variants)

    def _resolve_accuracy_col(self, row: dict, dataset: str) -> str:
        for ds in self._normalize_dataset(dataset):
            col = f"{ds}_test_accuracy"
            if col in row:
                return col
        return self.default_accuracy_col

    def _resolve_latency_col(self, row: dict, dataset: str, device: str) -> str:
        for ds in self._normalize_dataset(dataset):
            col = f"{ds}_{device}_latency"
            if col in row:
                return col
        return self.default_latency_col

    def query_accuracy(self, arch: list[int], dataset: str = "cifar100") -> float:
        row = self._row_for(arch)
        col = self._resolve_accuracy_col(row, dataset)
        return self._as_float(row.get(col))

    def query_latency(self, arch: list[int], dataset: str = "cifar100",
                      device: str = "edgegpu") -> float:
        row = self._row_for(arch)
        col = self._resolve_latency_col(row, dataset, device)
        return self._as_float(row.get(col))

    def get_metadata(self, arch: list[int]) -> dict:
        row = self._row_for(arch)
        return {
            "arch_id": int(row["arch_id"]) if row.get("arch_id") not in (None, "") else None,
            "architecture_details": row.get("architecture_details", ""),
            "ops": [row.get(f"op_{i}") for i in range(6)],
            "genes": [int(row[f"gene_{i}"]) for i in range(6)],
        }

