import random
import csv
from pathlib import Path
import numpy as np

NUM_OPS = 5
NUM_EDGES = 6


class SearchSpace:
    """NAS-Bench-201 architecture encoding (same as v1)."""

    def __init__(self, num_ops: int = NUM_OPS, num_edges: int = NUM_EDGES):
        self.num_ops = num_ops
        self.num_edges = num_edges

    def random_individual(self) -> list[int]:
        return [random.randint(0, self.num_ops - 1) for _ in range(self.num_edges)]


class CSVSearchSpace(SearchSpace):
    """Search space loaded from CSV with gene/op columns and arch_id metadata."""

    def __init__(self, csv_path: str):
        super().__init__(num_ops=NUM_OPS, num_edges=NUM_EDGES)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV search space file not found: {self.csv_path}")

        self._genotypes: list[list[int]] = []
        self._metadata_by_genotype: dict[tuple[int, ...], dict] = {}
        self._load_csv()

    def _load_csv(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = [f"gene_{i}" for i in range(6)]
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing required gene column: {col}")

            for row in reader:
                genes = [int(row[f"gene_{i}"]) for i in range(6)]
                key = tuple(genes)
                self._genotypes.append(genes)
                self._metadata_by_genotype[key] = {
                    "arch_id": int(row["arch_id"]) if row.get("arch_id") not in (None, "") else None,
                    "architecture_details": row.get("architecture_details", ""),
                    "ops": [row.get(f"op_{i}") for i in range(6)],
                    "genes": genes,
                }

        if not self._genotypes:
            raise ValueError("CSV search space has no architectures")

    def random_individual(self) -> list[int]:
        return random.choice(self._genotypes)[:]

    def all_genotypes(self) -> list[list[int]]:
        """Return every architecture genotype available in this finite search space."""
        return [genes[:] for genes in self._genotypes]

    def metadata_from_genotype(self, genotype: list[int]) -> dict:
        return self._metadata_by_genotype.get(tuple(genotype), {
            "arch_id": None,
            "architecture_details": "",
            "ops": [None] * self.num_edges,
            "genes": genotype[:],
        })


class NASSearchSpace:
    """NASNet-like Dvolver search space with normal/reduction cells."""

    OPS = [
        "identity",
        "avg_pool_3x3",
        "max_pool_3x3",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "sep_conv_7x7",
    ]

    def __init__(self, num_blocks_per_cell: int = 5):
        self.num_blocks_per_cell = num_blocks_per_cell
        self.num_ops = len(self.OPS)
        self.num_possible_sources = 2 + self.num_blocks_per_cell
        self.genes_per_block = 4
        self.genes_per_cell = self.num_blocks_per_cell * self.genes_per_block + self.num_possible_sources
        self.genes_per_architecture = self.genes_per_cell * 2

    def _random_block(self, block_idx: int) -> tuple[int, int, int, int]:
        max_input_idx = block_idx + 1
        i1 = random.randint(0, max_input_idx)
        o1 = random.randint(0, self.num_ops - 1)
        i2 = random.randint(0, max_input_idx)
        o2 = random.randint(0, self.num_ops - 1)
        return (i1, o1, i2, o2)

    def _random_cell(self) -> dict:
        blocks = [self._random_block(b) for b in range(self.num_blocks_per_cell)]
        mask = [random.randint(0, 1) for _ in range(self.num_possible_sources)]
        if sum(mask) == 0:
            mask[random.randint(0, self.num_possible_sources - 1)] = 1
        extra_connections = [idx for idx, bit in enumerate(mask) if bit == 1]
        return {
            "blocks": blocks,
            "extra_connections": extra_connections,
        }

    def random_architecture(self) -> dict:
        return {
            "normal_cell": self._random_cell(),
            "reduction_cell": self._random_cell(),
        }

    def _encode_cell(self, cell: dict) -> list[int]:
        genes: list[int] = []
        for block_idx, block in enumerate(cell["blocks"]):
            i1, o1, i2, o2 = block
            max_input_idx = block_idx + 1
            genes.extend([
                int(i1) % (max_input_idx + 1),
                int(o1) % self.num_ops,
                int(i2) % (max_input_idx + 1),
                int(o2) % self.num_ops,
            ])

        extras = set(int(x) for x in cell.get("extra_connections", []))
        for src in range(self.num_possible_sources):
            genes.append(1 if src in extras else 0)
        return genes

    def encode(self, architecture: dict) -> np.ndarray:
        normal = self._encode_cell(architecture["normal_cell"])
        reduction = self._encode_cell(architecture["reduction_cell"])
        vec = np.asarray(normal + reduction, dtype=np.int64)
        if vec.shape[0] != self.genes_per_architecture:
            raise ValueError("Unexpected encoded architecture length")
        return vec

    def _decode_cell(self, cell_vec: np.ndarray) -> dict:
        blocks: list[tuple[int, int, int, int]] = []
        ptr = 0
        for block_idx in range(self.num_blocks_per_cell):
            max_input_idx = block_idx + 1
            i1 = int(cell_vec[ptr]) % (max_input_idx + 1)
            o1 = int(cell_vec[ptr + 1]) % self.num_ops
            i2 = int(cell_vec[ptr + 2]) % (max_input_idx + 1)
            o2 = int(cell_vec[ptr + 3]) % self.num_ops
            blocks.append((i1, o1, i2, o2))
            ptr += 4

        extras_mask = [int(x) & 1 for x in cell_vec[ptr: ptr + self.num_possible_sources]]
        if sum(extras_mask) == 0:
            extras_mask[0] = 1
        extra_connections = [idx for idx, bit in enumerate(extras_mask) if bit == 1]
        return {
            "blocks": blocks,
            "extra_connections": extra_connections,
        }

    def decode(self, vector: np.ndarray) -> dict:
        vec = np.asarray(vector, dtype=np.int64).flatten()
        if vec.shape[0] != self.genes_per_architecture:
            raise ValueError(
                f"Expected vector of length {self.genes_per_architecture}, got {vec.shape[0]}"
            )

        split = self.genes_per_cell
        normal_vec = vec[:split]
        reduction_vec = vec[split:]
        return {
            "normal_cell": self._decode_cell(normal_vec),
            "reduction_cell": self._decode_cell(reduction_vec),
        }

    def random_individual(self) -> list[int]:
        return self.encode(self.random_architecture()).tolist()

    def random_value_for_gene(self, position: int) -> int:
        pos = int(position) % self.genes_per_architecture
        pos_in_cell = pos % self.genes_per_cell
        block_gene_count = self.num_blocks_per_cell * self.genes_per_block

        if pos_in_cell < block_gene_count:
            block_idx = pos_in_cell // self.genes_per_block
            field = pos_in_cell % self.genes_per_block
            if field in (0, 2):
                return random.randint(0, block_idx + 1)
            return random.randint(0, self.num_ops - 1)

        return random.randint(0, 1)

    def num_possible_architectures(self) -> int:
        return int(1e20)


class CSVGenotypeDvolverSearchSpace:
    """Dvolver-compatible search space over CSV NAS-Bench-201 genotype vectors."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV benchmark file not found: {self.csv_path}")

        self.num_edges = NUM_EDGES
        self.num_ops = NUM_OPS
        self._genotypes: list[list[int]] = []
        self._metadata_by_genotype: dict[tuple[int, ...], dict] = {}
        self._load_csv()

    def _load_csv(self) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = [f"gene_{i}" for i in range(self.num_edges)]
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing required gene column: {col}")

            for row in reader:
                genes = [int(row[f"gene_{i}"]) for i in range(self.num_edges)]
                key = tuple(genes)
                self._genotypes.append(genes)
                self._metadata_by_genotype[key] = {
                    "arch_id": int(row["arch_id"]) if row.get("arch_id") not in (None, "") else None,
                    "architecture_details": row.get("architecture_details", ""),
                    "ops": [row.get(f"op_{i}") for i in range(self.num_edges)],
                    "benchmark_genotype": genes[:],
                }

        if not self._genotypes:
            raise ValueError("CSV benchmark file has no genotype rows")

    def random_architecture(self) -> dict:
        genes = random.choice(self._genotypes)[:]
        meta = self._metadata_by_genotype.get(tuple(genes), {})
        return {
            "benchmark_genotype": genes,
            "arch_id": meta.get("arch_id"),
            "architecture_details": meta.get("architecture_details", ""),
        }

    def encode(self, architecture: dict) -> np.ndarray:
        genes = architecture.get("benchmark_genotype")
        if genes is None:
            raise ValueError("CSVGenotypeDvolverSearchSpace expects architecture['benchmark_genotype']")
        vec = np.asarray(genes, dtype=np.int64).flatten()
        if vec.shape[0] != self.num_edges:
            raise ValueError(f"Expected genotype length {self.num_edges}, got {vec.shape[0]}")
        return vec

    def decode(self, vector: np.ndarray) -> dict:
        vec = np.asarray(vector, dtype=np.int64).flatten()
        if vec.shape[0] != self.num_edges:
            raise ValueError(f"Expected vector length {self.num_edges}, got {vec.shape[0]}")
        genes = [int(x) % self.num_ops for x in vec.tolist()]
        meta = self._metadata_by_genotype.get(tuple(genes), {})
        return {
            "benchmark_genotype": genes,
            "arch_id": meta.get("arch_id"),
            "architecture_details": meta.get("architecture_details", ""),
        }

    def random_value_for_gene(self, position: int) -> int:
        _ = position
        return random.randint(0, self.num_ops - 1)

    def num_possible_architectures(self) -> int:
        return len(self._genotypes)

