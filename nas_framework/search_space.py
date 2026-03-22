import random
import csv
from pathlib import Path

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

