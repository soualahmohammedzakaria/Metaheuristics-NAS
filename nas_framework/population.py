from __future__ import annotations
from nas_framework.search_space import SearchSpace
from nas_framework.evaluator import Evaluator
from nas_framework.mo_utils import pareto_front as compute_pareto_front


class Individual:
    """A single candidate solution with its fitness."""

    def __init__(self, genotype: list[int],
                 fitness: tuple[float, ...] | None = None,
                 metadata: dict | None = None):
        self.genotype = genotype
        self.fitness = fitness
        self.metadata = metadata or {}
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def __repr__(self) -> str:
        return f"Individual({self.genotype}, fitness={self.fitness})"


class Population:
    """Manages a collection of individuals."""

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 size: int = 20):
        self.search_space = search_space
        self.evaluator = evaluator
        self.size = size
        self.individuals: list[Individual] = []

    def initialize(self) -> None:
        """Create random individuals and evaluate them."""
        self.individuals = []
        for _ in range(self.size):
            geno = self.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(geno)
            self.individuals.append(Individual(geno, fit, metadata=metadata))

    def best(self) -> Individual:
        """Return an individual from the first Pareto front (highest crowding)."""
        front = self.pareto_front()
        if front:
            return max(front, key=lambda ind: ind.crowding_distance)
        return self.individuals[0]

    def pareto_front(self) -> list[Individual]:
        return compute_pareto_front(self.individuals, self.evaluator.objective_directions)

    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)


# ---------------------------------------------------------------------------
# ABC-specific extensions  (HiveNAS, Shahawy & Benkhelifa, arXiv:2211.10250v2)
# ---------------------------------------------------------------------------

def _weighted_score(fitness: tuple[float, ...], directions: tuple[int, ...]) -> float:
    """Scalar score used for greedy keep-best comparison across objectives."""
    return sum(v * d for v, d in zip(fitness, directions))


class FoodSource:
    """Wraps an Individual with an ABC abandonment trial counter.

    Each food source tracks how many consecutive evaluations have been made
    without improvement.  Once trial_count reaches the abandonment_limit the
    scout phase resets it to a new random position.
    """

    def __init__(self, individual: Individual):
        self.individual = individual
        self.trial_count: int = 0

    def reset(self, individual: Individual) -> None:
        self.individual = individual
        self.trial_count = 0

    def update(self, candidate: Individual, directions: tuple[int, ...]) -> bool:
        """Greedy selection: keep candidate if it improves the scalar score.

        Returns True if the source was updated (improvement found).
        """
        if candidate.fitness is None:
            self.trial_count += 1
            return False
        if self.individual.fitness is None:
            self.individual = candidate
            self.trial_count = 0
            return True
        if _weighted_score(candidate.fitness, directions) > _weighted_score(self.individual.fitness, directions):
            self.individual = candidate
            self.trial_count = 0
            return True
        self.trial_count += 1
        return False


class ABCPopulation(Population):
    """Population of FoodSources for the ABC algorithm.

    Parameters
    ----------
    search_space, evaluator, size : same as Population.
    abandonment_limit : int
        Consecutive failed trials before a scout resets the source.
        Defaults to 10 * num_edges when not specified.
    """

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 size: int = 10, abandonment_limit: int | None = None):
        super().__init__(search_space, evaluator, size)
        self.abandonment_limit: int = (
            abandonment_limit if abandonment_limit is not None
            else 10 * search_space.num_edges
        )
        self.food_sources: list[FoodSource] = []

    def initialize(self) -> None:
        """Scout phase: randomly sample and evaluate all food sources."""
        self.food_sources = []
        self.individuals = []
        for _ in range(self.size):
            geno = self.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(geno)
            ind = Individual(geno, fit, metadata=metadata)
            self.food_sources.append(FoodSource(ind))
            self.individuals.append(ind)

    def sync_individuals(self) -> None:
        """Keep self.individuals aligned with current food-source states."""
        self.individuals = [fs.individual for fs in self.food_sources]

    def scout_reset(self, directions: tuple[int, ...]) -> int:
        """Reset exhausted sources; return the number of resets performed."""
        resets = 0
        for fs in self.food_sources:
            if fs.trial_count >= self.abandonment_limit:
                geno = self.search_space.random_individual()
                fit = self.evaluator.evaluate(geno)
                metadata = {}
                if hasattr(self.search_space, "metadata_from_genotype"):
                    metadata = self.search_space.metadata_from_genotype(geno)
                fs.reset(Individual(geno, fit, metadata=metadata))
                resets += 1
        return resets

# ---------------------------------------------------------------------------
# PSO-specific extensions  (MOIPSO, Shao et al., Scientific Reports 2025)
# ---------------------------------------------------------------------------

class Particle:
    """Wraps an Individual with PSO velocity and personal-best memory.

    Attributes
    ----------
    individual : Individual
        Current position (genotype + fitness).
    velocity : list[float]
        Continuous velocity per gene.  Interpreted as attraction strength
        in the discrete update step.
    pbest : Individual
        The best Individual this particle has personally visited, judged
        by Pareto rank then crowding distance (updated greedily).
    """

    def __init__(self, individual: Individual):
        self.individual = individual
        n = len(individual.genotype)
        self.velocity: list[float] = [0.0] * n
        self.pbest: Individual = individual

    def update_pbest(self, directions: tuple[int, ...]) -> None:
        """Greedily update pbest if current position is better."""
        if self.individual.fitness is None:
            return
        if self.pbest.fitness is None:
            self.pbest = self.individual
            return
        # Use weighted scalar score for greedy comparison
        curr_score = _weighted_score(self.individual.fitness, directions)
        best_score = _weighted_score(self.pbest.fitness, directions)
        if curr_score > best_score:
            self.pbest = self.individual


class PSOPopulation(Population):
    """Population of Particles for the MOIPSO algorithm.

    Evaluator, SearchSpace, and History components.

    Parameters
    ----------
    search_space, evaluator, size : same as Population.
    w : float
        Inertia weight (default 0.4, from Shao et al. Table 3).
    """

    def __init__(self, search_space: SearchSpace, evaluator: Evaluator,
                 size: int = 20, w: float = 0.4):
        super().__init__(search_space, evaluator, size)
        self.w = w
        self.particles: list[Particle] = []
        self._gbest: Individual | None = None

    def initialize(self) -> None:
        """Create random particles, evaluate them, initialise velocities to 0."""
        self.particles = []
        self.individuals = []
        for _ in range(self.size):
            geno = self.search_space.random_individual()
            fit = self.evaluator.evaluate(geno)
            metadata = {}
            if hasattr(self.search_space, "metadata_from_genotype"):
                metadata = self.search_space.metadata_from_genotype(geno)
            ind = Individual(geno, fit, metadata=metadata)
            p = Particle(ind)
            self.particles.append(p)
            self.individuals.append(ind)
        self._update_gbest()

    def _update_gbest(self) -> None:
        """Set gbest to the best individual by Pareto rank then crowding."""
        from nas_framework.mo_utils import assign_rank_and_crowding, pareto_sort_key
        alive = [p.individual for p in self.particles if p.individual.fitness is not None]
        if not alive:
            return
        assign_rank_and_crowding(alive, self.evaluator.objective_directions)
        self._gbest = min(alive, key=pareto_sort_key)

    def get_gbest(self) -> Individual | None:
        return self._gbest

    def update_velocities_and_positions(self, c1: float, c2: float) -> None:
        """Discrete PSO update for all particles.

        Velocity update (Eq. 5 from Shao et al.):
            v_d(t+1) = w * v_d(t) + c1*r1*(pbest_d - x_d) + c2*r2*(gbest_d - x_d)

        The full accumulated velocity v_d drives the gene-adoption decision via a
        scaled probability: p_change = min(1, |v_d| / V_MAX), where V_MAX = num_ops - 1
        is the maximum possible integer displacement in one gene.

        This correctly propagates inertia w across iterations:
          - high w  -> large |v_d| persists across steps -> high p_change -> more exploration
          - low  w  -> velocity decays quickly -> particles stabilise around pbest/gbest
          - w = 0   -> pure greedy: adopt pbest or gbest every step with p = min(1,|c_i*r_i|/V_MAX)

        Gene selection (using total velocity):
            - decompose v_d into social and personal components weighted by magnitude
            - with p_total: change gene, split between gbest and pbest proportional
              to their component magnitudes
            - otherwise:  keep current gene
        """
        import random as _rng
        gbest = self._gbest
        if gbest is None:
            return
        V_MAX = float(self.search_space.num_ops - 1)  # max displacement = 4 for 5-op space
        for p in self.particles:
            new_geno = list(p.individual.genotype)
            for d in range(len(new_geno)):
                r1 = _rng.random()
                r2 = _rng.random()
                # Full velocity update — inertia term carries momentum from previous step
                v_inertia  = self.w * p.velocity[d]
                v_personal = c1 * r1 * (p.pbest.genotype[d] - p.individual.genotype[d])
                v_social   = c2 * r2 * (gbest.genotype[d]   - p.individual.genotype[d])
                p.velocity[d] = v_inertia + v_personal + v_social
                v = p.velocity[d]
                # Probability of changing this gene, scaled by velocity magnitude
                p_change = min(1.0, abs(v) / V_MAX) if V_MAX > 0 else 0.0
                if _rng.random() >= p_change:
                    # Keep current gene — inertia dominates
                    continue
                # Decide whether to follow gbest or pbest, weighted by component sizes
                abs_social   = abs(v_social)
                abs_personal = abs(v_personal)
                total = abs_social + abs_personal
                if total == 0:
                    # No pull from either — random resample
                    new_geno[d] = _rng.randint(0, self.search_space.num_ops - 1)
                elif _rng.random() < abs_social / total:
                    new_geno[d] = gbest.genotype[d]
                else:
                    new_geno[d] = p.pbest.genotype[d]
            p.individual = Individual(new_geno)

    def sync_individuals(self) -> None:
        """Align self.individuals with current particle positions."""
        self.individuals = [p.individual for p in self.particles]

    def update_pbests(self) -> None:
        """Update pbest for every particle."""
        for p in self.particles:
            p.update_pbest(self.evaluator.objective_directions)
        self._update_gbest()