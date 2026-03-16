import random
from nas_framework.ip_layer import MAX_LENGTH, POPULATION_SIZE, LayerType, random_layer, is_valid_for_slot, resample_valid_for_slot
from nas_framework.mo_utils import dominates, pareto_front, assign_rank_and_crowding


class PSOParticle:
    def __init__(self):
        self.position = [0] * (MAX_LENGTH * 2)
        self.velocity = [0.0] * (MAX_LENGTH * 2)
        self.personal_best_position = list(self.position)
        # Using Individual-style tuples: acc, lat
        self.personal_best_fitness = (-float('inf'), float('inf'))
        self.current_fitness = (-float('inf'), float('inf'))
        
        # MO fields
        self.rank = 0
        self.crowding_distance = 0.0

    @property
    def fitness(self):
        # Alias for mo_utils compatibility
        return self.current_fitness

    def _initialize_position(self):
        pos = []
        for slot in range(MAX_LENGTH):
            b0, b1 = resample_valid_for_slot(slot, pos)
            pos.extend([b0, b1])
        return pos

    def init_position(self):
        self.position = self._initialize_position()
        self.personal_best_position = list(self.position)

    def update_velocity_and_position(self, global_best_position):
        w = 0.7298
        c1 = [1.49618, 1.49618]
        c2 = [1.49618, 1.49618]
        v_max = [4.0, 25.6]

        for i in range(len(self.position)):
            x = self.position[i]
            v = self.velocity[i]
            pb = self.personal_best_position[i]
            gb = global_best_position[i]
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            v_new = w * v + c1[i % 2] * r1 * (pb - x) + c2[i % 2] * r2 * (gb - x)

            v_new = max(-v_max[i % 2], min(v_max[i % 2], v_new))

            x_new = x + v_new

            if x_new > 255:
                x_new -= 255
            elif x_new < 0:
                x_new += 256

            self.position[i] = round(x_new)
            self.velocity[i] = v_new

        for slot in range(MAX_LENGTH):
            idx = slot * 2
            b0 = self.position[idx]
            if not is_valid_for_slot(slot, b0, self.position):
                b0, b1 = resample_valid_for_slot(slot, self.position)
                self.position[idx] = b0
                self.position[idx + 1] = b1

    def update_personal_best(self):
        # We need to maximize accuracy [0], but minimize latency [1]
        # Our mo_utils dominate expects tuple[int, ...] directions. (1, -1)
        # Since dominates operates on individuals, we can do a direct check:
        # no_worse = (acc >= pb_acc) and (lat <= pb_lat) -> since lat has -1 direction in mo_utils, we'll implement it manually
        
        c_acc, c_lat = self.current_fitness
        pb_acc, pb_lat = self.personal_best_fitness
        
        # Directions: Maximize Acc (+1), Minimize Lat (-1)
        v_current = (c_acc, -c_lat)
        v_best = (pb_acc, -pb_lat)
        
        # Dominance: 
        # AT LEAST equal in all objectives AND strictly better in at least ONE objective.
        no_worse = all(x >= y for x, y in zip(v_current, v_best))
        strictly_better = any(x > y for x, y in zip(v_current, v_best))
        
        if no_worse and strictly_better:
            self.personal_best_position = list(self.position)
            self.personal_best_fitness = self.current_fitness


class PSOPopulation:
    def __init__(self, size=POPULATION_SIZE, archive_size=50):
        self.size = size
        self.archive_size = archive_size
        self.archive: list[PSOParticle] = []
        self.particles = [PSOParticle() for _ in range(size)]
        for part in self.particles:
            part.init_position()
        self.directions = (1, -1)  # max acc, min lat

    def get_pareto_front(self):
        return pareto_front(self.particles, self.directions)
        
    def update_archive(self):
        """Merge personal bests into archive, keep non-dominated set."""
        candidates = list(self.archive) + list(self.particles)
        new_archive = pareto_front(candidates, self.directions)
        # cap by crowding distance if over limit
        if len(new_archive) > self.archive_size:
            assign_rank_and_crowding(new_archive, self.directions)
            new_archive.sort(key=lambda p: p.crowding_distance, reverse=True)
            new_archive = new_archive[:self.archive_size]
        self.archive = new_archive

    def initialize_global_best(self):
        # Not statically tracking one single absolute best
        pass

    def update_global_best(self):
        pass