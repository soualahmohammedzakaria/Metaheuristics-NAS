import random
from nas_framework.ip_layer import MAX_LENGTH, POPULATION_SIZE, LayerType, random_layer, is_valid_for_slot, resample_valid_for_slot


class PSOParticle:
    def __init__(self):
        self.position = [0] * (MAX_LENGTH * 2)
        self.velocity = [0.0] * (MAX_LENGTH * 2)
        self.personal_best_position = list(self.position)
        self.personal_best_fitness = -1e10
        self.current_fitness = -1e10

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
                x_new -= 256
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
        if self.current_fitness > self.personal_best_fitness:
            self.personal_best_position = list(self.position)
            self.personal_best_fitness = self.current_fitness


class PSOPopulation:
    def __init__(self, size=POPULATION_SIZE):
        self.size = size
        self.particles = [PSOParticle() for _ in range(size)]
        for part in self.particles:
            part.init_position()
        self.global_best_position = []
        self.global_best_fitness = -1e10

    def initialize_global_best(self):
        best_particle = None
        best_fitness = -1e10
        for part in self.particles:
            if part.personal_best_fitness > best_fitness:
                best_fitness = part.personal_best_fitness
                best_particle = part
        self.global_best_position = list(best_particle.personal_best_position)
        self.global_best_fitness = best_particle.personal_best_fitness

    def update_global_best(self):
        for part in self.particles:
            if part.personal_best_fitness > self.global_best_fitness:
                self.global_best_position = list(part.personal_best_position)
                self.global_best_fitness = part.personal_best_fitness