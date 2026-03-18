from nas_framework.search_strategy import SearchStrategy
from nas_framework.ip_pso_population import PSOPopulation
from nas_framework.ip_evaluator import IPPSOEvaluator
from nas_framework.termination import Termination
from nas_framework.history import History
from typing import Optional
from nas_framework.population import Population
import random


class IPPSOSearch(SearchStrategy):
    def __init__(self, population: PSOPopulation, evaluator: IPPSOEvaluator,
                 termination: Optional[Termination] = None, history: Optional[History] = None,
                 max_generations: int = 30, **kwargs):
        self.population = population
        self.evaluator = evaluator
        self.termination = termination
        self.history = history or History()
        # Accept alternative kwarg configs for benchmark setups.
        if 'budget' in kwargs:
            self.max_generations = int(kwargs['budget'] / self.population.size)
        else:
            self.max_generations = max_generations
            
        self.generations = 0
        self.evaluations = 0

    def run(self):
        """Run IPPSO search."""
        for particle in self.population.particles:
            particle.current_fitness = self.evaluator.evaluate(particle.position)
            particle.update_personal_best()
            self.evaluations += 1

        # ADDED: update archive after Gen 0 evaluation
        self.population.update_archive()

        # ADDED: record generation 0
        front = self.population.get_pareto_front()
        self.history.record(self.generations, self.evaluations,
                            self.population.particles, front)

        while self.generations < self.max_generations:
            # Force guide selection to use archive exclusively
            guide_pool = self.population.archive
            
            for particle in self.population.particles:
                # MOPSO uniformly samples global best from Pareto Archive
                global_guide = random.choice(guide_pool).personal_best_position \
                               if guide_pool else particle.personal_best_position
                    
                particle.update_velocity_and_position(global_guide)
                particle.current_fitness = self.evaluator.evaluate(particle.position)
                particle.update_personal_best()
                self.evaluations += 1

            self.generations += 1
            self.population.update_archive()

            # ADDED: record each generation
            front = self.population.get_pareto_front()
            self.history.record(self.generations, self.evaluations,
                                self.population.particles, front)

        return self

    def best(self):
        """Decode winning particle back into its actual CNN layer sequence."""
        from nas_framework.ip_layer import decode_layer, LayerType, MAX_LENGTH

        front = self.population.archive if self.population.archive \
                else self.population.get_pareto_front()
        if not front:
            front = self.population.particles

        best_p = max(front, key=lambda x: x.personal_best_fitness[0])
        position = best_p.personal_best_position

        # decode position into human-readable architecture
        layers = []
        for slot in range(MAX_LENGTH):
            b0 = position[slot * 2]
            b1 = position[slot * 2 + 1]
            layer = decode_layer(b0, b1)
            if layer.layer_type != LayerType.DISABLED:
                layers.append(repr(layer))

        class DecodedIndividual:
            def __init__(self, p, arch):
                self.fitness = p.personal_best_fitness
                self.metadata = {
                    "arch_id": "IPPSO_ARCH",
                    "architecture": arch,
                    "num_layers": len(arch),
                }

        return DecodedIndividual(best_p, layers)

    def pareto_front(self):
        return self.population.get_pareto_front()


class IPRandomSearch(IPPSOSearch):
    def run(self):
        """Random search baseline for IP search space."""
        # Initial evaluation
        for particle in self.population.particles:
            particle.current_fitness = self.evaluator.evaluate(particle.position)
            particle.update_personal_best()
            self.evaluations += 1

        self.population.update_archive()

        # Target evaluations based on budget
        total_budget = self.max_generations * self.population.size
        
        while self.evaluations < total_budget:
            for particle in self.population.particles:
                if self.evaluations >= total_budget:
                    break
                    
                # Sample random valid position
                particle.position = particle._initialize_position()
                particle.current_fitness = self.evaluator.evaluate(particle.position)
                particle.update_personal_best()
                self.evaluations += 1

            self.generations += 1
            self.population.update_archive()
            
            # Record history
            front = self.population.get_pareto_front()
            self.history.record(self.generations, self.evaluations,
                                self.population.particles, front)

        return self