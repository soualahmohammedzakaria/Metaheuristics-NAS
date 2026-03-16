from nas_framework.search_strategy import SearchStrategy
from nas_framework.ip_pso_population import PSOPopulation
from nas_framework.ip_evaluator import IPPSOEvaluator
from nas_framework.termination import Termination
from nas_framework.history import History
from typing import Optional


class IPPSOSearch(SearchStrategy):
    def __init__(self, population: PSOPopulation, evaluator: IPPSOEvaluator,
                 termination: Optional[Termination] = None, history: Optional[History] = None,
                 max_generations: int = 30):
        # Note: population is PSOPopulation, not the standard Population
        self.population = population
        self.evaluator = evaluator
        self.termination = termination
        self.history = history or History()
        self.max_generations = max_generations
        self.generations = 0
        self.evaluations = 0

    def run(self):
        """Run IPPSO search."""
        # Initialize fitness
        for particle in self.population.particles:
            particle.current_fitness = self.evaluator.evaluate(particle.position)
            particle.update_personal_best()
            self.evaluations += 1

        self.population.initialize_global_best()

        while self.generations < self.max_generations:
            for particle in self.population.particles:
                particle.update_velocity_and_position(self.population.global_best_position)
                particle.current_fitness = self.evaluator.evaluate(particle.position)
                particle.update_personal_best()
                self.evaluations += 1

            self.population.update_global_best()
            self.generations += 1

            # Record history if needed
            # For simplicity, skip

        # Step 5: Decode, retrain, evaluate on test set
        print("Final Generation complete. Retraining best architecture on full dataset...")
        test_accuracy = self.evaluator.retrain_and_evaluate_testset(self.population.global_best_position, epochs=10)
        print(f"Final Test Accuracy: {test_accuracy}")
        
        return self.population

    def get_best_architecture(self):
        """Get the best position."""
        return self.population.global_best_position