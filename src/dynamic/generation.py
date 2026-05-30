class ParallelGenerationTracker:
    def __init__(self, parallel_evaluator, population):
        self.evaluator = parallel_evaluator
        self.population = population

    def evaluate(self, genomes, config):
        config.current_generation = self.population.generation
        
        return self.evaluator.evaluate(genomes, config)
