from neat.reporting import BaseReporter

class DynamicThresholdReporter(BaseReporter):
    def __init__(self, 
                 target_species, 
                 adjust_rate, 
                 min_thresh, 
                 max_thresh,
                 ):

        self.target_species = target_species
        self.adjust_rate = adjust_rate
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def end_generation(self, config, population, species_set):
        current_species_count = len(species_set.species)
        
        current_threshold = config.species_set_config.compatibility_threshold

        if current_species_count > self.target_species:
            current_threshold += self.adjust_rate
        elif current_species_count < self.target_species:
            current_threshold -= self.adjust_rate

        current_threshold = max(self.min_thresh, min(self.max_thresh, current_threshold))

        config.species_set_config.compatibility_threshold = current_threshold
        
        print(f"Gen Species: {current_species_count} | Adjusted Threshold to: {current_threshold:.2f}")
