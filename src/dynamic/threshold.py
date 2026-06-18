from neat.reporting import BaseReporter

class DynamicThresholdReporter(BaseReporter):
    def __init__(self, 
                 target_species, 
                 adjust_rate, 
                 min_thresh, 
                 max_thresh,
                 transition_start,
                 transition_end,
                 end_species
                 ):

        self.target_species = target_species
        self.adjust_rate = adjust_rate
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.transition_start = transition_start
        self.transition_end = transition_end
        self.end_species = end_species
        self.transition_size = self.transition_end - self.transition_start
        self.species_diff = self.end_species - self.target_species

    def end_generation(self, config, population, species_set):
        current_species_count = len(species_set.species)
        
        current_threshold = config.species_set_config.compatibility_threshold
        gen = config.current_generation

        true_species_target: int = 0
        if gen < self.transition_start:
            true_species_target = self.target_species
        elif gen < self.transition_end:
            m = (gen - self.transition_start) / self.transition_size
            true_species_target = self.target_species + self.species_diff * m
        else:
            true_species_target = self.end_species

        if current_species_count > true_species_target:
            current_threshold += self.adjust_rate
        elif current_species_count < true_species_target:
            current_threshold -= self.adjust_rate

        current_threshold = max(self.min_thresh, min(self.max_thresh, current_threshold))

        config.species_set_config.compatibility_threshold = current_threshold
        
        print(f"Gen Species: {current_species_count} | Adjusted Threshold to: {current_threshold:.2f}")
