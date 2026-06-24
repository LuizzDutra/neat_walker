from neat.reporting import BaseReporter
from src.dynamic.species_schedule import SpeciesSchedule
from src.dynamic.weight_decay import WeightDecaySchedule


class DynamicReporter(BaseReporter):
    def __init__(
        self,
        target_species: int,
        adjust_rate: float,
        min_thresh: float,
        max_thresh: float,
        species_schedule: SpeciesSchedule | None = None,
        weight_decay: WeightDecaySchedule | None = None,
    ):
        self.base_target = target_species
        self.adjust_rate = adjust_rate
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.species_schedule = species_schedule
        self.weight_decay = weight_decay
        self._generation = 0

    # ------------------------------------------------------------------
    def start_generation(self, generation):
        self._generation = generation

    # ------------------------------------------------------------------
    def end_generation(self, config, population, species_set):
        gen = self._generation
        target = None
        current_species_count = len(species_set.species)
        current_threshold = config.species_set_config.compatibility_threshold
        
        if self.species_schedule is not None:
            target = self.species_schedule(gen)
            if current_species_count > target:
                current_threshold += self.adjust_rate
            elif current_species_count < target:
                current_threshold -= self.adjust_rate

            current_threshold = max(
                self.min_thresh, min(self.max_thresh, current_threshold)
            )
            config.species_set_config.compatibility_threshold = current_threshold

        sigma_str = ""
        if self.weight_decay is not None:
            sigma = self.weight_decay.apply(config, gen)
            sigma_str = f" | Weight σ: {sigma:.4f}"

        print(
            f"Gen {gen:>4} | Species: {current_species_count:>3} (target {target})"
            f" | Threshold: {current_threshold:.2f}"
            f"{sigma_str}"
        )
