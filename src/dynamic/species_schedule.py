import math
from enum import Enum


class ScheduleMode(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    STAGED = "staged"


# Staged schedule: list of (progress_threshold, fraction_of_s_start)
_DEFAULT_STAGES = [
    (0.00, 1.00),
    (0.33, 0.70),
    (0.66, 0.45),
]


class SpeciesSchedule:
    """Callable that returns the target species count for a given generation."""

    def __init__(
        self,
        mode: ScheduleMode = ScheduleMode.COSINE,
        s_start: int = 20,
        s_min: int = 4,
        gen_max: int = 600,
        stages: list[tuple[float, float]] | None = None,
    ):
        if s_min >= s_start:
            raise ValueError("s_min must be less than s_start")
        if gen_max <= 0:
            raise ValueError("gen_max must be positive")

        self.mode = ScheduleMode(mode)
        self.s_start = s_start
        self.s_min = s_min
        self.gen_max = gen_max
        self.stages = stages or _DEFAULT_STAGES

    # ------------------------------------------------------------------
    def __call__(self, generation: int) -> int:
        progress = min(1.0, generation / self.gen_max)

        if self.mode == ScheduleMode.COSINE:
            return self._cosine(progress)
        elif self.mode == ScheduleMode.LINEAR:
            return self._linear(progress)
        elif self.mode == ScheduleMode.STAGED:
            return self._staged(progress)
        else:
            raise ValueError(f"Unknown schedule mode: {self.mode}")

    # ------------------------------------------------------------------
    def _cosine(self, progress: float) -> int:
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        value = self.s_min + (self.s_start - self.s_min) * factor
        return max(self.s_min, round(value))

    def _linear(self, progress: float) -> int:
        value = self.s_start - (self.s_start - self.s_min) * progress
        return max(self.s_min, round(value))

    def _staged(self, progress: float) -> int:
        fraction = 1.0
        for threshold, frac in self.stages:
            if progress >= threshold:
                fraction = frac
        value = self.s_start * fraction
        return max(self.s_min, round(value))

    # ------------------------------------------------------------------
    def summary(self) -> str:
        checkpoints = [0, 25, 50, 75, 100]
        lines = [f"SpeciesSchedule({self.mode}) s_start={self.s_start} s_min={self.s_min}"]
        for pct in checkpoints:
            gen = int(self.gen_max * pct / 100)
            lines.append(f"  gen {gen:>4} ({pct:>3}%): target = {self(gen)}")
        return "\n".join(lines)
