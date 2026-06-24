import math
from enum import Enum


class DecayMode(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    COSINE_WARM = "cosine_warm"


class WeightDecaySchedule:

    def __init__(
        self,
        mode: DecayMode = DecayMode.EXPONENTIAL,
        sigma_start: float = 0.5,
        sigma_min: float = 0.02,
        gen_max: int = 600,
        decay_rate: float = 5.0,
        # Cosine warm-restart params
        t0: int = 100,          # initial restart period (generations)
        t_mult: float = 2.0,    # period multiplier on each restart
    ):
        if sigma_min >= sigma_start:
            raise ValueError("sigma_min must be less than sigma_start")
        if gen_max <= 0:
            raise ValueError("gen_max must be positive")

        self.mode = DecayMode(mode)
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.gen_max = gen_max
        self.decay_rate = decay_rate
        self.t0 = t0
        self.t_mult = t_mult

    # ------------------------------------------------------------------
    def __call__(self, generation: int) -> float:
        progress = min(1.0, generation / self.gen_max)

        if self.mode == DecayMode.EXPONENTIAL:
            return self._exponential(progress)
        elif self.mode == DecayMode.LINEAR:
            return self._linear(progress)
        elif self.mode == DecayMode.COSINE_WARM:
            return self._cosine_warm(generation)
        else:
            raise ValueError(f"Unknown decay mode: {self.mode}")

    # ------------------------------------------------------------------
    def _exponential(self, progress: float) -> float:
        value = self.sigma_min + (self.sigma_start - self.sigma_min) * math.exp(
            -self.decay_rate * progress
        )
        return max(self.sigma_min, value)

    def _linear(self, progress: float) -> float:
        value = self.sigma_start - (self.sigma_start - self.sigma_min) * progress
        return max(self.sigma_min, value)

    def _cosine_warm(self, generation: int) -> float:
        """Cosine annealing with warm restarts (SGDR-style)."""
        if self.t_mult == 1.0:
            cycle = generation // self.t0
            t_cur = generation % self.t0
            t_i = self.t0
        else:
            # Determine which cycle we're in
            cycle = math.floor(
                math.log(
                    max(1, 1 + generation / self.t0 * (self.t_mult - 1)),
                ) / math.log(self.t_mult)
            )
            t_start = int(self.t0 * (self.t_mult**cycle - 1) / (self.t_mult - 1))
            t_cur = generation - t_start
            t_i = int(self.t0 * self.t_mult**cycle)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t_cur / max(1, t_i)))
        value = self.sigma_min + (self.sigma_start - self.sigma_min) * cosine_factor
        return max(self.sigma_min, value)

    # ------------------------------------------------------------------
    def apply(self, config, generation: int) -> float:
        sigma = self(generation)
        config.genome_config.weight_mutate_power = sigma
        config.genome_config.bias_mutate_power = sigma
        return sigma

    # ------------------------------------------------------------------
    def summary(self) -> str:
        checkpoints = [0, 25, 50, 75, 100]
        lines = [
            f"WeightDecaySchedule({self.mode}) "
            f"sigma_start={self.sigma_start} sigma_min={self.sigma_min}"
        ]
        for pct in checkpoints:
            gen = int(self.gen_max * pct / 100)
            lines.append(f"  gen {gen:>4} ({pct:>3}%): sigma = {self(gen):.4f}")
        return "\n".join(lines)
