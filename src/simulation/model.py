from pydantic import BaseModel


class SimResult(BaseModel):
    reward: float = 0.0
    canon_reward: float = 0.0
    steps: int = 0
    has_fallen: bool = False
    has_stopped: bool = False
