from pydantic import BaseModel


class SimResult(BaseModel):
    reward: float = 0.0
    canon_reward: float = 0.0
    steps: int = 0
    has_fallen: bool = False
    has_stopped: bool = False
    total_splay: float = 0.0
    total_tilt: float = 0.0
    total_knee: float = 0.0
