from pydantic import BaseModel


class SimResult(BaseModel):
    reward: float
    steps: int
    has_fallen: bool
    has_stopped: bool
