"""Model parameters for the Schnakenberg reaction-diffusion system.

    du/dt = Du * Lap(u) + a - u + u^2 v
    dv/dt = Dv * Lap(v) + b - u^2 v

Homogeneous steady state:  u* = a + b,  v* = b / (a + b)^2
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SchnakenbergParams:
    a: float = 0.1
    b: float = 0.9
    Du: float = 1.0
    Dv: float = 40.0

    @property
    def u_star(self) -> float:
        return self.a + self.b

    @property
    def v_star(self) -> float:
        return self.b / (self.a + self.b) ** 2
