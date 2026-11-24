import numpy as np
import math

class CarParams:

    def __init__(self, wheel_radius, gear_ratio, track_width, loss_coeff, T_min, T_max, Ploss_scale, mass, yaw_inertia):

        self.wheel_radius = wheel_radius
        self.gear_ratio = gear_ratio
        self.track_width = track_width
        self.loss_coeff = loss_coeff
        self.T_min = T_min
        self.T_max = T_max
        self.Ploss_scale = Ploss_scale
        self.mass = mass
        self.yaw_inertia = yaw_inertia
    
class Reference:

    def __init__(self, Fx: float, Mz: float):
        self.Fx = Fx    # N
        self.Mz = Mz    # N.m

class WeelTork:
    def __init__(self, Fl: float, Fr: float, Rl: float, Rr: float):

        self.Fl = Fl
        self.Fr = Fr
        self.Rl = Rl
        self.Rr = Rr

    def __str__(self):
        return F"Fl: {self.Fl:.2f} Fr: {self.Fr:.2f} Rl: {self.Rl:.2f} Rr: {self.Rr:.2f}"

    def to_array(self):
        return np.array(
            [self.Fl, self.Fr, self.Rl, self.Rr],
            dtype=float
        )

class Vehicle_state:
    """
    Representa o estado dinâmico e a posição de um veículo no plano XY.

    Parameters
    ----------
    X : float
        Posição longitudinal no sistema de coordenadas global [m].

    Y : float
        Posição lateral no sistema de coordenadas global [m].

    psi : float
        Orientação (ângulo de guinada) do veículo em relação ao eixo X global [rad].

    vx : float
        Velocidade longitudinal do veículo no sistema de coordenadas local [m/s].

    omega : float
        Velocidade angular de guinada do veículo [rad/s].
    """

    def __init__(self, X, Y, psi, vx, omega):

        self.X = X
        self.Y = Y
        self.psi = psi
        self.vx = vx
        self.omega = omega


    def _get_angle_arrow(self, angle_rad: float) -> str:
        arrows = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]

        # Normaliza o ângulo para [0, 2π)
        angle_rad %= 2 * math.pi

        # Cada direção ocupa π/4 radianos
        index = round(angle_rad / (math.pi / 4)) % 8

        return arrows[index]
    
    def __str__(self):
        return f"P: ({self.X:.2f}m, {self.Y:.2f}m) V: {self.vx:.2f}m/s {self._get_angle_arrow(self.psi)} w: {self.omega}rad"