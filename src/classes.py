class CarParams:

    """
    Classe que armazena os parâmetros físicos e mecânicos do veículo
    utilizados em seu modelo dinâmico.

    Parameters
    ----------
    wheel_radius : float
        Raio efetivo das rodas [m]. Representa a distância entre o
        centro da roda e o ponto de contato com o solo.

    gear_ratio : float
        Relação de transmissão [-]. Representa a razão entre a
        velocidade angular do motor e a velocidade angular das rodas,
        influenciando o torque transmitido às rodas.

    track_width : float
        Bitola do veículo [m]. Distância lateral entre os centros
        das rodas esquerda e direita, influenciando a dinâmica
        de rotação do veículo em torno do eixo vertical (guinada).

    loss_coeff : float
        Coeficiente de perdas [-]. Representa as perdas mecânicas
        e/ou elétricas associadas ao sistema de transmissão de potência.

    T_min : float
        Torque mínimo disponível ou permitido [N.m]. Define o limite
        inferior de torque fornecido pelo sistema de acionamento.

    T_max : float
        Torque máximo disponível ou permitido [N.m]. Define o limite
        superior de torque fornecido pelo sistema de acionamento.

    Ploss_scale : float
        Fator de escala das perdas de potência [-]. Ajusta a magnitude
        das perdas energéticas consideradas no modelo do veículo.

    mass : float
        Massa total do veículo [kg]. Determina sua inércia translacional
        e influencia sua aceleração longitudinal e lateral.

    yaw_inertia : float
        Momento de inércia de guinada [kg.m²]. Representa a resistência
        do veículo à aceleração angular em torno do eixo vertical que
        passa pelo seu centro de massa.
    """

    def __init__(self, wheel_radius: float, gear_ratio: float, track_width: float, loss_coeff: float, T_min: float, T_max: float, Ploss_scale: float, mass: float, yaw_inertia: float):

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

    """
    Classe que armazena uma referencia de torque total para as rodas.

    Parameters
    ----------
    Fx : float
        Torque total

    Mz : float
        Momento de guinada
    """


    def __init__(self, Fx: float, Mz: float):
        self.Fx = Fx    # N
        self.Mz = Mz    # N.m

class WeelTork:
    """
    Classe que armazena uma referencia de torque total para as rodas.

    Parameters
    ----------

    Fl : float
        torque da roda esquerda da frente
    Fr : float
        torque da roda direita da frente
    Rl : float
        torque da roda esquerda da tras
    Rr : float
        torque da roda direita da tras
=
    """
    def __init__(self, Fl: float, Fr: float, Rl: float, Rr: float):

        self.Fl = Fl
        self.Fr = Fr
        self.Rl = Rl
        self.Rr = Rr

    def __str__(self):
        return F"Fl: {self.Fl:.2f} Fr: {self.Fr:.2f} Rl: {self.Rl:.2f} Rr: {self.Rr:.2f}"

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
    
    def __str__(self):
        return f"P: ({self.X:.2f}m, {self.Y:.2f}m) V: {self.vx:.2f}m/s {self._get_angle_arrow(self.psi)} w: {self.omega}rad"