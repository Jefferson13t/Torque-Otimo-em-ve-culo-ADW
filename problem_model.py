import numpy as np

from pymoo.core.problem import ElementwiseProblem

from classes import WeelTork, CarParams, Reference

def model(T: WeelTork, params: CarParams):
    """
    Modelo simplificado de distribuição de torque para veículo
    com quatro motores nas rodas.

    Parameters
    ----------
    T : WeelTork
        Torques [T_fl, T_fr, T_rl, T_rr] em N.m.

    params : CarParams
        Parâmetros do veículo.

    Returns
    -------
    Fx : float
        Força longitudinal [N].

    Mz : float
        Momento de guinada [N.m].

    Ploss : float
        Perdas totais dos motores [W].
    """

    r = params.wheel_radius
    i0 = params.gear_ratio
    d = params.track_width

    # Força longitudinal total
    Fx = (i0 / r) * (T.Fl + T.Fr + T.Rl + T.Rr)

    # Momento de guinada
    Mz = (d * i0 / (2 * r)) * (
        T.Fl - T.Fr + T.Rl - T.Rr
    )

    # Modelo simplificado de perdas dos motores
    # Ploss = k_loss * T^2
    k_loss = params.loss_coeff

    Ploss = k_loss * np.sum(np.asarray([T.Fl, T.Fr, T.Rl, T.Rr]) ** 2)

    return Fx, Mz, Ploss

class TorqueDistributionProblem(ElementwiseProblem):
# n_var: Número de variáveis de decisão

# n_obj: Número de funções objetivo

# n_ieq_constr: Número de restrições de desigualdade

# n_eq_constr: Número de restrições de igualdade

# xl: Limites inferiores das variáveis

# xu: Limites superiores das variáveis

# X: Valores das variáveis de decisão recebidos pelo modelo


# out: Dicionário utilizado para armazenar os resultados

# out["F"]: valores das funções objetivo.

# out["G"]: valores das restrições de desigualdade.

# out["H"]: valores das restrições de igualdade.

    def __init__(
        self,
        params: CarParams,
        ref: Reference,
        T_prev: WeelTork,
        weights=None
    ):

        self.params = params

        self.Fx_ref = ref.Fx
        self.Mz_ref = ref.Mz

        self.T_prev = np.asarray([T_prev.Fl, T_prev.Fr, T_prev.Rl, T_prev.Rr], dtype=float)

        if weights is None:
            weights = {
                "force": 1.0,
                "yaw": 1.0,
                "energy": 0.1,
                "torque_change": 0.01
            }

        self.weights = weights

        # Limites de torque por motor
        T_min = params.T_min
        T_max = params.T_max

        # Escalas para normalização
        self.Fx_scale = max(abs(self.Fx_ref), 100.0)
        self.Mz_scale = max(abs(self.Mz_ref), 100.0)
        self.Ploss_scale = params.Ploss_scale
        self.T_scale = max(abs(T_max), abs(T_min), 1.0)

        super().__init__(
            n_var=4,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.full(4, T_min),
            xu=np.full(4, T_max)
        )

    def _evaluate(self, x, out, *args, **kwargs):

        T = WeelTork(
            Fl=x[0],
            Fr=x[1],
            Rl=x[2],
            Rr=x[3],
        )

        Fx, Mz, Ploss = model(T, self.params)

        # Objetivos originais
        # f1 = (Fx - self.Fx_ref)**2
        f1 = (Mz - self.Mz_ref)**2
        f2 = Ploss

        # Normalização
        f1_norm = f1
        f2_norm = f2

        out["F"] = np.column_stack([
            # f1_norm,
            f1_norm,
            f2_norm
        ])

        # Restrição de força longitudinal
        epsilon = 5  # Tolerância em N

        g1 = self.Fx_ref - Fx - epsilon

        out["G"] = g1.reshape(-1, 1)
