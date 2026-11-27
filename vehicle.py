import numpy as np
import math

from problem_model import model

from classes import WeelTork, Vehicle_state, CarParams


def vehicle_dynamics(T: WeelTork, state: Vehicle_state, params: CarParams, dt: float) -> Vehicle_state:
    """Utiliza o estado atual e o torque aplicado para calcular o estado do carrinho no proximo instante de tempo."""

    # Estado atual
    X = state.X
    Y = state.Y 
    psi = state.psi 
    vx = state.vx 
    omega = state.omega

    # Forças e momento gerados pelos motores
    Fx, Mz, Ploss = model(T, params)

    # Parâmetros dinâmicos
    m = params.mass
    Iz = params.yaw_inertia

    # Acelerações
    ax = Fx / m
    alpha = Mz / Iz

    # Integração de Euler semi-implícita
    vx_next = vx + ax * dt
    omega_next = omega + alpha * dt

    psi_next = psi + omega_next * dt

    X_next = X + vx_next * np.cos(psi_next) * dt
    Y_next = Y + vx_next * np.sin(psi_next) * dt

    return Vehicle_state(
        X_next,
        Y_next,
        psi_next,
        vx_next,
        omega_next
    )

def force_moment_to_torque(Fx, Mz, params: CarParams) -> WeelTork:

    r = params.wheel_radius
    i = params.gear_ratio
    d = params.track_width

    # Soma dos torques das rodas
    T_sum = Fx * r / i

    # Diferença entre os torques esquerdo e direito
    T_diff = 2 * r * Mz / (d * i)

    # Torque total de cada lado
    T_left = (T_sum + T_diff) / 2
    T_right = (T_sum - T_diff) / 2

    # Distribuição entre os eixos dianteiro e traseiro
    T_fl = T_left / 2
    T_rl = T_left / 2

    T_fr = T_right / 2
    T_rr = T_right / 2

    # Saturação dos motores
    T_min = params.T_min
    T_max = params.T_max

    return WeelTork(
        Fl=np.clip(T_fl, T_min, T_max),
        Fr=np.clip(T_fr, T_min, T_max),
        Rl=np.clip(T_rl, T_min, T_max),
        Rr=np.clip(T_rr, T_min, T_max)
    )

def generate_torque_reference(
    caminho,
    estado_inicial: Vehicle_state,
    params: CarParams,
    cell_width: float,
    dt: float = 0.01,
    v_ref: float = 0.3,
    Kp_psi: float = 1.0,
    Kd_psi: float = 2.0,
    Kp_v: float = 20.0
) -> tuple[list[WeelTork], list[Vehicle_state]]:
    """
    Gera referências de torque para as rodas do veículo a partir de um
    caminho definido por waypoints, utilizando controladores de orientação
    PD e velocidade proporcional (P).

    A função converte os waypoints da grade em coordenadas cartesianas,
    calcula as referências de força longitudinal e momento de guinada
    e as converte em torques aplicados ao modelo dinâmico do veículo.
    A simulação termina quando o destino é alcançado ou após 10000 iterações.

    Args:
        caminho (list[tuple[int, int]]): Lista de waypoints no formato
            (linha, coluna), representando o caminho desejado.
        estado_inicial (Vehicle_state): Estado inicial do veículo,
            contendo posição, orientação e velocidades.
        params (CarParams): Parâmetros físicos do veículo utilizados
            no cálculo dos torques e na simulação dinâmica.
        cell_width (float): Dimensão da célula da grade, em metros.
            Utilizada na conversão dos waypoints e nas tolerâncias
            de distância para avanço e chegada.
        dt (float, optional): Passo de tempo da simulação, em segundos.
            Padrão: 0.01.
        v_ref (float, optional): Velocidade longitudinal desejada,
            em m/s. Padrão: 0.3.
        Kp_psi (float, optional): Ganho proporcional do controlador
            de orientação. Padrão: 1.0.
        Kd_psi (float, optional): Ganho derivativo do controlador
            de orientação. Padrão: 2.0.
        Kp_v (float, optional): Ganho proporcional do controlador
            de velocidade longitudinal. Padrão: 20.0.

    Returns:
        tuple[list[WeelTork], list[Vehicle_state]]:
            - Lista de referências de torque para as rodas.
            - Lista de estados do veículo calculados durante a simulação.

    Raises:
        ValueError: Se o caminho for None ou estiver vazio.

    Notes:
        A força longitudinal é limitada ao intervalo [0, 30] N e o
        momento de guinada ao intervalo [-2, 2] N.m.
        O erro angular é normalizado para [-pi, pi].
    """

    torques_referencia = []
    estados_calculados = []

    if caminho is None or len(caminho) == 0:
        raise ValueError("O caminho está vazio!")

    # Converter caminho (linha, coluna) em coordenadas (X, Y)
    waypoints = np.array([
        (
            (col + 0.5) * cell_width,
            (row + 0.5) * cell_width
        )
        for row, col in caminho
    ])

    state = estado_inicial

    waypoint_idx = 1

    # Simulação ao longo do caminho
    for k in range(10000):

        estados_calculados.append(state)

        # Verificar chegada ao destino
        distancia_final = np.linalg.norm(
            np.array([state.X, state.Y]) - waypoints[-1]
        )

        if distancia_final < cell_width / 3:
            break

        # Selecionar próximo waypoint
        while waypoint_idx < len(waypoints) - 1:

            target = waypoints[waypoint_idx]

            distancia = np.linalg.norm(
                target - np.array([state.X, state.Y])
            )

            if distancia < cell_width / 2:
            # if distancia < 0.04:
                waypoint_idx += 1
            else:
                break

        target = waypoints[waypoint_idx]

        # Orientação desejada
        psi_ref = np.arctan2(
            target[1] - state.Y,
            target[0] - state.X
        )

        # Erro angular normalizado
        e_psi = np.arctan2(
            np.sin(psi_ref - state.psi),
            np.cos(psi_ref - state.psi)
        )

        # Controle de orientação
        Mz_ref = Kp_psi * e_psi - Kd_psi * state.omega

        # Controle de velocidade
        Fx_ref = Kp_v * (v_ref - state.vx)

        # Limites de força e momento
        Fx_ref = np.clip(Fx_ref, 0.0, 30.0)
        Mz_ref = np.clip(Mz_ref, -2.0, 2.0)

        # Converter referências em torques
        T_ref = force_moment_to_torque(
            Fx_ref,
            Mz_ref,
            params
        )

        torques_referencia.append(T_ref)

        # Atualizar estado do veículo
        next_state = vehicle_dynamics(
            T_ref,
            state,
            params,
            dt
        )

        # Compatibilidade com retorno em array NumPy
        if isinstance(next_state, np.ndarray):
            state = Vehicle_state(*next_state)
        else:
            state = next_state

    return torques_referencia, estados_calculados