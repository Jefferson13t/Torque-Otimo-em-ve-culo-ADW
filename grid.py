import numpy as np
import matplotlib.pyplot as plt

from classes import Vehicle_state

import heapq
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

def build_cost_map(
    width: int,
    height: int,
    obstacle_density: float = 0.50,
    n_regions: int = 3,
    smoothness: float =1.0,
    seed=None 
) -> list[list[int]]:
    """
    Gera um mapa de custos bidimensional representando o ambiente,
    considerando a distribuição de obstáculos e regiões com diferentes
    características de custo.

    Parameters
    ----------
    width : int
        Largura do mapa, expressa em número de células.

    height : int
        Altura do mapa, expressa em número de células.

    obstacle_density : float, optional
        Densidade de obstáculos no ambiente, expressa como uma fração
        entre 0 e 1. Controla a proporção de células ocupadas por
        obstáculos. O valor padrão é 0.50.

    n_regions : int, optional
        Número de regiões utilizadas na composição do mapa de custos.
        Permite representar diferentes áreas com características
        distintas de custo. O valor padrão é 3.

    smoothness : float, optional
        Parâmetro que controla a suavidade espacial da distribuição
        dos custos no mapa. Valores maiores tendem a produzir
        transições mais suaves entre regiões. O valor padrão é 1.0.

    seed : int, optional
        Semente utilizada pelo gerador de números aleatórios para
        garantir a reprodutibilidade do mapa gerado. Quando None,
        a geração utiliza uma semente aleatória. O valor padrão é None.

    Returns
    -------
    list[list[int]]
        Matriz bidimensional de custos, com dimensões height x width.
        Cada elemento representa o custo associado à respectiva célula
        do ambiente, podendo representar diferentes níveis de
        dificuldade de navegação ou a presença de obstáculos.
    """


    rng = np.random.default_rng(seed)

    # 1. Criar campo aleatório espacialmente correlacionado
    noise = rng.random((height, width))

    terrain = gaussian_filter(
        noise,
        sigma=smoothness
    )

    # Normalizar entre 0 e 1
    terrain = (
        (terrain - terrain.min()) /
        (terrain.max() - terrain.min() + 1e-10)
    )

    # 2. Criar regiões com diferentes características
    cost_map = 1 + 9 * terrain

    # 3. Adicionar variações locais
    local_noise = rng.normal(
        loc=0,
        scale=0.7,
        size=(height, width)
    )

    cost_map += local_noise

    # 4. Criar regiões de alto custo
    for _ in range(n_regions):

        cx = rng.integers(0, width)
        cy = rng.integers(0, height)

        radius = rng.integers(
            max(2, min(width, height) // 12),
            max(3, min(width, height) // 5)
        )

        yy, xx = np.ogrid[:height, :width]

        distance = np.sqrt(
            (xx - cx)**2 + (yy - cy)**2
        )

        influence = np.exp(
            -(distance**2) / (2 * radius**2)
        )

        cost_map += 4 * influence

    # 5. Limitar custos entre 1 e 10
    cost_map = np.clip(cost_map, 1, 100)

    # 6. Criar obstáculos
    obstacles = rng.random((height, width)) < obstacle_density

    cost_map[obstacles] = 0

    # 7. Converter para inteiros
    cost_map = np.rint(cost_map).astype(int)

    return cost_map.tolist()


def dijkstra(grid: list[list[int]], inicio: tuple[int, int], destino: tuple[int, int]):
    """
    Roda o dijkstra a partir da celula inicial e retorna o caminho até a celula final.

    Parameters
    ----------
    grid : list[list[int]]
        matriz de custo do ambiente

    inicio : tuple[int, int]
        celula de inicio

    destino : tuple[int, int]
        celula de destino

    """

    linhas = len(grid)
    colunas = len(grid[0])

    # Movimentos: cima, baixo, esquerda, direita e diagonais
    movimentos = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        ( 1,  1, math.sqrt(2))
    ]

    distancias = {inicio: 0}
    anteriores = {}
    fila = [(0, inicio)]

    while fila:
        custo_atual, atual = heapq.heappop(fila)

        if atual == destino:
            break

        if custo_atual > distancias.get(atual, float('inf')):
            continue

        i, j = atual

        for di, dj, distancia in movimentos:
            ni, nj = i + di, j + dj

            # Verifica limites do grid
            if not (0 <= ni < linhas and 0 <= nj < colunas):
                continue

            # Ignora obstáculos
            if grid[ni][nj] == 0:
                continue

            vizinho = (ni, nj)

            # Custo para entrar na célula
            novo_custo = (
                custo_atual +
                grid[ni][nj] * distancia
            )

            if novo_custo < distancias.get(vizinho, float('inf')):
                distancias[vizinho] = novo_custo
                anteriores[vizinho] = atual

                heapq.heappush(
                    fila,
                    (novo_custo, vizinho)
                )

    # Verifica se existe caminho
    if destino not in distancias:
        return []

    # Reconstrói o caminho
    caminho = []
    atual = destino

    while atual != inicio:
        caminho.append(atual)
        atual = anteriores[atual]

    caminho.append(inicio)

    return caminho[::-1]


def dijkstra_reverso(grid: list[list[int]], destino: tuple[int, int]):
    """
    Roda o dijkstra a partir da celula de destino para calcular a distancia minima de todas as celulas até ela.

    Parameters
    ----------
    grid : list[list[int]]
        matriz de custo do ambiente

    destino : tuple[int, int]
        celula de destino

    """

    linhas = len(grid)
    colunas = len(grid[0])

    movimentos = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        ( 1,  1, math.sqrt(2))
    ]

    distancias = {destino: 0}
    proximos = {}
    fila = [(0, destino)]

    while fila:
        custo_atual, atual = heapq.heappop(fila)

        if custo_atual > distancias.get(atual, float('inf')):
            continue

        i, j = atual

        for di, dj, distancia in movimentos:
            ni, nj = i + di, j + dj

            if not (0 <= ni < linhas and 0 <= nj < colunas):
                continue

            if grid[ni][nj] == 0:
                continue

            vizinho = (ni, nj)

            novo_custo = custo_atual + grid[ni][nj] * distancia

            if novo_custo < distancias.get(vizinho, float('inf')):
                distancias[vizinho] = novo_custo
                proximos[vizinho] = atual

                heapq.heappush(
                    fila,
                    (novo_custo, vizinho)
                )

    # Objeto de consulta
    resultado = {}

    for celula, custo in distancias.items():
        resultado[celula] = proximos.get(celula)

    return resultado


import numpy as np
import matplotlib.pyplot as plt


def plot_vector_field(policy, cell_width=1.0, figsize=(10, 10)):
    """
    Plota o campo vetorial de uma política de navegação em uma grade.

    Args:
        policy (dict):
            Dicionário em que cada chave representa uma célula (linha, coluna)
            e o valor representa a próxima célula desejada.
            Células com valor None representam o destino.

        cell_width (float):
            Dimensão de cada célula da grade.

        figsize (tuple):
            Dimensões da figura.
    """

    fig, ax = plt.subplots(figsize=figsize)

    for (row, col), next_cell in policy.items():

        x = (col + 0.5) * cell_width
        y = (row + 0.5) * cell_width

        # Destino
        if next_cell is None:
            ax.scatter(x, y, color="red", s=100, marker="*",
                       label="Destino")
            continue

        next_row, next_col = next_cell

        # Vetor de deslocamento
        dx = (next_col - col) * cell_width
        dy = (next_row - row) * cell_width

        ax.quiver(
            x, y,
            dx, dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
            width=0.003
        )

    # Configuração da grade
    rows = [cell[0] for cell in policy]
    cols = [cell[1] for cell in policy]

    ax.set_xticks(np.arange(min(cols), max(cols) + 1) * cell_width)
    ax.set_yticks(np.arange(min(rows), max(rows) + 1) * cell_width)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Campo Vetorial da Política de Navegação")

    # Inverter eixo Y para manter convenção da matriz (linha, coluna)
    ax.invert_yaxis()

    # Evitar entradas duplicadas na legenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()

def plot_trajectory(
    grid: list[list[int]],
    CELL_WIDTH: float,
    paths: list[list[Vehicle_state]],
) -> None:
    """
    Plota o grid do ambiente e os caminhos percorridos.

    Parameters
    ----------
    grid : list[list[int]]
        matriz de custo do ambiente

    CELL_WIDTH : float
        tamanho de cada celula do ambiente

    paths : list[list[Vehicle_state]]
        lista de caminhos percorridos para plotagem

    """

    if not grid or not grid[0]:
        raise ValueError("O grid está vazio.")

    if not paths:
        raise ValueError("A lista de caminhos está vazia.")


    grid = np.array(grid)
    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotar grid
    ax.imshow(
        grid,
        cmap='RdYlGn_r',
        origin='upper',
        extent=[
            0,
            cols * CELL_WIDTH,
            rows * CELL_WIDTH,
            0
        ],
        interpolation='nearest',
        alpha=0.35,
        vmin=0,
        vmax=10
    )

    # Linhas do grid
    ax.set_xticks(
        np.arange(0, cols * CELL_WIDTH + CELL_WIDTH, CELL_WIDTH)
    )
    ax.set_yticks(
        np.arange(0, rows * CELL_WIDTH + CELL_WIDTH, CELL_WIDTH)
    )

    ax.grid(True, color='black', linewidth=0.5, alpha=0.4)

    # Cores diferentes para cada trajetória
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(paths))
    )

    arrow_length = CELL_WIDTH * 0.5

    for idx, (states, color) in enumerate(zip(paths, colors)):

        X = np.array([state.X for state in states])
        Y = np.array([state.Y for state in states])
        psi = np.array([state.psi for state in states])

        label = f'Trajetória {idx + 1}'

        # Trajetória
        ax.plot(
            X, Y,
            color=color,
            linewidth=2,
            label=label,
            zorder=4
        )

        # Posição inicial
        ax.scatter(
            X[0], Y[0],
            color=color,
            edgecolors='black',
            s=80,
            marker='o',
            zorder=5
        )

        # Posição final
        ax.scatter(
            X[-1], Y[-1],
            color=color,
            edgecolors='black',
            s=80,
            marker='X',
            zorder=5
        )

        # Orientação inicial e final
        for i in [0, -1]:
            ax.quiver(
                X[i], Y[i],
                np.cos(psi[i]),
                np.sin(psi[i]),
                angles='xy',
                scale_units='xy',
                scale=1 / arrow_length,
                color=color,
                width=0.005,
                zorder=6
            )

        # Orientações intermediárias
        for i in range(50, len(states), 50):
            ax.quiver(
                X[i], Y[i],
                np.cos(psi[i]),
                np.sin(psi[i]),
                angles='xy',
                scale_units='xy',
                scale=1 / arrow_length,
                color=color,
                width=0.005,
                alpha=0.7,
                zorder=6
            )

    # Limites do mapa
    ax.set_xlim(0, cols * CELL_WIDTH)
    ax.set_ylim(rows * CELL_WIDTH, 0)

    ax.set_aspect('equal')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Comparação de trajetórias')

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
