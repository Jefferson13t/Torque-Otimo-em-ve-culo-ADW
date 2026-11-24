import numpy as np
import matplotlib.pyplot as plt

from classes import Vehicle_state

import heapq
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

def build_cost_map(
    width,
    height,
    obstacle_density=0.50,
    n_regions=3,
    smoothness=1.0,
    seed=None
) -> list[list[int]]:

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


def dijkstra(grid, inicio, destino):
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


def next_cell_dijkstra(
    grid,
    estado: Vehicle_state,
    destino: tuple[int, int],
    cell_width: float
):
    """
    Recalcula o caminho usando Dijkstra a partir da posição atual
    do veículo e retorna a próxima célula a ser perseguida.

    Retorna:
        proxima_celula: próxima célula do caminho
        caminho: caminho completo calculado
        celula_atual: célula onde o veículo está
    """

    linhas = len(grid)
    colunas = len(grid[0])

    # Converter posição (X,Y) em célula (linha,coluna)
    linha = int(estado.Y // cell_width)
    coluna = int(estado.X // cell_width)

    celula_atual = (linha, coluna)

    # Verificar se está dentro do grid
    if not (0 <= linha < linhas and 0 <= coluna < colunas):
        return None, [], celula_atual

    # Executar Dijkstra a partir da posição atual
    caminho = dijkstra(
        grid,
        celula_atual,
        destino
    )

    # Verificar se existe caminho
    if not caminho:
        return None, [], celula_atual

    # Selecionar próxima célula
    if len(caminho) > 1:
        proxima_celula = caminho[1]
    else:
        proxima_celula = celula_atual

    return proxima_celula, caminho, celula_atual

def plot_trajectory(
    states: list[Vehicle_state],
    grid: list[list[int]],
    path: list[tuple[int, int]],
    CELL_WIDTH: float = 0.1,
) -> None:

    if not states:
        raise ValueError("A lista de estados está vazia.")

    if not grid or not grid[0]:
        raise ValueError("O grid está vazio.")

    if not path:
        raise ValueError("O caminho está vazio.")

    # Extrair posições do veículo
    X = np.array([state.X for state in states])
    Y = np.array([state.Y for state in states])
    psi = np.array([state.psi for state in states])

    grid = np.array(grid)

    rows, cols = grid.shape

    # Coordenadas dos centros das células
    path_X = np.array([
        (j + 0.5) * CELL_WIDTH for i, j in path
    ])

    path_Y = np.array([
        (i + 0.5) * CELL_WIDTH for i, j in path
    ])

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
    ax.set_xticks(np.arange(0, cols * CELL_WIDTH + CELL_WIDTH, CELL_WIDTH))
    ax.set_yticks(np.arange(0, rows * CELL_WIDTH + CELL_WIDTH, CELL_WIDTH))

    ax.grid(True, color='black', linewidth=0.5, alpha=0.4)

    # Caminho do Dijkstra
    ax.plot(
        path_X,
        path_Y,
        color='orange',
        linewidth=2,
        linestyle='--',
        marker='.',
        markersize=5,
        label='Caminho Dijkstra',
        zorder=3
    )

    # Trajetória real do veículo
    ax.plot(
        X, Y,
        color='blue',
        linewidth=2,
        label='Trajetória do veículo',
        zorder=4
    )

    # Posição inicial
    ax.scatter(
        X[0], Y[0],
        color='lime',
        edgecolors='black',
        s=100,
        marker='o',
        label='Início',
        zorder=5
    )

    # Posição final
    ax.scatter(
        X[-1], Y[-1],
        color='red',
        edgecolors='black',
        s=100,
        marker='X',
        label='Final',
        zorder=5
    )

    # Orientação inicial e final
    arrow_length = CELL_WIDTH * 0.5

    for idx, label in [(0, 'Inicial'), (-1, 'Final')]:
        ax.quiver(
            X[idx], Y[idx],
            np.cos(psi[idx]),
            np.sin(psi[idx]),
            angles='xy',
            scale_units='xy',
            scale=1 / arrow_length,
            color='black',
            width=0.005,
            zorder=6
        )

    # Limites do mapa
    ax.set_xlim(0, cols * CELL_WIDTH)
    ax.set_ylim(rows * CELL_WIDTH, 0)

    ax.set_aspect('equal')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajetória do veículo e caminho Dijkstra')

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


