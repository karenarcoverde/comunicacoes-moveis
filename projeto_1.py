import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.path import Path
import pandas as pd


# ── Erlang-B ──────────────────────────────────────────────────────────
def erlang_b(A, C):
    if A <= 0 or C == 0:
        return 1.0

    inv = 1.0

    for k in range(1, C + 1):
        inv = 1.0 + inv * k / A

    return 1.0 / inv


# ── Geometria hexagonal ───────────────────────────────────────────────
def hex_center(i, j, R):
    x = R * math.sqrt(3) * (i + j * 0.5)
    y = R * 1.5 * j
    return x, y


def build_grid(grid_r, R):
    centers = {}

    for i in range(-grid_r, grid_r + 1):
        for j in range(-grid_r, grid_r + 1):
            if abs(i + j) <= grid_r:
                centers[(i, j)] = hex_center(i, j, R)

    return centers


def get_cluster_cells(N):
    directions = [
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1)
    ]

    cells = [(0, 0)]
    seen = {(0, 0)}
    frontier = [(0, 0)]

    while len(cells) < N:
        nxt = []

        for ci, cj in frontier:
            for di, dj in directions:
                nb = (ci + di, cj + dj)

                if nb not in seen:
                    seen.add(nb)
                    cells.append(nb)
                    nxt.append(nb)

                    if len(cells) == N:
                        break

            if len(cells) == N:
                break

        frontier = nxt

        if not frontier:
            break

    return cells[:N]


def get_cochannel(N):
    params = {
        3: (1, 1),
        4: (2, 0),
        7: (2, 1)
    }

    i0, j0 = params[N]

    def rot60(i, j):
        return (-j, i + j)

    translations = []

    vi, vj = i0, j0

    for _ in range(6):
        translations.append((vi, vj))
        vi, vj = rot60(vi, vj)

    return translations


# ── Funções auxiliares para SIR ───────────────────────────────────────
def calc_sir_db(user_pos, serving_pos, interferer_positions, n_path, d0=0.001):
    ux, uy = user_pos
    sx, sy = serving_pos

    ds = math.sqrt((ux - sx) ** 2 + (uy - sy) ** 2)
    ds = max(ds, d0)

    Ps = ds ** (-n_path)

    I_total = 0.0

    for ix, iy in interferer_positions:
        di = math.sqrt((ux - ix) ** 2 + (uy - iy) ** 2)
        di = max(di, d0)
        I_total += di ** (-n_path)

    sir_linear = Ps / (I_total + 1e-12)

    return 10 * math.log10(sir_linear)


def hex_vertices(cx, cy, radius, orientation=0):
    angles = orientation + np.linspace(0, 2 * np.pi, 7)[:-1]

    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ])


def point_inside_hex(x, y, cx, cy, radius):
    verts = hex_vertices(cx, cy, radius)
    path = Path(verts)

    return path.contains_point((x, y))


def find_serving_cell(x, y, visible_cells, grid_centers, hex_r):
    for cell in visible_cells:
        cx, cy = grid_centers[cell]

        if point_inside_hex(x, y, cx, cy, hex_r):
            return cell

    return None


# ── Entradas ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Cobertura Celular", page_icon="📡")

with st.sidebar:
    st.header("Parâmetros de Entrada")

    R = st.number_input(
        "Raio da célula R (km)",
        min_value=0.1,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

    N = st.selectbox(
        "Tamanho do cluster N",
        [3, 4, 7],
        index=2
    )

    S = st.number_input(
        "Número de canais totais",
        min_value=1,
        max_value=2000,
        value=395,
        step=1
    )

    A = st.number_input(
        "Tráfego oferecido (Erlangs)",
        min_value=0.1,
        max_value=5000.0,
        value=100.0,
        step=1.0
    )

    alpha = st.slider(
        "Não-linearidade do PA da BS (fator de compressão de ganho)",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.5
    )


tab1, tab2 = st.tabs(["Malha Hexagonal", "REM CCI"])

# Expoente de perda de percurso usado no cálculo da SIR
n_path = 4


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — MALHA HEXAGONAL
# ──────────────────────────────────────────────────────────────────────
with tab1:
    cluster_cells = get_cluster_cells(N)
    cluster_set = set(cluster_cells)
    translations = get_cochannel(N)
    grid_centers = build_grid(8, R)

    COLORS = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1"
    ]

    cluster_index = {
        cell: idx + 1
        for idx, cell in enumerate(cluster_cells)
    }

    cell_color = {
        cell: COLORS[idx % len(COLORS)]
        for idx, cell in enumerate(cluster_cells)
    }

    cochannel_map = {}
    cochannel_color = {}

    for cell in cluster_cells:
        ci, cj = cell
        label = cluster_index[cell]
        color = cell_color[cell]

        for ti, tj in translations:
            pos = (ci + ti, cj + tj)

            if pos not in cluster_set:
                cochannel_map[pos] = label
                cochannel_color[pos] = color

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_facecolor("#f5f5f5")

    hex_r = R * 0.97

    for (i, j), (cx, cy) in grid_centers.items():

        if (i, j) in cluster_set:
            patch = RegularPolygon(
                (cx, cy),
                6,
                radius=hex_r,
                orientation=0,
                facecolor=cell_color[(i, j)],
                edgecolor="white",
                linewidth=1.8
            )

            ax.add_patch(patch)

            ax.text(
                cx,
                cy,
                str(cluster_index[(i, j)]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white"
            )

        elif (i, j) in cochannel_map:
            patch = RegularPolygon(
                (cx, cy),
                6,
                radius=hex_r,
                orientation=0,
                facecolor=cochannel_color[(i, j)],
                edgecolor="red",
                linewidth=2.0,
                alpha=0.45
            )

            ax.add_patch(patch)

            ax.text(
                cx,
                cy,
                str(cochannel_map[(i, j)]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="red"
            )

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]

    pad = R * 1.5

    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = [
        mpatches.Patch(
            facecolor=COLORS[idx % len(COLORS)],
            edgecolor="white",
            label=f"Célula {idx + 1} (cluster / co-canal)"
        )
        for idx in range(len(cluster_cells))
    ]

    legend_handles.append(
        mpatches.Patch(
            facecolor="white",
            edgecolor="red",
            linewidth=2,
            label="Célula co-canal do cluster"
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#aaaaaa"
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — REM CCI COM SELEÇÃO DE CÉLULA
# ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("REM CCI com análise por célula")

    cluster_cells = get_cluster_cells(N)
    cluster_set = set(cluster_cells)
    translations = get_cochannel(N)
    grid_centers = build_grid(8, R)

    COLORS = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1"
    ]

    cluster_index = {
        cell: idx + 1
        for idx, cell in enumerate(cluster_cells)
    }

    cell_color = {
        cell: COLORS[idx % len(COLORS)]
        for idx, cell in enumerate(cluster_cells)
    }

    # Célula escolhida pelo usuário
    selected_label = st.selectbox(
        "Escolha a célula para atualizar a tabela e destacar no REM",
        list(range(1, N + 1)),
        index=0
    )

    selected_cell = cluster_cells[selected_label - 1]

    # Mapa das células co-canais
    cochannel_map = {}

    for cell in cluster_cells:
        ci, cj = cell

        for ti, tj in translations:
            pos = (ci + ti, cj + tj)

            if pos not in cluster_set:
                cochannel_map[pos] = cluster_index[cell]

    visible_cells = list(cluster_set) + list(cochannel_map.keys())

    cell_label_map = {
        cell: cluster_index[cell]
        for cell in cluster_set
    }

    cell_label_map.update(cochannel_map)

    fig2, ax2 = plt.subplots(figsize=(9, 9), dpi=100)
    ax2.set_facecolor("#f5f5f5")

    hex_r = R * 0.97

    # Desenha a malha de fundo
    for (i, j), (cx, cy) in grid_centers.items():

        if (i, j) in cluster_set:
            patch = RegularPolygon(
                (cx, cy),
                6,
                radius=hex_r,
                orientation=0,
                facecolor=cell_color[(i, j)],
                edgecolor="white",
                linewidth=1.8,
                alpha=0.35
            )

            ax2.add_patch(patch)

            ax2.text(
                cx,
                cy,
                str(cluster_index[(i, j)]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white"
            )

        elif (i, j) in cochannel_map:
            patch = RegularPolygon(
                (cx, cy),
                6,
                radius=hex_r,
                orientation=0,
                facecolor="white",
                edgecolor="red",
                linewidth=1.2,
                alpha=0.25
            )

            ax2.add_patch(patch)

            ax2.text(
                cx,
                cy,
                str(cochannel_map[(i, j)]),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white"
            )

    # Região do REM
    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]

    x_min = min(all_x) - R
    x_max = max(all_x) + R
    y_min = min(all_y) - R
    y_max = max(all_y) + R

    resolution = 180

    xs_grid = np.linspace(x_min, x_max, resolution)
    ys_grid = np.linspace(y_min, y_max, resolution)

    rem_x = []
    rem_y = []
    rem_sir = []

    # Guarda SIR separada por célula do cluster principal
    sir_por_celula = {
        label: []
        for label in range(1, N + 1)
    }

    for x in xs_grid:
        for y in ys_grid:
            serving_cell = find_serving_cell(
                x,
                y,
                visible_cells,
                grid_centers,
                hex_r
            )

            if serving_cell is None:
                continue

            serving_label = cell_label_map[serving_cell]
            serving_pos = grid_centers[serving_cell]

            interferer_positions = []

            for other_cell in visible_cells:
                if other_cell == serving_cell:
                    continue

                if cell_label_map[other_cell] == serving_label:
                    interferer_positions.append(grid_centers[other_cell])

            if len(interferer_positions) == 0:
                continue

            sir_db = calc_sir_db(
                user_pos=(x, y),
                serving_pos=serving_pos,
                interferer_positions=interferer_positions,
                n_path=n_path
            )

            rem_x.append(x)
            rem_y.append(y)
            rem_sir.append(sir_db)

            # Só guarda para as células do cluster principal
            if serving_cell in cluster_set:
                sir_por_celula[serving_label].append(sir_db)

    rem_sir = np.array(rem_sir)

    if len(rem_sir) > 0:
        sir_vmin = np.percentile(rem_sir, 5)
        sir_vmax = np.percentile(rem_sir, 95)
    else:
        sir_vmin = 0
        sir_vmax = 50

    cmap_sir = plt.colormaps["turbo"]
    norm_sir = Normalize(vmin=sir_vmin, vmax=sir_vmax)

    sc = ax2.scatter(
        rem_x,
        rem_y,
        c=rem_sir,
        cmap=cmap_sir,
        norm=norm_sir,
        s=5,
        marker="s",
        alpha=0.9
    )

    # Circula a célula selecionada no REM
    # Desenha preenchimento branco na célula selecionada
    selected_cx, selected_cy = grid_centers[selected_cell]

    # Sobreposição do hexágono branco para destaque
    highlight_patch = RegularPolygon(
        (selected_cx, selected_cy),
        6,
        radius=hex_r,
        orientation=0,
        facecolor="none",
        edgecolor="white",
        linewidth=2,
    )
    ax2.add_patch(highlight_patch)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect("equal")
    ax2.axis("off")

    cbar = fig2.colorbar(
        sc,
        ax=ax2,
        fraction=0.03,
        pad=0.02
    )

    cbar.set_label("SIR [dB]", fontsize=9)

    valores_barra = cbar.get_ticks()

    if len(valores_barra) > 1:
        sir_min_db = valores_barra[1]
    else:
        sir_min_db = sir_vmin

    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

    # ── Tabela antes/depois da célula selecionada ──────────────────────
    valores_sir_celula = np.array(sir_por_celula[selected_label])

    if len(valores_sir_celula) > 0:
        C_teorico = S // N

        fracao_critica = np.mean(valores_sir_celula < sir_min_db)
        fracao_util = 1.0 - fracao_critica

        C_efetivo = max(1, int(C_teorico * fracao_util))

        B_teorico = erlang_b(A, C_teorico)
        B_efetivo = erlang_b(A, C_efetivo)

        st.markdown(
            f"### Validação da capacidade com Erlang-B e SIR — Célula {selected_label}"
        )

        tabela_capacidade = [
            {
                "Célula": f"Célula {selected_label}",
                "Cenário": "Antes — capacidade teórica",
                "Critério usado": "Apenas canais disponíveis",
                "Canais considerados por célula": C_teorico,
                "Área crítica de SIR": "Não considerada",
                "Probabilidade de bloqueio de Erlang-B": f"{100 * B_teorico:.2f}%"
            },
            {
                "Célula": f"Célula {selected_label}",
                "Cenário": "Depois — capacidade efetiva",
                "Critério usado": f"SIR mínima = {sir_min_db:.1f} dB",
                "Canais considerados por célula": C_efetivo,
                "Área crítica de SIR": f"{100 * fracao_critica:.2f}%",
                "Probabilidade de bloqueio de Erlang-B": f"{100 * B_efetivo:.2f}%"
            }
        ]

        df_tabela = pd.DataFrame(tabela_capacidade)

        st.dataframe(
            df_tabela,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "A tabela é atualizada somente para a célula selecionada. "
            "A área crítica é calculada usando apenas os pontos do REM que estão "
            "dentro dessa célula."
        )

    else:
        st.warning(
            "Não foram encontrados pontos válidos de SIR dentro da célula selecionada."
        )