import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.path import Path
import pandas as pd
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates


# ── Função para converter índice em letra ─────────────────────────────
def label_letra(idx):
    return chr(ord("A") + idx)


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


# ──────────────────────────────────────────────────────────────────────
# CACHE DO REM
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Calculando pontos da SIR do REM...")
def calcular_dados_rem_cache(N, _R, _n_path):
    cluster_cells = get_cluster_cells(N)
    cluster_set = set(cluster_cells)
    translations = get_cochannel(N)
    grid_centers = build_grid(8, _R)

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

    for cell in cluster_cells:
        ci, cj = cell

        for ti, tj in translations:
            pos = (ci + ti, cj + tj)

            if pos not in cluster_set and pos in grid_centers:
                cochannel_map[pos] = cluster_index[cell]

    visible_cells = list(cluster_cells) + sorted(list(cochannel_map.keys()))

    cell_label_map = {
        cell: cluster_index[cell]
        for cell in cluster_set
    }

    cell_label_map.update(cochannel_map)

    rem_display_number = {
        cell: idx + 1
        for idx, cell in enumerate(visible_cells)
    }

    number_to_cell = {
        number: cell
        for cell, number in rem_display_number.items()
    }

    hex_r = _R * 0.97

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]

    
    x_min = min(all_x) - _R
    x_max = max(all_x) + _R
    y_min = min(all_y) - _R
    y_max = max(all_y) + _R

    resolution = 180

    xs_grid = np.linspace(x_min, x_max, resolution)
    ys_grid = np.linspace(y_min, y_max, resolution)

    rem_x = []
    rem_y = []
    rem_sir = []

    sir_por_celula = {
        cell: []
        for cell in visible_cells
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
                n_path=_n_path
            )

            rem_x.append(x)
            rem_y.append(y)
            rem_sir.append(sir_db)

            sir_por_celula[serving_cell].append(sir_db)

    rem_sir = np.array(rem_sir)

    if len(rem_sir) > 0:
        sir_vmin = np.percentile(rem_sir, 5)
        sir_vmax = np.percentile(rem_sir, 95)
    else:
        sir_vmin = 0
        sir_vmax = 50

    return {
        "cluster_cells": cluster_cells,
        "cluster_set": cluster_set,
        "grid_centers": grid_centers,
        "COLORS": COLORS,
        "cluster_index": cluster_index,
        "cell_color": cell_color,
        "cochannel_map": cochannel_map,
        "visible_cells": visible_cells,
        "cell_label_map": cell_label_map,
        "rem_display_number": rem_display_number,
        "number_to_cell": number_to_cell,
        "hex_r": hex_r,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "rem_x": rem_x,
        "rem_y": rem_y,
        "rem_sir": rem_sir,
        "sir_por_celula": sir_por_celula,
        "sir_vmin": sir_vmin,
        "sir_vmax": sir_vmax
    }


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


tab1, tab2, tab3 = st.tabs(["Malha Hexagonal", "REM CCI", "PA"])

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
        cell: idx
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
        label = label_letra(cluster_index[cell])
        color = cell_color[cell]

        for ti, tj in translations:
            pos = (ci + ti, cj + tj)

            if pos not in cluster_set:
                cochannel_map[pos] = label
                cochannel_color[pos] = color

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_facecolor("#f5f5f5")

    hex_r = R

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
                label_letra(cluster_index[(i, j)]),
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
                cochannel_map[(i, j)],
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
            label=f"Célula {label_letra(idx)} (cluster / co-canal)"
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
# TAB 2 — REM CCI clicável com tabela automática
# ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("REM CCI com análise por célula")

    # Adiciona instruções de uso
    st.info(
        "💡 Como usar:\n"
        "- Clique em qualquer célula hexagonal para selecioná-la.\n"
        "- A célula selecionada será destacada em vermelho.\n"
        "- A tabela abaixo será automaticamente atualizada com os parâmetros dessa célula.\n"
        "- Altere as entradas: raio, tamanho do cluster, número de canais totais e tráfego; caso seja necessário."
    )

    dados_rem = calcular_dados_rem_cache(N, R, n_path)

    cluster_cells = dados_rem["cluster_cells"]
    cluster_set = dados_rem["cluster_set"]
    grid_centers = dados_rem["grid_centers"]
    COLORS = dados_rem["COLORS"]
    cell_color = dados_rem["cell_color"]
    cochannel_map = dados_rem["cochannel_map"]
    visible_cells = dados_rem["visible_cells"]
    cell_label_map = dados_rem["cell_label_map"]
    rem_display_number = dados_rem["rem_display_number"]
    hex_r = dados_rem["hex_r"]

    rem_x = dados_rem["rem_x"]
    rem_y = dados_rem["rem_y"]
    rem_sir = dados_rem["rem_sir"]

    sir_por_celula = dados_rem["sir_por_celula"]
    sir_vmin = dados_rem["sir_vmin"]
    sir_vmax = dados_rem["sir_vmax"]

    # ── Inicializa célula selecionada na sessão
    if "selected_rem_cell" not in st.session_state:
        st.session_state.selected_rem_cell = visible_cells[0]

    # Se mudar N/R e a célula antiga não existir mais, reseta
    if st.session_state.selected_rem_cell not in visible_cells:
        st.session_state.selected_rem_cell = visible_cells[0]

    # ── Figura do REM
    selected_cell = st.session_state.selected_rem_cell
    fig2, ax2 = plt.subplots(figsize=(9, 9), dpi=100)
    ax2.set_facecolor("#f5f5f5")

    # Desenha hexágonos de fundo
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
        elif (i, j) in cochannel_map:
            logical_label = cochannel_map[(i, j)]
            color = COLORS[(logical_label - 1) % len(COLORS)]
            patch = RegularPolygon(
                (cx, cy),
                6,
                radius=hex_r,
                orientation=0,
                facecolor=color,
                edgecolor="red",
                linewidth=1.2,
                alpha=0.20
            )
            ax2.add_patch(patch)

    # Letras lógicas no REM
    for cell in visible_cells:
        cx, cy = grid_centers[cell]
        letra_logica = label_letra(cell_label_map[cell] - 1)
        ax2.text(
            cx,
            cy,
            letra_logica,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=10
        )

    # Destaque da célula selecionada
    cx, cy = grid_centers[selected_cell]
    highlight = RegularPolygon(
        (cx, cy),
        6,
        radius=hex_r * 1.08,
        orientation=0,
        facecolor="none",
        edgecolor="red",
        linewidth=2.8,
        zorder=30
    )
    ax2.add_patch(highlight)

    # Scatter do SIR
    sc = ax2.scatter(
        rem_x,
        rem_y,
        c=rem_sir,
        cmap=plt.colormaps["turbo"],
        norm=Normalize(vmin=sir_vmin, vmax=sir_vmax),
        s=5,
        marker="o",
        alpha=0.9
    )

    ax2.set_xlim(-12, 12)  # limites fixos
    ax2.set_ylim(-12, 12)
    ax2.set_aspect("equal")
    ax2.axis("off")

    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("SIR [dB]", fontsize=9)
    valores_barra = cbar.get_ticks()
    sir_min_db = valores_barra[1] if len(valores_barra) > 1 else sir_vmin

    # ── REM clicável
    fig2.canvas.draw()
    fig_width, fig_height = fig2.canvas.get_width_height()
    ax_bbox = ax2.get_window_extent()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig2.savefig(tmpfile.name, dpi=100, bbox_inches=None)
        tmpfile_path = tmpfile.name

    click = streamlit_image_coordinates(tmpfile_path, key="rem_click", use_column_width=True)

    if click is not None:
        click_x = click["x"]
        click_y = click["y"]
        scale_x = fig_width / click["width"]
        scale_y = fig_height / click["height"]
        display_x = click_x * scale_x
        display_y = fig_height - click_y * scale_y

        if ax_bbox.x0 <= display_x <= ax_bbox.x1 and ax_bbox.y0 <= display_y <= ax_bbox.y1:
            data_x, data_y = ax2.transData.inverted().transform((display_x, display_y))
            clicked_cell = find_serving_cell(data_x, data_y, visible_cells, grid_centers, hex_r)
            if clicked_cell is not None:
                if clicked_cell != st.session_state.selected_rem_cell:
                    st.session_state.selected_rem_cell = clicked_cell
                    st.rerun()

    plt.close(fig2)

    # ── Atualiza variáveis da tabela após clique
    selected_cell = st.session_state.selected_rem_cell
    selected_visual_number = rem_display_number[selected_cell]
    selected_logical_label = cell_label_map[selected_cell]
    selected_logical_letter = label_letra(selected_logical_label - 1)

    # ── Tabela baseada na célula selecionada
    valores_sir_celula = np.array(sir_por_celula[selected_cell])
    if len(valores_sir_celula) > 0:
        C_teorico = S // N
        fracao_critica = np.mean(valores_sir_celula < sir_min_db)
        C_efetivo = max(1, int(C_teorico*(1-fracao_critica)))
        B_teorico = erlang_b(A, C_teorico)
        B_efetivo = erlang_b(A, C_efetivo)

        st.markdown(f"### Validação da capacidade com Erlang-B e SIR — Célula {selected_logical_letter}")

        tabela_capacidade = [
            {
                "Célula": f"{selected_logical_letter}",
                "Tipo": "Cluster principal" if selected_cell in cluster_set else "Co-canal",
                "Cenário": "Antes — capacidade teórica",
                "Critério usado": "Apenas canais disponíveis",
                "Canais considerados": C_teorico,
                "Área crítica de SIR": "Não considerada",
                "Probabilidade de bloqueio de Erlang-B": f"{100 * B_teorico:.2f}%"
            },
            {
                "Célula": f"{selected_logical_letter}",
                "Tipo": "Cluster principal" if selected_cell in cluster_set else "Co-canal",
                "Cenário": "Depois — capacidade efetiva",
                "Critério usado": f"SIR mínima = {sir_min_db:.1f} dB",
                "Canais considerados": C_efetivo,
                "Área crítica de SIR": f"{100 * fracao_critica:.2f}%",
                "Probabilidade de bloqueio de Erlang-B": f"{100 * B_efetivo:.2f}%"
            }
        ]
        st.dataframe(pd.DataFrame(tabela_capacidade), width="stretch", hide_index=True)
    else:
        st.warning("Não foram encontrados pontos válidos de SIR dentro da célula selecionada.")

# ──────────────────────────────────────────────────────────────────────
# TAB 3 — ESPECTRO DE TRANSMISSÃO REAL DA BS
# ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Espectro de transmissão real da BS")
    # Adiciona instruções de uso
    st.info(
        "💡 Como usar:\n"
        "- Altere a entrada \"não-linearidade do PA da BS\" para ver mudanças na banda de guarda.\n"
    )

    canais_por_celula = S // N

    freq = np.linspace(-1.2, 1.2, 3000)

    nivel_nao_linear = (alpha - 1.0) / (6.0 - 1.0)

    banda_principal = np.exp(-(freq / 0.42) ** 10)

    ondulacao = 1 + 0.03 * np.sin(2 * np.pi * canais_por_celula * freq)

    sinal_principal = banda_principal * ondulacao

    espalhamento_esq = np.exp(-((freq + 0.58) / 0.16) ** 2)
    espalhamento_dir = np.exp(-((freq - 0.58) / 0.16) ** 2)

    espalhamento = nivel_nao_linear * 0.12 * (
        espalhamento_esq + espalhamento_dir
    )

    produto_3ordem_esq = np.exp(-((freq + 0.72) / 0.05) ** 2)
    produto_3ordem_dir = np.exp(-((freq - 0.72) / 0.05) ** 2)

    produtos_3ordem = nivel_nao_linear * 0.18 * (
        produto_3ordem_esq + produto_3ordem_dir
    )

    piso = 1e-8

    espectro_ideal = banda_principal + piso
    espectro_real = sinal_principal + espalhamento + produtos_3ordem + piso

    espectro_ideal_db = 10 * np.log10(espectro_ideal / np.max(espectro_ideal))
    espectro_real_db = 10 * np.log10(espectro_real / np.max(espectro_ideal))

    fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=100)
    ax3.set_facecolor("#f8f8f8")

    ax3.axvspan(
        -0.50,
        0.50,
        alpha=0.10,
        color="#378ADD",
        label="Banda útil da BS"
    )

    ax3.axvspan(
        -0.80,
        -0.50,
        alpha=0.12,
        color="#E24B4A",
        label="Bandas de guarda"
    )

    ax3.axvspan(
        0.50,
        0.80,
        alpha=0.12,
        color="#E24B4A"
    )

    ax3.axvspan(
        -1.20,
        -0.80,
        alpha=0.06,
        color="gray",
        label="Região adjacente"
    )

    ax3.axvspan(
        0.80,
        1.20,
        alpha=0.06,
        color="gray"
    )

    ax3.plot(
        freq,
        espectro_ideal_db,
        linestyle="--",
        linewidth=2,
        color="#444444",
        label="Espectro ideal"
    )

    ax3.plot(
        freq,
        espectro_real_db,
        linewidth=2.2,
        color="#E24B4A",
        label="Espectro real com PA não-linear"
    )

    ax3.annotate(
        "Produto de 3ª ordem",
        xy=(-0.72, espectro_real_db[np.argmin(np.abs(freq + 0.72))]),
        xytext=(-1.05, -12),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=9
    )

    ax3.annotate(
        "Produto de 3ª ordem",
        xy=(0.72, espectro_real_db[np.argmin(np.abs(freq - 0.72))]),
        xytext=(0.82, -12),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=9
    )

    ax3.text(
        0,
        -4,
        "Banda útil",
        ha="center",
        fontsize=9,
        color="#1a5fa8"
    )

    ax3.text(
        -0.65,
        -35,
        "Guarda",
        ha="center",
        fontsize=9,
        color="#a32d2d"
    )

    ax3.text(
        0.65,
        -35,
        "Guarda",
        ha="center",
        fontsize=9,
        color="#a32d2d"
    )

    ax3.set_xlabel("Frequência normalizada")
    ax3.set_ylabel("Potência relativa [dB]")
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-80, 5)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc="lower center", ncol=2)

    st.pyplot(fig3, width="stretch")
    plt.close(fig3)