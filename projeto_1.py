import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math
import matplotlib.cm as cm
from matplotlib.colors import Normalize


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
    directions = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    cells = [(0, 0)]
    seen = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < N:
        nxt = []
        for ci, cj in frontier:
            for di, dj in directions:
                nb = (ci+di, cj+dj)
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
    params = {3: (1,1), 4: (2,0), 7: (2,1)}
    i0, j0 = params[N]

    def rot60(i, j):
        return (-j, i + j)

    translations = []
    vi, vj = i0, j0
    for _ in range(6):
        translations.append((vi, vj))
        vi, vj = rot60(vi, vj)
    return translations

# ── Entradas ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Cobertura Celular", page_icon="📡")

with st.sidebar:
    st.header("Parâmetros de Entrada")
    R     = st.number_input("Raio da célula R (km)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
    N     = st.selectbox("Tamanho do cluster N", [3, 4, 7], index=2)
    S     = st.number_input(" Número de canais totais", min_value=1, max_value=2000, value=395, step=1)
    A     = st.number_input("Tráfego oferecido (Erlangs)", min_value=0.1, max_value=5000.0, value=100.0, step=1.0)
    alpha = st.slider("Não-linearidade do PA da BS (fator de compressão de ganho)", 1.0, 6.0, 3.0, 0.5)

# ── Criar aba ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Malha Hexagonal", "REM CCI"])

with tab1:
    # ── Malha hexagonal ───────────────────────────────────────────────
    cluster_cells = get_cluster_cells(N)
    cluster_set   = set(cluster_cells)
    translations  = get_cochannel(N)
    grid_centers  = build_grid(8, R)

    COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]

    cluster_index = {cell: idx + 1 for idx, cell in enumerate(cluster_cells)}
    cell_color = {cell: COLORS[idx % len(COLORS)] for idx, cell in enumerate(cluster_cells)}

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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f5f5f5")
    hex_r = R * 0.97

    for (i, j), (cx, cy) in grid_centers.items():

        if (i, j) in cluster_set:
            fc = cell_color[(i, j)]
            ec = "white"
            lw = 1.8

            patch = RegularPolygon(
                (cx, cy), 6, radius=hex_r,
                orientation=0, facecolor=fc,
                edgecolor=ec, linewidth=lw
            )
            ax.add_patch(patch)

            idx = cluster_index[(i, j)]
            ax.text(cx, cy, str(idx), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

        elif (i, j) in cochannel_map:
            fc = cochannel_color[(i, j)]
            ec = "red"
            lw = 2.0

            patch = RegularPolygon(
                (cx, cy), 6, radius=hex_r,
                orientation=0, facecolor=fc,
                edgecolor=ec, linewidth=lw, alpha=0.45
            )
            ax.add_patch(patch)

            ax.text(cx, cy, str(cochannel_map[(i, j)]), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='red')

        else:
            continue

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]
    pad = R * 1.5
    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
    ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_handles = []
    for idx, cell in enumerate(cluster_cells):
        legend_handles.append(
            mpatches.Patch(
                facecolor=COLORS[idx % len(COLORS)],
                edgecolor="white",
                label=f"Célula {idx + 1} (cluster / co-canal)"
            )
        )

    legend_handles.append(
        mpatches.Patch(
            facecolor="white",
            edgecolor="red",
            linewidth=2,
            label="Célula co-canal do cluster"
        )
    )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#aaaaaa")

    st.pyplot(fig, width='stretch')
    plt.close(fig)

with tab2:
    st.subheader("REM CCI por célula de referência")

    # escolhe qual célula do cluster será analisada
    ref_label = st.selectbox(
        "Escolha a célula de referência para analisar a CCI",
        list(range(1, N + 1)),
        index=0
    )

    cluster_cells = get_cluster_cells(N)
    cluster_set   = set(cluster_cells)
    translations  = get_cochannel(N)
    grid_centers  = build_grid(8, R)

    COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
    cluster_index = {cell: idx + 1 for idx, cell in enumerate(cluster_cells)}
    cell_color    = {cell: COLORS[idx % len(COLORS)] for idx, cell in enumerate(cluster_cells)}

    # mapa das células co-canais
    cochannel_map = {}
    for cell in cluster_cells:
        ci, cj = cell
        for ti, tj in translations:
            pos = (ci + ti, cj + tj)
            if pos not in cluster_set:
                cochannel_map[pos] = cluster_index[cell]

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap_cci = plt.colormaps["YlOrRd"]

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_facecolor("#f5f5f5")
    hex_r = R * 0.97

    for (i, j), (cx, cy) in grid_centers.items():

        # células do cluster principal
        if (i, j) in cluster_set:
            label = cluster_index[(i, j)]

            # destaca a célula de referência
            if label == ref_label:
                edgecolor = "black"
                linewidth = 2.5
            else:
                edgecolor = "white"
                linewidth = 1.8

            patch = RegularPolygon(
                (cx, cy), 6, radius=hex_r, orientation=0,
                facecolor=cell_color[(i, j)],
                edgecolor=edgecolor, linewidth=linewidth
            )
            ax2.add_patch(patch)

            ax2.text(
                cx, cy, str(label),
                ha='center', va='center',
                fontsize=9, fontweight='bold', color='white'
            )

        # células co-canais
        elif (i, j) in cochannel_map:
            label = cochannel_map[(i, j)]

            # se for co-canal da célula escolhida -> alta CCI
            if label == ref_label:
                heat = 1.0
                fc = cmap_cci(0.95)
                edge = "red"
                lw = 2.2
                alpha_patch = 0.95
                txt_color = "white"
            else:
                # outras células co-canais ficam com baixa intensidade visual
                heat = 0.15
                fc = cmap_cci(0.25)
                edge = "#ff9999"
                lw = 1.2
                alpha_patch = 0.45
                txt_color = "darkred"

            patch = RegularPolygon(
                (cx, cy), 6, radius=hex_r, orientation=0,
                facecolor=fc,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha_patch
            )
            ax2.add_patch(patch)

            ax2.text(
                cx, cy, str(label),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=txt_color
            )

        else:
            continue

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]
    pad   = R * 1.5
    ax2.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax2.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # barra de cores
    sm = ScalarMappable(cmap=cmap_cci, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label(
        f"Intensidade de CCI relativa à célula {ref_label}",
        fontsize=9
    )

    # legenda
    legend_handles = []
    for idx, cell in enumerate(cluster_cells):
        legend_handles.append(
            mpatches.Patch(
                facecolor=COLORS[idx % len(COLORS)],
                edgecolor='white',
                label=f"Célula {idx + 1}"
            )
        )

    legend_handles.append(
        mpatches.Patch(
            facecolor=cmap_cci(0.95),
            edgecolor='red',
            linewidth=2,
            label=f"Alta CCI da célula {ref_label}"
        )
    )

    legend_handles.append(
        mpatches.Patch(
            facecolor=cmap_cci(0.25),
            edgecolor='#ff9999',
            linewidth=1,
            label="Outros co-canais"
        )
    )

    ax2.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
        edgecolor="#aaaaaa"
    )

    st.pyplot(fig2, width='stretch')
    plt.close(fig2)