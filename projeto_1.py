import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path

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
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
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
    params = {3: (1, 1), 4: (2, 0), 7: (2, 1)}
    i0, j0 = params[N]

    def rot60(i, j):
        return (-j, i + j)

    translations = []
    vi, vj = i0, j0
    for _ in range(6):
        translations.append((vi, vj))
        vi, vj = rot60(vi, vj)
    return translations

# ── Funções auxiliares para SIR ponto a ponto ──────────────────────────
def calc_sir_db(user_pos, serving_pos, interferer_positions, n_path, d0=0.001):
    ux, uy = user_pos
    sx, sy = serving_pos
    ds = max(math.sqrt((ux - sx)**2 + (uy - sy)**2), d0)
    Ps = ds**(-n_path)
    I_total = sum(max(math.sqrt((ux - ix)**2 + (uy - iy)**2), d0)**(-n_path) for ix, iy in interferer_positions)
    sir_linear = Ps / (I_total + 1e-12)
    return 10 * math.log10(sir_linear)

def hex_vertices(cx, cy, radius, orientation=0):
    angles = orientation + np.linspace(0, 2*np.pi, 7)[:-1]
    return np.column_stack([cx + radius*np.cos(angles), cy + radius*np.sin(angles)])

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
    R = st.number_input("Raio da célula R (km)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
    N = st.selectbox("Tamanho do cluster N", [3, 4, 7], index=2)
    S = st.number_input("Número de canais totais", min_value=1, max_value=2000, value=395, step=1)
    A = st.number_input("Tráfego oferecido (Erlangs)", min_value=0.1, max_value=5000.0, value=100.0, step=1.0)
    alpha = st.slider("Não-linearidade do PA da BS (fator de compressão de ganho)", 1.0, 6.0, 3.0, 0.5)

tab1, tab2 = st.tabs(["Malha Hexagonal", "REM CCI"])
n_path = 4 
# ─────────────────────────────────────────────────────────────
# TAB 1 — Malha hexagonal
# ─────────────────────────────────────────────────────────────
with tab1:
    cluster_cells = get_cluster_cells(N)
    cluster_set = set(cluster_cells)
    translations = get_cochannel(N)
    grid_centers = build_grid(8, R)

    COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]
    cluster_index = {cell: idx+1 for idx, cell in enumerate(cluster_cells)}
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
        if (i,j) in cluster_set:
            fc = cell_color[(i,j)]
            patch = RegularPolygon((cx,cy),6,radius=hex_r,orientation=0,facecolor=fc,edgecolor="white",linewidth=1.8)
            ax.add_patch(patch)
            ax.text(cx, cy, str(cluster_index[(i,j)]), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        elif (i,j) in cochannel_map:
            fc = cochannel_color[(i,j)]
            patch = RegularPolygon((cx,cy),6,radius=hex_r,orientation=0,facecolor=fc,edgecolor="red",linewidth=2.0,alpha=0.45)
            ax.add_patch(patch)
            ax.text(cx, cy, str(cochannel_map[(i,j)]), ha="center", va="center", fontsize=9, fontweight="bold", color="red")

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]
    pad = R*1.5
    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
    ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = [mpatches.Patch(facecolor=COLORS[idx%len(COLORS)], edgecolor="white", label=f"Célula {idx+1} (cluster / co-canal)") for idx in range(len(cluster_cells))]
    legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="red", linewidth=2, label="Célula co-canal do cluster"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#aaaaaa")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# TAB 2 — REM CCI com SIR ponto a ponto
# ─────────────────────────────────────────────────────────────
with tab2:
    cluster_cells = get_cluster_cells(N)
    cluster_set = set(cluster_cells)
    translations = get_cochannel(N)
    grid_centers = build_grid(8, R)

    COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]
    cluster_index = {cell: idx+1 for idx, cell in enumerate(cluster_cells)}
    cell_color = {cell: COLORS[idx % len(COLORS)] for idx, cell in enumerate(cluster_cells)}

    cochannel_map = {}
    for cell in cluster_cells:
        ci, cj = cell
        for ti, tj in translations:
            pos = (ci + ti, cj + tj)
            if pos not in cluster_set:
                cochannel_map[pos] = cluster_index[cell]

    visible_cells = list(cluster_set) + list(cochannel_map.keys())
    cell_label_map = {}
    for cell in cluster_set:
        cell_label_map[cell] = cluster_index[cell]
    for cell, label in cochannel_map.items():
        cell_label_map[cell] = label

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_facecolor("#f5f5f5")
    hex_r = R * 0.97

    # Desenha malha de fundo
    for (i,j),(cx,cy) in grid_centers.items():
        if (i,j) in cluster_set:
            patch = RegularPolygon((cx,cy),6,radius=hex_r,orientation=0,facecolor=cell_color[(i,j)],edgecolor="white",linewidth=1.8,alpha=0.35)
            ax2.add_patch(patch)
            ax2.text(cx,cy,str(cluster_index[(i,j)]),ha="center",va="center",fontsize=9,fontweight="bold",color="black")
        elif (i,j) in cochannel_map:
            patch = RegularPolygon((cx,cy),6,radius=hex_r,orientation=0,facecolor="white",edgecolor="red",linewidth=1.2,alpha=0.25)
            ax2.add_patch(patch)
            ax2.text(cx,cy,str(cochannel_map[(i,j)]),ha="center",va="center",fontsize=8,fontweight="bold",color="red")

    all_x = [v[0] for v in grid_centers.values()]
    all_y = [v[1] for v in grid_centers.values()]
    x_min = min(all_x) - R
    x_max = max(all_x) + R
    y_min = min(all_y) - R
    y_max = max(all_y) + R

    resolution = 220
    xs_grid = np.linspace(x_min, x_max, resolution)
    ys_grid = np.linspace(y_min, y_max, resolution)
    rem_x, rem_y, rem_sir = [], [], []

    for x in xs_grid:
        for y in ys_grid:
            serving_cell = find_serving_cell(x, y, visible_cells, grid_centers, hex_r)
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

            sir_db = calc_sir_db((x, y), serving_pos, interferer_positions, n_path)
            rem_x.append(x)
            rem_y.append(y)
            rem_sir.append(sir_db)

    rem_sir = np.array(rem_sir)
    if len(rem_sir) > 0:
        sir_vmin = np.percentile(rem_sir, 5)
        sir_vmax = np.percentile(rem_sir, 95)
    else:
        sir_vmin = 0
        sir_vmax = 50

    cmap_sir = plt.colormaps["RdYlGn"]
    norm_sir = Normalize(vmin=sir_vmin, vmax=sir_vmax)

    sc = ax2.scatter(rem_x, rem_y, c=rem_sir, cmap=cmap_sir, norm=norm_sir, s=8, marker="s", alpha=0.75)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect("equal")
    ax2.axis("off")

    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("SIR [dB]", fontsize=9)

    st.pyplot(fig2, width="stretch")
    plt.close(fig2)