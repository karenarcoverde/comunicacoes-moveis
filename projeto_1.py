import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math

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

# ── Malha hexagonal ───────────────────────────────────────────────────
cluster_cells = get_cluster_cells(N)
cluster_set   = set(cluster_cells)
translations  = get_cochannel(N)

cochannel_set = set()
for ti, tj in translations:
    cochannel_set.add((ti, tj))

grid_centers = build_grid(8, R)

COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]
cell_color = {cell: COLORS[i % len(COLORS)] for i, cell in enumerate(cluster_cells)}

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("#f5f5f5")

hex_r = R * 0.97

for (i, j), (cx, cy) in grid_centers.items():
    if (i, j) in cluster_set:
        fc = cell_color[(i, j)]
        ec = "white"
        lw = 1.8
    elif (i, j) in cochannel_set:
        fc = cell_color.get((0, 0), COLORS[0])
        ec = "red"
        lw = 2.0
        patch = RegularPolygon((cx, cy), 6, radius=hex_r,
                               orientation=0, facecolor=fc,
                               edgecolor=ec, linewidth=lw, alpha=0.4)
        ax.add_patch(patch)
        ax.text(cx, cy, "CC", ha='center', va='center',
                fontsize=7, color='red', fontweight='bold')
        continue
    else:
        fc = "#cccccc"
        ec = "#999999"
        lw = 0.5

    patch = RegularPolygon((cx, cy), 6, radius=hex_r,
                           orientation=0, facecolor=fc,
                           edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)

    if (i, j) in cluster_set:
        idx = cluster_cells.index((i, j)) + 1
        ax.text(cx, cy, str(idx), ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

all_x = [v[0] for v in grid_centers.values()]
all_y = [v[1] for v in grid_centers.values()]
pad = R * 1.5
ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
ax.set_aspect('equal')
ax.axis('off')


# ── Legenda ───────────────────────────────────────────────────────────
legend_handles = []
for idx, cell in enumerate(cluster_cells):
    legend_handles.append(
        mpatches.Patch(facecolor=COLORS[idx % len(COLORS)], edgecolor="white",
                       label=f"Célula {idx + 1} (cluster)")
    )
legend_handles.append(
    mpatches.Patch(facecolor=cell_color.get((0, 0), COLORS[0]), edgecolor="red",
                   linewidth=2, alpha=0.4, label="Célula co-canal (CC)")
)
legend_handles.append(
    mpatches.Patch(facecolor="#cccccc", edgecolor="#999999", label="Outras células")
)
ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor="#aaaaaa")

st.pyplot(fig, width='stretch')
plt.close(fig)