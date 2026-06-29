import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Desvanecimento 3GPP 5G",
    page_icon="📡",
    layout="centered"
)

# -------------------------------
# ENTRADAS SIDEBAR
# -------------------------------
v_kmh = st.sidebar.slider("Velocidade do terminal (km/h)", 1, 300, 60)

freq_band = st.sidebar.selectbox(
    "Faixa de frequência",
    ["Micro-ondas (3–30 GHz)", "Milimétricas (30–300 GHz)"]
)

channel_ui = st.sidebar.selectbox(
    "Perfil de canal 3GPP",
    ["TDL-A", "TDL-B", "TDL-C"]
)

rms_delay  = st.sidebar.slider("RMS Delay Spread (ns)", 10, 1000, 150)
noise_floor = st.sidebar.slider("Limiar de ruído (dB)", -120, -5, -90)

# -------------------------------
# CONVERSÕES FÍSICAS
# -------------------------------
v = v_kmh / 3.6
c = 3e8
fc = 4e9 if "Micro" in freq_band else 40e9
f_doppler = v * fc / c

# -------------------------------
# TDL REAL (3GPP TR 38.901)
# -------------------------------
tdl_profiles = {
    "TDL-A": {
        "delays": np.array([0.0000,0.3819,0.4025,0.5868,0.4610,0.5375,0.6708,
                            0.5750,0.7618,1.5375,1.8978,2.2242,2.1718,2.4942,
                            2.5119,3.0582,4.0810,4.4579,4.5695,4.7966,5.0066,
                            5.3043,9.6586]),
        "powers": np.array([-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,
                             -6.6,-16.7,-12.4,-15.2,-10.8,-11.3,-12.7,-16.2,
                             -18.3,-18.9,-16.6,-19.9,-29.7])
    },
    "TDL-B": {
        "delays": np.array([0.0000,0.1072,0.2155,0.2095,0.2870,0.2986,0.3752,
                            0.5055,0.3681,0.3697,0.5700,0.5283,1.1021,1.2756,
                            1.5474,1.7842,2.0169,2.8294,3.0219,3.6187,4.1067,
                            4.2790,4.7834]),
        "powers": np.array([0,-2.2,-4,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3,
                             -8.9,-9,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,
                             -11.4,-14.9,-9.2,-11.3])
    },
    "TDL-C": {
        "delays": np.array([0.0000,0.2099,0.2219,0.2329,0.2176,0.6366,0.6448,
                            0.6560,0.6584,0.7935,0.8213,0.9336,1.2285,1.3083,
                            2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,
                            6.6374,7.0427,8.6523]),
        "powers": np.array([-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,
                             -10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,
                             -15.8,-17.1,-16,-15.7,-21.6,-22.8])
    }
}

profile    = tdl_profiles[channel_ui]
delays     = profile["delays"].astype(float)
powers_db  = profile["powers"].astype(float)

def compute_rms_delay(delays, powers_db):
    pl = 10 ** (powers_db / 10)
    pl = pl / np.sum(pl)
    mean  = np.sum(pl * delays)
    mean2 = np.sum(pl * delays**2)
    return np.sqrt(mean2 - mean**2)

current_rms = compute_rms_delay(delays, powers_db)
scale       = rms_delay / (current_rms + 1e-12)
delays      = delays * scale

powers_linear = 10 ** (powers_db / 10)
pdp    = powers_linear / np.max(powers_linear)
pdp_db = 10 * np.log10(pdp + 1e-12)

# -------------------------------
# SELEÇÃO DO GRÁFICO
# -------------------------------
grafico_ui = st.selectbox(
    "Visualização",
    [
        "PDP – Power Delay Profile",
        "LCR / AFD – Envoltória temporal com cruzamentos de nível"
    ]
)

# ================================
# PLOT PDP
# ================================
if grafico_ui == "PDP – Power Delay Profile":
    fig, ax = plt.subplots()
    ax.stem(delays, pdp_db, label=channel_ui)
    ax.axhline(noise_floor, color="red", linestyle="--", label="Limiar de Ruído")
    ax.set_title(f"PDP - {channel_ui}")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Potência (dB)")
    ax.legend()
    st.pyplot(fig)

# ================================
# PLOT ENVOLTÓRIA TEMPORAL (LCR/AFD)
# ================================
else:
    fd = max(f_doppler, 0.1)

    # --------------------------------------------------
    # Gera envoltória de Rayleigh – modelo de Clarke
    # z(t) = |β(t)|,  E{z²} = 2σ²  →  σ = 1/√2
    # --------------------------------------------------
    np.random.seed(42)
    N_sin     = 40
    N_samples = 6000
    sigma     = 1.0 / np.sqrt(2)          # E{z²} = 2σ² = 1

    t = np.linspace(0, 6.0 / fd, N_samples)   # ~6 ciclos Doppler

    phi_n = np.random.uniform(0, 2*np.pi, N_sin)
    theta_n = np.random.uniform(0, 2*np.pi, N_sin)
    alpha_n = 2*np.pi*np.arange(1, N_sin+1) / N_sin   # ângulos de chegada uniformes

    I = np.sum([np.cos(2*np.pi*fd*np.cos(alpha_n[k])*t + phi_n[k])
                for k in range(N_sin)], axis=0) / np.sqrt(N_sin)
    Q = np.sum([np.sin(2*np.pi*fd*np.cos(alpha_n[k])*t + theta_n[k])
                for k in range(N_sin)], axis=0) / np.sqrt(N_sin)

    z = np.sqrt(I**2 + Q**2)              # envoltória linear z(t) = |β(t)|

    # σ² estimado da simulação
    sigma2_est = np.mean(z**2) / 2        # E{z²} = 2σ²
    sigma_est  = np.sqrt(sigma2_est)

    # --------------------------------------------------
    # Limiar Z  (como fração do RMS da envoltória)
    # ρ = Z / √(2σ²) = Z / √(E{z²})
    # Permite o usuário escolher ρ via slider
    # --------------------------------------------------
    rho_slider = st.slider(
        "Nível normalizado  ρ = Z / √(2σ²)",
        min_value=0.10, max_value=2.0, value=0.70, step=0.05
    )

    Z_thresh = rho_slider * np.sqrt(2) * sigma_est   # limiar linear

    # --------------------------------------------------
    # LCR teórico (fórmula do slide)
    # L_z = √(2π) · f_D · ρ · exp(−ρ²)
    # --------------------------------------------------
    rho_val  = rho_slider
    LCR_teo  = np.sqrt(2*np.pi) * fd * rho_val * np.exp(-rho_val**2)

    # AFD teórico (fórmula do slide)
    # t̄_z = σ / (Z · f_D · √π) · [exp(Z²/2σ²) − 1]
    #       = [exp(ρ²) − 1] / (√(2π) · f_D · ρ)
    AFD_teo  = (np.exp(rho_val**2) - 1) / (np.sqrt(2*np.pi) * fd * rho_val + 1e-30)

    # --------------------------------------------------
    # Cruzamentos descendentes  (z desce abaixo de Z)
    # --------------------------------------------------
    above = z >= Z_thresh
    cross_down_idx = np.where(np.diff(above.astype(int)) == -1)[0]

    # Durações de fade
    fade_start = np.where(np.diff(above.astype(int)) == -1)[0]
    fade_end   = np.where(np.diff(above.astype(int)) ==  1)[0]
    if len(fade_end) > 0 and len(fade_start) > 0 and fade_end[0] < fade_start[0]:
        fade_end = fade_end[1:]
    n_pairs = min(len(fade_start), len(fade_end))
    fade_start = fade_start[:n_pairs]
    fade_end   = fade_end[:n_pairs]
    fade_dur   = t[fade_end] - t[fade_start] if n_pairs > 0 else np.array([])

    LCR_med = len(cross_down_idx) / (t[-1] - t[0])
    AFD_med = np.mean(fade_dur) * 1e3 if len(fade_dur) > 0 else 0.0

    # --------------------------------------------------
    # Eixo de tempo em ms
    # --------------------------------------------------
    t_ms = t * 1e3

    # --------------------------------------------------
    # FIGURA  –  estilo do slide
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Envoltória z(t)
    ax.plot(t_ms, z, color="#1a5fa8", linewidth=1.0, label=r"$z(t) = |\beta(t)|$")

    # Linha do limiar Z
    ax.axhline(Z_thresh, color="crimson", linewidth=1.8, linestyle="-",
               label=f"Limiar $Z$  (ρ = {rho_val:.2f})")

    # Sombreia fades (z < Z)
    ax.fill_between(t_ms, z, Z_thresh,
                    where=(z < Z_thresh),
                    color="crimson", alpha=0.18)

    z_max    = z.max()
    y_top    = z_max * 1.15   # teto do eixo y

    # Cruzamentos descendentes → pequena seta ↓ sobre o limiar
    for idx in cross_down_idx[:25]:
        ax.annotate("", xy=(t_ms[idx], Z_thresh - 0.04),
                    xytext=(t_ms[idx], Z_thresh + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color="crimson",
                                   lw=1.3, mutation_scale=8))

    # Anota t_{z,i} nos dois primeiros fades (igual ao slide)
    for i in range(min(2, n_pairs)):
        t0 = t_ms[fade_start[i]]
        t1 = t_ms[fade_end[i]]
        y_arr = Z_thresh * 0.40
        ax.annotate("", xy=(t1, y_arr), xytext=(t0, y_arr),
                    arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
        ax.text((t0+t1)/2, y_arr - 0.06,
                rf"$t_{{z,{i+1}}}$",
                ha="center", va="top", fontsize=10, color="crimson")

    # Rótulo Z no eixo y
    ax.text(-0.012 * t_ms[-1], Z_thresh, "$Z$",
            ha="right", va="center", fontsize=12, color="crimson", fontweight="bold")

    ax.set_xlabel("$t$ (ms)", fontsize=12)
    ax.set_ylabel("$z(t)$", fontsize=12)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(0, y_top)
    # Legenda customizada
    legend_elements = [
        Line2D([0], [0], color="#1a5fa8", lw=1.0, label=r"$z(t)=|\beta(t)|$"),
        Line2D([0], [0], color="crimson", lw=1.8,
            label=f"Limiar $Z$  ($\\rho$ = {rho_val:.2f})"),
        Line2D([0], [0], color="crimson", marker='v', linestyle='None', markersize=8,
            label="Level Crossing Rate (LCR)"),
        Line2D([0], [0], color="crimson", lw=1.4,
            label=r"$t_{z,i}$: Average Fade Duration (AFD)")
    ]

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.subplots_adjust(top=0.70)
    st.pyplot(fig)
