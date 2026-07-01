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

rms_delay   = st.sidebar.slider("RMS Delay Spread (ns)", 10, 1000, 150)
noise_floor = st.sidebar.slider("Limiar de ruído (dB)", -120, -5, -90)

# -------------------------------
# CONVERSÕES FÍSICAS
# -------------------------------
v  = v_kmh / 3.6
c  = 3e8
fc = 4e9 if "Micro" in freq_band else 40e9
f_doppler = v * fc / c

# Referência fixa para comparação: 60 km/h, 4 GHz
V_REF  = 60 / 3.6
FC_REF = 4e9
FD_REF = max(V_REF * FC_REF / c, 0.1)

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
# GERADOR DE CANAL RAYLEIGH
# -------------------------------
def gera_canal_rayleigh(fd_sim, seed=42, N_sin=40, N_samples=6000, T_total=None):
    if T_total is None:
        # Mantém o mesmo eixo temporal para comparação com a referência
        T_total = 6.0 / FD_REF

    t = np.linspace(0, T_total, N_samples)

    rng = np.random.default_rng(seed)
    phi_n   = rng.uniform(0, 2*np.pi, N_sin)
    theta_n = rng.uniform(0, 2*np.pi, N_sin)
    alpha_n = 2*np.pi*np.arange(1, N_sin + 1) / N_sin

    alpha_col = alpha_n[:, None]
    t_row     = t[None, :]

    I = np.sum(
        np.cos(2*np.pi*fd_sim*np.cos(alpha_col)*t_row + phi_n[:, None]),
        axis=0
    ) / np.sqrt(N_sin)

    Q = np.sum(
        np.sin(2*np.pi*fd_sim*np.cos(alpha_col)*t_row + theta_n[:, None]),
        axis=0
    ) / np.sqrt(N_sin)

    h = I + 1j * Q
    z = np.abs(h)
    return t, h, z

# -------------------------------
# STFT MANUAL
# -------------------------------
def compute_stft(x, fs, nperseg=256, noverlap=192):
    step = nperseg - noverlap
    window = np.hanning(nperseg)

    frames = []
    times = []

    for start in range(0, len(x) - nperseg + 1, step):
        segment = x[start:start + nperseg] * window
        X = np.fft.fftshift(np.fft.fft(segment, n=nperseg))
        frames.append(X)
        times.append((start + nperseg / 2) / fs)

    S = np.array(frames).T
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1/fs))
    times = np.array(times)

    return times, freqs, S

# -------------------------------
# SELEÇÃO DO GRÁFICO
# -------------------------------
grafico_ui = st.selectbox(
    "Visualização",
    [
        "PDP – Power Delay Profile",
        "LCR / AFD – Envoltória temporal com cruzamentos de nível",
        "Espetrograma 2D/3D (STFT) – variabilidade temporal induzida por f_m"
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
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)

# ================================
# PLOT LCR / AFD
# ================================
elif grafico_ui == "LCR / AFD – Envoltória temporal com cruzamentos de nível":
    fd = max(f_doppler, 0.1)

    t, h_cur, z_cur = gera_canal_rayleigh(fd, seed=42)
    t_ms = t * 1e3

    sigma_cur = np.sqrt(np.mean(z_cur**2) / 2)

    rho_slider = st.slider(
        "Nível normalizado ρ = Z / √(2σ²)",
        min_value=0.10, max_value=2.0, value=0.70, step=0.05
    )
    rho_val = rho_slider
    Z_cur = rho_val * np.sqrt(2) * sigma_cur

    LCR_cur = np.sqrt(2*np.pi) * fd * rho_val * np.exp(-rho_val**2)
    AFD_cur = (np.exp(rho_val**2) - 1) / (np.sqrt(2*np.pi) * fd * rho_val + 1e-30)

    above_cur      = z_cur >= Z_cur
    cross_down_cur = np.where(np.diff(above_cur.astype(int)) == -1)[0]
    fade_start_cur = np.where(np.diff(above_cur.astype(int)) == -1)[0]
    fade_end_cur   = np.where(np.diff(above_cur.astype(int)) ==  1)[0]

    if len(fade_end_cur) > 0 and len(fade_start_cur) > 0 and fade_end_cur[0] < fade_start_cur[0]:
        fade_end_cur = fade_end_cur[1:]

    n_pairs_cur = min(len(fade_start_cur), len(fade_end_cur))

    fade_dur_cur = (
        t[fade_end_cur[:n_pairs_cur]] - t[fade_start_cur[:n_pairs_cur]]
        if n_pairs_cur > 0 else np.array([])
    )

    LCR_med_cur = len(cross_down_cur) / (t[-1] - t[0])
    AFD_med_cur = np.mean(fade_dur_cur) * 1e3 if len(fade_dur_cur) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(t_ms, z_cur, color="#1a5fa8", linewidth=1.0, label=r"$z(t)$ atual")
    ax.axhline(Z_cur, color="crimson", linewidth=1.8, linestyle="-",
               label=f"Limiar $Z$ atual (ρ={rho_val:.2f})")

    ax.fill_between(t_ms, z_cur, Z_cur, where=(z_cur < Z_cur),
                    color="crimson", alpha=0.15)

    for idx in cross_down_cur[:20]:
        ax.annotate("", xy=(t_ms[idx], Z_cur - 0.04),
                    xytext=(t_ms[idx], Z_cur + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color="crimson",
                                    lw=1.3, mutation_scale=8))

    fade_start_cur = fade_start_cur[:n_pairs_cur]
    fade_end_cur   = fade_end_cur[:n_pairs_cur]

    for i in range(min(2, n_pairs_cur)):
        t0 = t_ms[fade_start_cur[i]]
        t1 = t_ms[fade_end_cur[i]]
        y_arr = Z_cur * 0.35
        ax.annotate("", xy=(t1, y_arr), xytext=(t0, y_arr),
                    arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
        ax.text((t0+t1)/2, y_arr - 0.05,
                rf"$t_{{z,{i+1}}}$",
                ha="center", va="top", fontsize=10, color="crimson")

    ax.text(-0.012 * t_ms[-1], Z_cur, "$Z$",
            ha="right", va="center", fontsize=11, color="crimson", fontweight="bold")

    ax.set_xlabel("$t$ (ms)", fontsize=12)
    ax.set_ylabel("$z(t)$", fontsize=12)
    ax.set_xlim(t_ms[0], t_ms[-1])

    legend_elements = [
        Line2D([0],[0], color="#1a5fa8", lw=1.0, label=r"$z(t)$"),
        Line2D([0],[0], color="crimson", lw=1.8, label=f"Limiar $Z$ (ρ={rho_val:.2f})"),
        Line2D([0],[0], color="crimson", marker='v', linestyle='None', markersize=8,
               label="LCR – Level Crossing Rate"),
        Line2D([0],[0], color="crimson", lw=1.4,
               label=r"$t_{z,i}$: AFD – Average Fade Duration"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.subplots_adjust(top=0.85)

    st.pyplot(fig)

# ================================
# PLOT STFT 2D / 3D
# ================================
else:
    fd = max(f_doppler, 0.1)

    modo_stft = st.radio(
        "Modo do espetrograma",
        ["2D", "3D"],
        horizontal=True
    )

    # Tempo de observação ajustado ao Doppler atual
    T_total_stft = np.clip(10.0 / fd, 0.15, 2.0)
    t_stft, h_stft, z_stft = gera_canal_rayleigh(
        fd_sim=fd,
        seed=42,
        N_samples=4096,
        T_total=T_total_stft
    )

    fs = 1.0 / (t_stft[1] - t_stft[0])

    # Normaliza o canal complexo
    x = h_stft / np.sqrt(np.mean(np.abs(h_stft)**2) + 1e-12)

    t_frames, freqs, S = compute_stft(x, fs, nperseg=256, noverlap=192)

    S_mag = np.abs(S)
    S_db = 20 * np.log10(S_mag + 1e-12)
    S_db = S_db - np.max(S_db)

    # Foco visual na faixa Doppler relevante
    f_lim = max(1.5 * fd, 20.0)
    mask_f = (freqs >= -f_lim) & (freqs <= f_lim)
    freqs_plot = freqs[mask_f]
    S_db_plot = S_db[mask_f, :]
    t_frames_ms = t_frames * 1e3

    if modo_stft == "2D":
        fig, ax = plt.subplots(figsize=(11, 5.5))

        im = ax.imshow(
            S_db_plot,
            aspect="auto",
            origin="lower",
            extent=[t_frames_ms[0], t_frames_ms[-1], freqs_plot[0], freqs_plot[-1]],
            cmap="viridis"
        )

        ax.axhline(fd, color="white", linestyle="--", linewidth=1.2, label=r"$+f_m$")
        ax.axhline(-fd, color="white", linestyle="--", linewidth=1.2, label=r"$-f_m$")

        ax.set_title("Espetrograma 2D (STFT) do canal")
        ax.set_xlabel("Tempo (ms)")
        ax.set_ylabel("Frequência Doppler (Hz)")
        ax.legend(loc="upper right")
        ax.grid(False)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Magnitude normalizada (dB)")

        st.pyplot(fig)

    else:
        fig = plt.figure(figsize=(11, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        T_grid, F_grid = np.meshgrid(t_frames_ms, freqs_plot)

        # Redução para deixar o gráfico 3D mais leve
        step_f = max(1, len(freqs_plot) // 80)
        step_t = max(1, len(t_frames_ms) // 80)

        T_s = T_grid[::step_f, ::step_t]
        F_s = F_grid[::step_f, ::step_t]
        S_s = S_db_plot[::step_f, ::step_t]

        surf = ax.plot_surface(T_s, F_s, S_s, cmap="viridis", linewidth=0, antialiased=True)

        ax.set_title("Espetrograma 3D (STFT) do canal")
        ax.set_xlabel("Tempo (ms)")
        ax.set_ylabel("Frequência Doppler (Hz)")
        ax.set_zlabel("Magnitude normalizada (dB)")

        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="Magnitude (dB)")

        st.pyplot(fig)