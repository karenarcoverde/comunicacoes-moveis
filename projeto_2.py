import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem
import pandas as pd

st.set_page_config(
    page_title="Modelos de propagação",
    page_icon="📡",
    layout="centered"
)

# Parâmetros fixos no código — nunca aparecem no sidebar
PARAMS_FIXOS = pd.DataFrame([
    {"Parâmetro": "Potência de transmissão (Pt)", "Valor": "43 dBm",        "Modelo": "Todos"},
    {"Parâmetro": "Distância de referência (d₀)", "Valor": "1 m",           "Modelo": "Log-dist. / Indoor"},
    {"Parâmetro": "Expoente de perda (n)",        "Valor": "3",             "Modelo": "Log-distância"},
    {"Parâmetro": "Número de andares (floors)",   "Valor": "1",             "Modelo": "Indoor"},
    {"Parâmetro": "FAF (1 andar)",                "Valor": "12,9 dB",       "Modelo": "Indoor"},
    {"Parâmetro": "Expoente nsf (d > d₀)",        "Valor": "2,0",           "Modelo": "Indoor"},
    {"Parâmetro": "Altura Tx (h_tx)",             "Valor": "30 m",          "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Altura Rx (h_rx)",             "Valor": "1,5 m",         "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Altura edifícios (h_b)",       "Valor": "25 m",          "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Largura das ruas (w)",         "Valor": "20 m",          "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Espaç. entre edif. (b)",       "Valor": "20 m",          "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Ângulo de incidência (φ)",     "Valor": "30°",           "Modelo": "Walfisch-Bertoni"},
    {"Parâmetro": "Ganho das antenas (Gt = Gr)",  "Valor": "0 dBi",         "Modelo": "Todos"},
    {"Parâmetro": "BW de referência",             "Valor": "1 MHz",         "Modelo": "Todos"},
    {"Parâmetro": "Símbolos simulados",           "Valor": "4 000",         "Modelo": "Todos"},
])

# -------------------------------
# SIDEBAR (ENTRADAS)
# -------------------------------
model = st.sidebar.selectbox(
    "Seleção do modelo",
    ["Log-distância", "Walfisch-Bertoni", "Indoor"]
)

M = st.sidebar.selectbox(
    "Modulação QAM",
    [4, 16, 64]
)

if model == "Indoor":
    d_tx_rx = st.sidebar.slider("Distância Tx-Rx (m)", 1, 550, 100)
else:
    d_tx_rx = st.sidebar.slider("Distância Tx-Rx (m)", 1, 10000, 1000)
f = st.sidebar.slider("Frequência (GHz)", 0.5, 30.0, 2.0)
f = f * 1e9
sigma = st.sidebar.slider("Desvio padrão do sombreamento σ (dB)", 0.0, 12.0, 4.0)
noise_figure = st.sidebar.slider("Figura de ruído (dB)", 0.0, 10.0, 5.0)
awgn_db = st.sidebar.slider("AWGN (dB)", 0.0, 20.0, 10.0)
phase_noise_deg = st.sidebar.slider("Ruído de fase (graus)", 0.0, 20.0, 1.0)


# -------------------------------
# FUNÇÕES
# -------------------------------

def fspl(d, f):
    c = 3e8
    Gt = Gr = 1
    lam = c / f
    term = (Gt * Gr * lam**2) / ((4 * np.pi)**2 * d**2)
    PL = -10 * np.log10(term)
    return PL


def log_distance(d, d0, PL0, n):
    return PL0 + 10 * n * np.log10(d / d0)


def indoor_simple(d, f, floors=1):
    PL0 = fspl(1, f)
    d0 = 1
    nsf = np.where(d > d0, 2.0, 3.5)
    FAF_table = {1: 12.9, 2: 18.7, 3: 24.0, 4: 27.0}
    FAF = FAF_table.get(floors, 0)
    return PL0 + 10 * nsf * np.log10(d / d0) + FAF


def Ka(f_mhz, d_km, h_b, h_t):
    delta_hb = h_t - h_b
    mask_high_f = f_mhz > 2000
    mask_far = d_km >= 0.5
    mask_bh = h_b > h_t
    Ka_high_far = 54 - 0.8 * delta_hb
    Ka_high_near = 54 - 1.6 * delta_hb * d_km
    Ka_low_far = 73 - 0.8 * delta_hb
    Ka_low_near = 73 - 1.6 * delta_hb * d_km
    Ka_high = np.where(mask_far, Ka_high_far, Ka_high_near)
    Ka_low = np.where(mask_far, Ka_low_far, Ka_low_near)
    return np.where(mask_high_f,
                    np.where(mask_bh, 54, Ka_high),
                    np.where(mask_bh, 71.4, Ka_low))


def walfisch_ikegami(d, f, h_tx=30, h_rx=1.5, h_b=25, w=20, b=20, phi=30):
    d_km = d / 1000
    f_mhz = f / 1e6
    log_d = np.log10(d_km)
    log_f = np.log10(f_mhz)
    L0 = 32.4 + 20 * log_d + 20 * log_f
    Lori = np.where(phi <= 35,
                    -10 + 0.354 * phi,
                    np.where(phi <= 55,
                             2.5 + 0.075 * (phi - 35),
                             4 - 0.114 * (phi - 55)))
    delta_h = h_tx - h_rx
    Lrts = -8.2 - 10 * np.log10(w) + 20 * np.log10(delta_h) + Lori
    Lbsh = np.where(h_b > h_tx, -18 * np.log10(1 + delta_h), 0)
    Ka_val = Ka(f_mhz, d_km, h_b, h_tx)
    Kd = 18
    Kf = -4 + 0.7 * (f_mhz / 1000)
    Lmsd = Lbsh + Ka_val + Kd * log_d + Kf * log_f - 9 * np.log10(b)
    return np.where((Lrts + Lmsd) > 0, L0 + Lrts + Lmsd, L0)


@st.cache_data
def compute_PL(model, d_tx_rx, f, sigma):
    d = np.linspace(1, d_tx_rx, 150)
    if model == "Log-distância":
        PL0 = fspl(1, f)
        PL = log_distance(d, 1, PL0, n=3)
    elif model == "Indoor":
        PL = indoor_simple(d, f, floors=1)
    elif model == "Walfisch-Bertoni":
        PL = walfisch_ikegami(d, f, h_tx=30, h_rx=1.5, h_b=25, w=20, b=20, phi=30)
    shadow = np.random.normal(0, sigma, len(d))
    PL_shadow = PL + shadow
    return d, PL_shadow


d, PL = compute_PL(model, d_tx_rx, f, sigma)


@st.cache_data
def compute_constellation(PL_last, noise_figure, M, awgn_db, phase_noise_deg, n_symbols=4000):
    modem = QAMModem(M)
    bits = np.random.randint(0, 2, int(np.log2(M) * n_symbols))
    symbols_tx = modem.modulate(bits)

    I_tx = symbols_tx.real
    Q_tx = symbols_tx.imag

    # Link budget
    Pt_dBm = 43
    Pr_dBm = Pt_dBm - PL_last
    noise_dBm = -174 + 10 * np.log10(1e6) + noise_figure + awgn_db
    SNR_dB = Pr_dBm - noise_dBm
    SNR_linear = 10 ** (SNR_dB / 10)

    Es = np.mean(np.abs(symbols_tx) ** 2)
    noise_std = np.sqrt(Es / (2 * SNR_linear))

    # Canal AWGN
    I_rx = I_tx + np.random.normal(0, noise_std, len(I_tx))
    Q_rx = Q_tx + np.random.normal(0, noise_std, len(Q_tx))

    # Ruído de fase
    phase_noise_rad = np.deg2rad(phase_noise_deg)
    phi = np.random.normal(0, phase_noise_rad, len(I_rx))
    symbols_rx = (I_rx + 1j * Q_rx) * np.exp(1j * phi)
    I_rx = symbols_rx.real
    Q_rx = symbols_rx.imag

    # -----------------------------------------
    # Cálculo do EVM (%)
    # EVM = sqrt( mean(|s_rx - s_tx|²) / mean(|s_tx|²) ) × 100
    # -----------------------------------------
    symbols_rx_complex = I_rx + 1j * Q_rx
    erro = symbols_rx_complex - symbols_tx
    evm_percent = np.sqrt(np.mean(np.abs(erro) ** 2) / np.mean(np.abs(symbols_tx) ** 2)) * 100

    return (
        modem.constellation.real,
        modem.constellation.imag,
        I_rx,
        Q_rx,
        evm_percent,
        SNR_dB,
    )


def qam_limits(M):
    m = int(np.sqrt(M))
    lim = m - 1
    return -(lim + 1), (lim + 1)


# -------------------------------
# TABS
# -------------------------------
tab = st.radio(
    "Selecione a visualização",
    ["Curva de atenuação", "Constelação", "Cenário"],
    horizontal=True
)


# -------------------------------
# TAB 1
# -------------------------------
if tab == "Curva de atenuação":
     # Adiciona instruções de uso
    st.info(
        "💡 Como usar:\n"
        "- Esta aba exibe a perda de percurso (Path Loss) em função da distância entre transmissor e receptor.\n"
        "- Altere as entradas: Modelo de propagação, distância Tx-Rx, frequência e desvio padrão do sombreamento."
    )

    fig, ax = plt.subplots()
    ax.plot(d, PL, label=model)
    ax.set_xlabel("Distância (m)")
    ax.set_ylabel("PL (dB)")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# TAB 2
# -------------------------------
if tab=="Constelação":
     # Adiciona instruções de uso
    st.info(
        "💡 Como usar:\n"
        "- Diagrama de constelação simulado após o canal, mostrando o espalhamento dos símbolos recebidos em torno dos símbolos ideais.\n"
        "- Tx: símbolos ideais transmitidos, Rx: símbolos recebidos com ruído.\n"
        "- EVM (%) = Error Vector Magnitude.\n"
        "- Altere as entradas: Modelo de propagação, modulação M-QAM, distância Tx-Rx, frequência, desvio padrão do sombreamento, figura de ruído, AWGN e ruído de fase."
    )

    I_ideal, Q_ideal, I_rx, Q_rx, evm_percent, SNR_dB = compute_constellation(
        PL[-1], noise_figure, M, awgn_db, phase_noise_deg
    )

    # -----------------------------------------
    # Painel EVM
    # -----------------------------------------

    st.metric(
        label="EVM instantâneo",
        value=f"{evm_percent:.2f} %",
    )

    # -----------------------------------------
    # Plot da constelação
    # -----------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.scatter(I_rx, Q_rx, s=4, alpha=0.6, label="Rx")
    ax2.scatter(
        I_ideal, Q_ideal,
        s=45,
        facecolors="none",
        edgecolors="orange",
        linewidths=1.0,
        label="Tx"
    )
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.grid()

    lim = qam_limits(M)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.legend()

    st.pyplot(fig2)


# -------------------------------
# TAB 3 – Cenário
# -------------------------------
AMBIENTES = pd.DataFrame([
    {
        "Modelo":        "Log-distância",
        "Ambiente":      "Suburbano"
    },
    {
        "Modelo":        "Walfisch-Bertoni",
        "Ambiente":      "Urbano denso (macro-célula)"
    },
    {
        "Modelo":        "Indoor",
        "Ambiente":      "Ambiente interno (escritório)"
    },
])

# -------------------------------
# TAB 3 – Cenário
# -------------------------------
if tab == "Cenário":
    st.subheader("Parâmetros fixos")
    st.dataframe(PARAMS_FIXOS, width='stretch', hide_index=True)

    st.subheader("Tipo de ambiente por modelo")
    st.dataframe(AMBIENTES, width='stretch', hide_index=True)