import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from commpy.modulation import QAMModem

# -------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Modelos de propagação",
    page_icon="📡",
    layout="centered"
)


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

d_tx_rx = st.sidebar.slider("Distância Tx-Rx (m)", 1, 10000, 1000)
f = st.sidebar.slider("Frequência (GHz)", 0.5, 30.0, 2.0)
f = f *1e9 
sigma = st.sidebar.slider("Desvio padrão do sombreamento σ (dB)", 0.0, 12.0, 4.0)

noise_figure = st.sidebar.slider("Figura de ruído (dB)", 0.0, 10.0, 5.0)

awgn_db = st.sidebar.slider(
    "AWGN (dB)",
    0.0, 20.0, 10.0
)


# -------------------------------
# FUNÇÕES
# -------------------------------

def fspl(d, f):
    # perda de percurso PL(d0) (slide 7)
    # Gt=Gr=1 (antena isotrópica) e L=1 (sem perdas no sistema)
    c = 3e8  # velocidade da luz
    Gt=Gr=1
    lam = c / f  # lambda = c/f

    term = (Gt * Gr * lam**2) / ((4 * np.pi)**2 * d**2)

    PL = -10 * np.log10(term)

    return PL


def log_distance(d, d0, PL0, n): # PL 
    return PL0 + 10*n*np.log10(d/d0)

def indoor_simple(d, f, floors=1):
    # Modelo de Seidel
    # referência em d0 = 1m
    PL0 = fspl(1, f)

    d0 = 1

    # expoente indoor (valores típicos do slide)
    # 2 a 3.5 dependendo do ambiente
    nsf = np.where(d > d0, 2.0, 3.5)

    # Floor Attenuation Factor (FAF) do slide (WiFi indoor)
    FAF_table = {
        1: 12.9,
        2: 18.7,
        3: 24.0,
        4: 27.0
    }

    FAF = FAF_table.get(floors, 0)

    return PL0 + 10*nsf*np.log10(d/d0) + FAF

# -------------------------------
# Walfisch-Ikegami NLOS (SLIDE)
# -------------------------------

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

    # -----------------------------
    # L0
    # -----------------------------
    L0 = 32.4 + 20*log_d + 20*log_f

    # -----------------------------
    # Lrts (vetorizado sem IF)
    # -----------------------------
    Lori = np.where(phi <= 35,
                    -10 + 0.354*phi,
            np.where(phi <= 55,
                    2.5 + 0.075*(phi - 35),
                    4 - 0.114*(phi - 55)))

    delta_h = h_tx - h_rx

    Lrts = -8.2 - 10*np.log10(w) + 20*np.log10(delta_h) + Lori

    # -----------------------------
    # Lbsh
    # -----------------------------
    Lbsh = np.where(h_b > h_tx,
                    -18*np.log10(1 + delta_h),
                    0)

    # -----------------------------
    # KA (rápido agora)
    # -----------------------------
    Ka_val = Ka(f_mhz, d_km, h_b, h_tx)

    # -----------------------------
    # MSD (otimizado)
    # -----------------------------
    Kd = 18
    Kf = -4 + 0.7*(f_mhz/1000)

    Lmsd = (
        Lbsh
        + Ka_val
        + Kd*log_d
        + Kf*log_f
        - 9*np.log10(b)
    )

    # -----------------------------
    # decisão NLOS
    # -----------------------------
    return np.where((Lrts + Lmsd) > 0,
                    L0 + Lrts + Lmsd,
                    L0)



@st.cache_data
def compute_PL(model, d_tx_rx, f, sigma):
    # -------------------------------
# DISTÂNCIA (MANTIDO)
# -------------------------------
    d = np.linspace(1, d_tx_rx, 150)

    # -------------------------------
    # MODELO
    # -------------------------------
    if model == "Log-distância":
        # n = 3 -> urbano com sombreamento
        # d0 = 1m -> distância de referência
        PL0 = fspl(1, f) # potência recebida no ponto d0 = 1m
        PL = log_distance(d, 1, PL0, n=3)

    # outdoor
    # elif model == "Hata":
        # # áreas urbanas
        # # altura das antenas em metros
        # h_tx = 30
        # h_rx = 1.5
        
        # # conversões IMPORTANTES
        # f_MHz = f * 1000   # GHz -> MHz
        # d_km = d / 1000    # metros -> km

        # # função de correção a(h_rx)
        # a_hrx = (1.1*np.log10(f_MHz) - 0.7)*h_rx - (1.56*np.log10(f_MHz) - 0.8)

        # PL = (
        #     69.55
        #     + 26.16*np.log10(f_MHz)
        #     - 13.82*np.log10(h_tx)
        #     - a_hrx
        #     + (44.9 - 6.55*np.log10(h_tx)) * np.log10(d_km)
        # )
        
    elif model == "Indoor":
        PL = indoor_simple(d, f, floors=1)

    # outdoor
    elif model == "Walfisch-Bertoni":
        PL = walfisch_ikegami(d,f,h_tx=30,h_rx=1.5,h_b=25,w=20,b=20,phi=30)

    # -------------------------------
    # SOMBREAMENTO
    # -------------------------------
    # sombreamento log-normal (árvores, prédios, morros, etc.)
    shadow = np.random.normal(0, sigma, len(d)) # variável aleatória com média 0 e desvio padrão sigma -> distribuição gaussiana
    PL_shadow = PL + shadow

    return d,PL_shadow

# -------------------------------
# CACHE LOGIC
# -------------------------------
d,PL = compute_PL(model, d_tx_rx, f, sigma)



@st.cache_data
def compute_constellation(PL_last, noise_figure, M, awgn_db, n_symbols=4000):

    modem = QAMModem(M)

    bits = np.random.randint(0, 2, int(np.log2(M) * n_symbols))
    symbols_tx = modem.modulate(bits)

    I_tx = symbols_tx.real
    Q_tx = symbols_tx.imag
   
    # -------------------------------
    # LINK BUDGET (dB)
    # -------------------------------
    Pt_dBm = 43 # macro BS
    Pr_dBm = Pt_dBm - PL_last

    # piso de ruído do sistema + AWGN slider + NF
    noise_dBm = -174 + 10*np.log10(1e6) + noise_figure + awgn_db

    SNR_dB = Pr_dBm - noise_dBm

    SNR_linear = 10 ** (SNR_dB / 10)

    # por símbolo
    Es = np.mean(np.abs(symbols_tx)**2)
    noise_std = np.sqrt(Es / (2 * SNR_linear)) 
    # SNR = Es/N0
    #sigma^2 = N0/2
    # sigma^2 = Es/2*SINR


    # -------------------------------
    # canal AWGN
    # -------------------------------
    I_rx = I_tx + np.random.normal(0, noise_std, len(I_tx))
    Q_rx = Q_tx + np.random.normal(0, noise_std, len(Q_tx))

    return modem.constellation.real, modem.constellation.imag, I_rx, Q_rx

def qam_limits(M):
    m = int(np.sqrt(M))
    lim = m - 1
    return -(lim + 1), (lim + 1)

# -------------------------------
# TABS 
# -------------------------------
tab = st.radio(
    "Selecione a visualização",
    ["Curva de atenuação", "Constelação"],
    horizontal=True
)

# -------------------------------
# TAB 1
# -------------------------------
if tab == "Curva de atenuação":
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
else:
    I_ideal, Q_ideal, I_rx, Q_rx = compute_constellation(PL[-1], noise_figure,M, awgn_db)

    # -------------------------------
    # PLOT
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    # pontos recebidos = nuvem
    ax2.scatter(I_rx, Q_rx, s=4, alpha=0.6, label="Rx")

    # pontos ideais transmitidos
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
