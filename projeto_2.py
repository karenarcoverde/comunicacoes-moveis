import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
    ["Log-distância", "Hata", "Walfisch-Bertoni", "Indoor"]
)

d_tx_rx = st.sidebar.slider("Distância Tx-Rx (m)", 1, 10000, 1000)
f = st.sidebar.slider("Frequência (GHz)", 0.5, 30.0, 2.0)
f = f *1e9 
sigma = st.sidebar.slider("Desvio padrão do sombreamento σ (dB)", 0.0, 12.0, 4.0)

noise_figure = st.sidebar.slider("Figura de ruído (dB)", 0.0, 10.0, 5.0)

# -------------------------------
# FUNÇÕES
# -------------------------------

def fspl(d, f):
    c = 3e8  # velocidade da luz
    # FSPL perda de percurso na situação de LOS
    # Gt=Gr=1 (antena isotrópica) e L=1 (sem perdas no sistema)
    return 20*np.log10(4*np.pi*d*f/c)


def log_distance(d, d0, PL0, n):
    return PL0 + 10*n*np.log10(d/d0)

def indoor_simple(d, f):
    # potência recebida no ponto d0 = 1m
    PL0 = fspl(1, f)
    return PL0 + 30*np.log10(d/1)

def walfisch_bertoni(d, f):
    return fspl(d, f) + 0.3 * 10*np.log10(d)

# -------------------------------
# DISTÂNCIA (MANTIDO)
# -------------------------------
d = np.linspace(1, d_tx_rx, 300)

# -------------------------------
# MODELO
# -------------------------------
if model == "Log-distância":
    # n = 3 -> urbano com sombreamento
    # d0 = 1m -> distância de referência
    PL0 = fspl(1, f) # potência recebida no ponto d0 = 1m
    PL = log_distance(d, 1, PL0, n=3)

# outdoor
elif model == "Hata":
    # áreas urbanas
    # altura das antenas em metros
    h_tx = 30
    h_rx = 1.5

    # função de correção a(h_rx)
    a_hrx = (1.1*np.log10(f) - 0.7)*h_rx - (1.56*np.log10(f) - 0.8)

    PL = (
        69.55
        + 26.16*np.log10(f)
        - 13.82*np.log10(h_tx)
        - a_hrx
        + (44.9 - 6.55*np.log10(h_tx)) * np.log10(d)
    )

elif model == "Indoor":
    PL = indoor_simple(d, f)

# outdoor
elif model == "Walfisch-Bertoni":
    PL = walfisch_bertoni(d, f)

# -------------------------------
# SOMBREAMENTO
# -------------------------------
# sombreamento log-normal (árvores, prédios, morros, etc.)
shadow = np.random.normal(0, sigma, len(d)) # variável aleatória com média 0 e desvio padrão sigma -> distribuição gaussiana
PL_shadow = PL + shadow


# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Curva de atenuação", "Constelação"])

# -------------------------------
# TAB 1 - GRÁFICO
# -------------------------------
with tab1:
    fig, ax = plt.subplots()

    ax.plot(d, PL_shadow, label=model)

    ax.set_xlabel("Distância (m)")
    ax.set_ylabel("PL (dB)")
    ax.grid()
    ax.legend()

    st.pyplot(fig)