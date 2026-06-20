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

sigma = st.sidebar.slider("Desvio padrão do sombreamento σ (dB)", 0.0, 12.0, 4.0)

noise_figure = st.sidebar.slider("Figura de ruído (dB)", 0.0, 10.0, 5.0)

# -------------------------------
# FUNÇÕES
# -------------------------------
def fspl(d, f):
    return 32.45 + 20*np.log10(f) + 20*np.log10(d)

def log_distance(d, d0, PL0, n):
    return PL0 + 10*n*np.log10(d/d0)

def indoor_simple(d, f):
    PL0 = fspl(0.01, f)
    return PL0 + 30*np.log10(d/0.01)

# -------------------------------
# DISTÂNCIA
# -------------------------------
d = np.linspace(1, d_tx_rx, 300) / 1000

# -------------------------------
# MODELO
# -------------------------------
if model == "Log-distância":
    PL0 = fspl(0.01, f)
    PL = log_distance(d, 0.01, PL0, n=3)

elif model == "Hata":
    f_mhz = f * 1000
    PL = (
        69.55
        + 26.16*np.log10(f_mhz)
        - 13.82*np.log10(30)
        + (44.9 - 6.55*np.log10(30))*np.log10(d)
    )

elif model == "Indoor":
    PL = indoor_simple(d, f)

elif model == "Walfisch-Bertoni":
    PL = fspl(d, f) + 10*np.log10(d*1000)*0.3

# -------------------------------
# SOMBREAMENTO
# -------------------------------
shadow = np.random.normal(0, sigma, len(d))
PL_shadow = PL + shadow

Pr = 0 - PL_shadow

# -------------------------------
# TABS (ABA ESQUERDA NO TOPO)
# -------------------------------
tab1,tab2 = st.tabs(["Curva de atenuação","Constelação"])

# -------------------------------
# TAB 1 - GRÁFICO
# -------------------------------
with tab1:
    fig, ax = plt.subplots()
    ax.plot(d*1000, Pr, label="Potência recebida (dBm)")
    ax.plot(d*1000, -PL_shadow, label="Path Loss (dB)", alpha=0.6)

    ax.set_xlabel("Distância (m)")
    ax.set_ylabel("dB / dBm")
    ax.grid()
    ax.legend()

    st.pyplot(fig)
