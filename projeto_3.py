import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Desvanecimento 3GPP 5G",
    page_icon="📡",
    layout="centered"
)


# -------------------------------
# ENTRADAS
# -------------------------------
v_kmh = st.sidebar.slider("Velocidade do terminal (km/h)", 0, 300, 60)

freq_band = st.sidebar.selectbox(
    "Faixa de frequência",
    ["Micro-ondas (3–30 GHz)", "Milimétricas (30–300 GHz)"]
)

channel_ui = st.sidebar.selectbox(
    "Perfil de canal 3GPP",
    ["TDL-A", "TDL-B", "TDL-C"]
)

rms_delay = st.sidebar.slider("RMS Delay Spread (ns)", 10, 1000, 150)

noise_floor = st.sidebar.slider("Limiar de ruído (dB)", -120, -5, -90)

# -------------------------------
# CONVERSÕES FÍSICAS
# -------------------------------
v = v_kmh / 3.6
c = 3e8

if "Micro" in freq_band:
    fc = 4e9
else:
    fc = 40e9

f_doppler = v * fc / c

# -------------------------------
# TDL REAL (3GPP TR 38.901)
# -------------------------------
tdl_profiles = {
    "TDL-A": {
        "delays": np.array([
            0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
            0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942,
            2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066,
            5.3043, 9.6586
        ]),
        "powers": np.array([
            -13.4, 0, -2.2, -4, -6, -8.2, -9.9,
            -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2,
            -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6,
            -19.9, -29.7
        ])
    },

    "TDL-B": {
        "delays": np.array([
            0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752,
            0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756,
            1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067,
            4.2790, 4.7834
        ]),
        "powers": np.array([
            0, -2.2, -4, -3.2, -9.8, -1.2, -3.4,
            -5.2, -7.6, -3, -8.9, -9, -4.8, -5.7,
            -7.5, -1.9, -7.6, -12.2, -9.8, -11.4,
            -14.9, -9.2, -11.3
        ])
    },

    "TDL-C": {
        "delays": np.array([
            0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448,
            0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
            2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065,
            6.6374, 7.0427, 8.6523
        ]),
        "powers": np.array([
            -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2,
            -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8,
            -8.7, -13.2, -13.9, -13.9, -15.8, -17.1,
            -16, -15.7, -21.6, -22.8
        ])
    }
}

profile = tdl_profiles[channel_ui]

delays = profile["delays"].astype(float)
powers_db = profile["powers"].astype(float)

# -------------------------------
# FUNÇÃO RMS REAL (3GPP)
# -------------------------------
def compute_rms_delay(delays, powers_db):
    power_linear = 10 ** (powers_db / 10)

    power_linear = power_linear / np.sum(power_linear)  # energia normalizada

    mean = np.sum(power_linear * delays)
    mean2 = np.sum(power_linear * delays**2)

    return np.sqrt(mean2 - mean**2)

# -------------------------------
# AJUSTE PARA RMS
# -------------------------------
current_rms = compute_rms_delay(delays, powers_db)

scale = rms_delay / (current_rms + 1e-12)
delays = delays * scale

# -------------------------------
# PDP (Power Delay Profile)
# -------------------------------
powers_linear = 10 ** (powers_db / 10)

pdp = powers_linear / np.max(powers_linear)

pdp_db = 10 * np.log10(pdp + 1e-12)


# -------------------------------
# PLOT
# -------------------------------
fig, ax = plt.subplots()

ax.stem(delays, pdp_db, label=channel_ui)
ax.axhline(noise_floor, color="red", linestyle="--", label="Noise Floor")

ax.set_title(f"PDP - {channel_ui}")
ax.set_xlabel("Delay (ns)")
ax.set_ylabel("Potência (dB)")
ax.legend()

st.pyplot(fig)
