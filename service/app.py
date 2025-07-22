import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Atlas Explorer",
    page_icon="🌬️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the annual wind‑atlas statistics shipped with the app.

    The CSV is expected to live **in the same folder** as this `app.py`, named
    `wind_atlas_annual.csv`.
    """
    csv_path = Path(__file__).parent / "wind_atlas_annual.csv"
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path)


df = load_data()

# -----------------------------------------------------------------------------
# Sidebar – station selector
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
stations = sorted(df["station"].unique())
sel_station = st.sidebar.selectbox("Station ID", stations, format_func=str)

row = df.loc[df["station"] == sel_station].iloc[0]

# -----------------------------------------------------------------------------
# Title & headline metrics
# -----------------------------------------------------------------------------
st.title(f"Wind Atlas · Station {sel_station} · {int(row['period'])}")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Mean wind speed (m/s)", f"{row['mean']:.2f}")
with m2:
    st.metric("p90 wind speed (m/s)", f"{row['p90']:.1f}")
with m3:
    st.metric("Weibull k", f"{row['weibull_k']:.2f}")
with m4:
    st.metric("Weibull c (m/s)", f"{row['weibull_c']:.2f}")
with m5:
    st.metric("Power density (W/m²)", f"{row['power_density']:.0f}")

# -----------------------------------------------------------------------------
# Wind‑rose chart
# -----------------------------------------------------------------------------

direction_bins = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
labels = [f"{d}°" for d in direction_bins]
freq_cols = [f"dir_{d}" for d in direction_bins]
freqs = row[freq_cols].values

# Ensure the polar chart closes the circle by appending the first element
freqs = np.append(freqs, freqs[0])
angles = np.deg2rad(direction_bins + [360])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, freqs, linewidth=1.5)
ax.fill(angles, freqs, alpha=0.3)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetagrids(direction_bins, labels)
ax.set_title("Wind‑direction frequency distribution", pad=20)

st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Raw data toggle
# -----------------------------------------------------------------------------
if st.checkbox("Show raw annual record"):
    st.dataframe(row.to_frame().T)
