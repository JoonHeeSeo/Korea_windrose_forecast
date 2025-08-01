import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Rose Explorer – Compare Years",
    page_icon="🎏",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(year: int) -> pd.DataFrame:
    """Load an annual wind‑rose CSV named `wind_rose_annual_{year}.csv`."""
    csv_path = Path(__file__).parent / f"wind_rose_annual_{year}.csv"
    if not csv_path.exists():
        st.error(f"CSV not found: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    df["year"] = year
    return df

# -----------------------------------------------------------------------------
# Configure available years (edit this list when new files arrive)
# -----------------------------------------------------------------------------
DEFAULT_YEARS = list(range(2013, 2025))
if not DEFAULT_YEARS:
    st.error("No CSV files specified in DEFAULT_YEARS list.")
    st.stop()

ALL_DATA = {yr: load_data(yr) for yr in DEFAULT_YEARS}

# -----------------------------------------------------------------------------
# Sidebar – user controls (two selectboxes but allow duplicates)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
col_y1, col_y2 = st.sidebar.columns(2)
year1 = col_y1.selectbox("Year A", options=DEFAULT_YEARS, index=0, key="year_a")
year2_default_idx = 1 if len(DEFAULT_YEARS) > 1 else 0
year2 = col_y2.selectbox("Year B", options=DEFAULT_YEARS, index=year2_default_idx, key="year_b")

# If duplicate, we treat as single‑year view
if year1 == year2:
    sel_years = [year1]
    mode_label = "Single year"
else:
    sel_years = [year1, year2]
    mode_label = "Comparison"

# Display mode selector only when comparing two different years
if len(sel_years) == 2:
    display_mode = st.sidebar.radio("Display mode", ["Overlay", "Side‑by‑side"], index=0)
else:
    display_mode = "Single"

# -----------------------------------------------------------------------------
# Determine common stations across the selected year(s)
# -----------------------------------------------------------------------------
station_sets = [set(ALL_DATA[yr]["station"].unique()) for yr in sel_years]
common_stations = sorted(set.intersection(*station_sets))

if not common_stations:
    st.error("No common stations found across the selected year(s).")
    st.stop()

sel_station = st.sidebar.selectbox("Station ID", common_stations, format_func=str)

# Retrieve rows for the selected station & years
rows = {yr: ALL_DATA[yr].loc[ALL_DATA[yr]["station"] == sel_station].iloc[0] for yr in sel_years}

# -----------------------------------------------------------------------------
# Title & headline metrics
# -----------------------------------------------------------------------------
if len(sel_years) == 1:
    yr = sel_years[0]
    row = rows[yr]
    st.title(f"Wind Rose · Station {sel_station} · {yr}")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Mean (m/s)", f"{row['mean']:.2f}")
    with m2:
        st.metric("p90 (m/s)", f"{row['p90']:.1f}")
    with m3:
        st.metric("Weibull k", f"{row['weibull_k']:.2f}")
    with m4:
        st.metric("Weibull c (m/s)", f"{row['weibull_c']:.2f}")
    with m5:
        st.metric("Power density (W/m²)", f"{row['power_density']:.0f}")
else:
    y1, y2 = sel_years
    row1, row2 = rows[y1], rows[y2]
    st.title(f"Wind Rose · Station {sel_station} · {y1} vs {y2}")

    col1, col2 = st.columns(2)
    for col, yr, rw in zip((col1, col2), (y1, y2), (row1, row2)):
        with col:
            st.subheader(str(yr))
            st.metric("Mean (m/s)", f"{rw['mean']:.2f}")
            st.metric("p90 (m/s)", f"{rw['p90']:.1f}")
            st.metric("Weibull k", f"{rw['weibull_k']:.2f}")
            st.metric("Weibull c (m/s)", f"{rw['weibull_c']:.2f}")
            st.metric("Power density (W/m²)", f"{rw['power_density']:.0f}")

# -----------------------------------------------------------------------------
# Constants for wind‑rose plot
# -----------------------------------------------------------------------------
DIRECTION_BINS = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
FREQ_COLS = [f"dir_{d}" for d in DIRECTION_BINS]


def build_polar_chart(freqs: np.ndarray, label: str | None = None):
    """Return a matplotlib Figure (polar) for a year's frequencies."""
    freqs = np.append(freqs, freqs[0])  # close circle
    angles = np.deg2rad(DIRECTION_BINS + [360])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, freqs, linewidth=1.5, label=label)
    ax.fill(angles, freqs, alpha=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(DIRECTION_BINS, [f"{d}°" for d in DIRECTION_BINS])
    if label:
        ax.legend(loc="upper right", bbox_to_anchor=(1.07, 1.10))
    return fig

# -----------------------------------------------------------------------------
# Chart rendering
# -----------------------------------------------------------------------------
if len(sel_years) == 1:
    yr = sel_years[0]
    row = rows[yr]
    fig = build_polar_chart(row[FREQ_COLS].values)
    st.pyplot(fig, use_container_width=True)
else:
    y1, y2 = sel_years
    row1, row2 = rows[y1], rows[y2]
    if display_mode == "Overlay":
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        for yr, rw in ((y1, row1), (y2, row2)):
            freqs = np.append(rw[FREQ_COLS].values, rw[FREQ_COLS].values[0])
            angles = np.deg2rad(DIRECTION_BINS + [360])
            ax.plot(angles, freqs, linewidth=1.5, label=str(yr))
            ax.fill(angles, freqs, alpha=0.25)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(DIRECTION_BINS, [f"{d}°" for d in DIRECTION_BINS])
        ax.set_title("Wind‑direction frequency distribution (overlay)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.07, 1.10))
        st.pyplot(fig, use_container_width=True)
    else:  # Side‑by‑side
        c1, c2 = st.columns(2)
        for col, yr, rw in zip((c1, c2), (y1, y2), (row1, row2)):
            fig = build_polar_chart(rw[FREQ_COLS].values, label=str(yr))
            col.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Raw data toggle
# -----------------------------------------------------------------------------
if st.checkbox("Show raw annual records"):
    raw_df = pd.concat([rows[yr].to_frame().T for yr in sel_years], ignore_index=True)
    st.dataframe(raw_df)
