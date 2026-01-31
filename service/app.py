import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add data directory to path for crosswind module
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from crosswind import (
    load_airports,
    calc_crosswind,
    calc_headwind,
    calc_risk_score,
    get_risk_statistics,
    CROSSWIND_LIMITS,
    TAILWIND_LIMITS,
)

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


@st.cache_data(show_spinner=False)
def load_airport_data() -> pd.DataFrame:
    """Load airport/runway data."""
    csv_path = Path(__file__).parent.parent / "data" / "airports.csv"
    if not csv_path.exists():
        return None
    return load_airports(csv_path)

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

# -----------------------------------------------------------------------------
# Crosswind Analysis Section
# -----------------------------------------------------------------------------
st.divider()
st.header("Crosswind Analysis")

airports_df = load_airport_data()

if airports_df is not None:
    # Airport selection
    airport_options = airports_df["icao"].unique().tolist()
    airport_names = {
        row["icao"]: f"{row['icao']} - {row['name']}"
        for _, row in airports_df.drop_duplicates("icao").iterrows()
    }

    sel_airport = st.selectbox(
        "Select Airport",
        options=airport_options,
        format_func=lambda x: airport_names.get(x, x),
    )

    # Get runways for selected airport
    airport_runways = airports_df[airports_df["icao"] == sel_airport]
    runway_options = airport_runways["runway"].tolist()
    sel_runway = st.selectbox("Select Runway", options=runway_options)

    # Get runway heading
    runway_heading = airport_runways[airport_runways["runway"] == sel_runway]["heading"].iloc[0]

    st.info(f"Runway heading: **{runway_heading}°**")

    # Crosswind calculation for selected station
    # Using mean wind speed and dominant direction from wind rose data
    yr = sel_years[0]
    row = rows[yr]

    # Find dominant wind direction (highest frequency)
    dir_cols = [c for c in row.index if c.startswith("dir_")]
    if dir_cols:
        dominant_dir_col = max(dir_cols, key=lambda c: row[c])
        dominant_dir = int(dominant_dir_col.replace("dir_", ""))
        mean_wspd = row["mean"]

        # Calculate crosswind components
        crosswind = calc_crosswind(
            np.array([mean_wspd]),
            np.array([dominant_dir]),
            runway_heading
        )[0]
        headwind = calc_headwind(
            np.array([mean_wspd]),
            np.array([dominant_dir]),
            runway_heading
        )[0]

        # Calculate risk score
        risk_score, risk_level = calc_risk_score(abs(crosswind), headwind)

        # Display crosswind metrics
        st.subheader("Crosswind Components (based on mean wind)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            xw_direction = "L→R" if crosswind > 0 else "R→L"
            st.metric(
                "Crosswind",
                f"{abs(crosswind):.1f} m/s",
                delta=xw_direction,
                delta_color="off"
            )

        with col2:
            hw_type = "Headwind" if headwind >= 0 else "Tailwind"
            st.metric(
                hw_type,
                f"{abs(headwind):.1f} m/s",
            )

        with col3:
            st.metric("Risk Score", f"{risk_score}/100")

        with col4:
            # Risk level with color
            level_colors = {
                "LOW": "🟢",
                "MODERATE": "🟡",
                "HIGH": "🟠",
                "CRITICAL": "🔴",
            }
            st.metric("Risk Level", f"{level_colors.get(risk_level, '')} {risk_level}")

        # Risk level explanation
        st.subheader("Risk Level Criteria")

        risk_table = pd.DataFrame({
            "Level": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
            "Crosswind (kt)": ["< 10", "10 - 15", "15 - 20", "> 20"],
            "Tailwind (kt)": ["< 5", "5 - 10", "10 - 15", "> 15"],
            "Score Range": ["0 - 25", "26 - 50", "51 - 75", "76 - 100"],
        })
        st.table(risk_table)

        # Crosswind visualization
        st.subheader("Wind Component Diagram")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw runway
        runway_length = 1.0
        runway_x = [0, runway_length * np.sin(np.radians(runway_heading))]
        runway_y = [0, runway_length * np.cos(np.radians(runway_heading))]
        ax.plot(runway_x, runway_y, 'k-', linewidth=8, label=f'Runway {sel_runway}')

        # Draw wind vector
        wind_scale = 0.05  # scale factor for visualization
        wind_x = mean_wspd * wind_scale * np.sin(np.radians(dominant_dir))
        wind_y = mean_wspd * wind_scale * np.cos(np.radians(dominant_dir))
        ax.annotate('', xy=(wind_x, wind_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(wind_x * 1.1, wind_y * 1.1, f'Wind\n{mean_wspd:.1f}m/s\n@{dominant_dir}°',
                ha='center', fontsize=9, color='blue')

        # Draw crosswind component
        xw_x = abs(crosswind) * wind_scale * np.sin(np.radians(runway_heading + 90))
        xw_y = abs(crosswind) * wind_scale * np.cos(np.radians(runway_heading + 90))
        if crosswind < 0:
            xw_x, xw_y = -xw_x, -xw_y
        ax.annotate('', xy=(xw_x, xw_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(xw_x * 1.3, xw_y * 1.3, f'XW\n{abs(crosswind):.1f}',
                ha='center', fontsize=9, color='red')

        # Draw headwind component
        hw_x = headwind * wind_scale * np.sin(np.radians(runway_heading))
        hw_y = headwind * wind_scale * np.cos(np.radians(runway_heading))
        ax.annotate('', xy=(hw_x, hw_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        hw_label = 'HW' if headwind >= 0 else 'TW'
        ax.text(hw_x * 1.3, hw_y * 1.3, f'{hw_label}\n{abs(headwind):.1f}',
                ha='center', fontsize=9, color='green')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('East-West')
        ax.set_ylabel('North-South')
        ax.set_title(f'Wind Components for {sel_airport} Runway {sel_runway}')

        # Add legend
        from matplotlib.patches import FancyArrowPatch
        ax.plot([], [], 'b-', linewidth=2, label='Wind Vector')
        ax.plot([], [], 'r-', linewidth=2, label='Crosswind (XW)')
        ax.plot([], [], 'g-', linewidth=2, label='Head/Tailwind (HW/TW)')
        ax.legend(loc='upper right')

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

else:
    st.warning("Airport data not found. Please ensure `data/airports.csv` exists.")
