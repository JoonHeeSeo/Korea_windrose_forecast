import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from pathlib import Path

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATH = Path("atlas_results/wind_atlas_annual_geo.csv")  # 좌표 포함 CSV
MAP_INITIAL = {"lat": 36.0, "lon": 127.5, "zoom": 6}

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="Korea Wind Atlas", layout="wide")
st.title("🇰🇷 Korea Wind Atlas – Streamlit Viewer")

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # 좌표 없는 행 제거
    return df.dropna(subset=["latitude", "longitude"])

df = load_data(DATA_PATH)

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR CONTROLS
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.header("🔧 Filters")
    metric = st.selectbox("Metric to color-map", ["mean", "p90", "power_density"], index=0)
    min_cov = st.slider("Minimum coverage (0~1)", 0.0, 1.0, 0.8, 0.05)
    stations = st.multiselect("Select stations (blank = all)", df["station"].sort_values().unique())
    show_rose = st.checkbox("Show wind rose for selected station", value=True)

# 필터 적용
mask = df["n"] / (24*365) >= min_cov  # assuming hourly data
if stations:
    mask &= df["station"].isin(stations)
filtered = df[mask].copy()

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAP – pydeck ScatterplotLayer
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position="[longitude, latitude]",
    get_radius=7000,
    get_fill_color="[scale, 140, 200-scale]",
    pickable=True,
)
# 색상 스케일 계산
min_v, max_v = filtered[metric].min(), filtered[metric].max()
scale = ((filtered[metric] - min_v)/(max_v - min_v + 1e-6)*255).astype(int)
layer.data = layer.data.assign(scale=scale)

view_state = pdk.ViewState(latitude=MAP_INITIAL["lat"], longitude=MAP_INITIAL["lon"], zoom=MAP_INITIAL["zoom"])

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{station}\n"+metric+": {"+metric+"}"}))

st.caption(f"Colored by `{metric}`; {len(filtered)} stations shown.")

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA TABLE
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("📊 Data table"):
    st.dataframe(filtered.round(2), use_container_width=True)

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WIND ROSE PLOT (single station)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if show_rose and len(stations)==1:
    st.header(f"🌸 Wind Rose – Station {stations[0]}")
    # 원본 시계열 CSV 경로 추정: atlas_results/rose_<station>_2023.png 이미 만들었다면 불러오기
    rose_png = Path("atlas_results")/f"rose_{stations[0]}_2023.png"
    if rose_png.exists():
        st.image(str(rose_png))
    else:
        st.info("No pre-generated rose plot found. Generating on the fly…")
        try:
            from windrose import WindroseAxes
            csv_master = Path("data/KR_wind_all_stations.csv")
            if not csv_master.exists():
                st.error("Master CSV not found. Provide KR_wind_all_stations.csv in data/ directory.")
            else:
                raw = pd.read_csv(csv_master, parse_dates=["datetime"])
                g = raw[raw.station==int(stations[0])]
                if g.empty:
                    st.warning("No data for this station in master CSV.")
                else:
                    fig = plt.figure(figsize=(6,6))
                    ax = WindroseAxes.from_ax(fig=fig)
                    ax.bar(g.wdir, g.wspd, normed=True, opening=0.9, edgecolor='white')
                    ax.set_legend()
                    st.pyplot(fig)
        except ImportError:
            st.error("windrose package not installed. Run `pip install windrose`.")
