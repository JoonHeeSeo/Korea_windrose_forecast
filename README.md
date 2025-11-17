# Korea_windrose_forecast

[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-blue)](https://korea-windrose-forecast.streamlit.app/)

https://korea-windrose-forecast.streamlit.app/

## 📖 Overview

Interactive wind rose explorer for Korean stations, built from historical wind data.  
You can compare **annual wind roses** and basic **wind resource statistics** (Weibull, power density, etc.) between years.

### Key Features

- Meteostat-based wind data for **Korean stations (`KR`)**
- Hourly or daily time resolution
- Annual or monthly aggregation
- Weibull fitting & mean power density
- 16-direction wind rose frequencies
- Streamlit UI for **year-to-year comparison** at a selected station

## 🌐 Live Demo

```text
https://korea-windrose-forecast.streamlit.app/
```

> or run locally with `streamlit run service/app.py` (see below).

---

## 🛠️ Installation

```bash
# Python 3.13+ is recommended
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r service/requirements.txt
```

## ⬇️ Data

Weather observations are pulled on‑demand from **[Meteostat](https://dev.meteostat.net/)** and cached locally.

```bash
python data/download_weather.py --start 2013-01-01 --end 2013-12-31 --interval hourly --limit 50 --out_dir korea_wind --merge
python data/wind_rose.py --input korea_wind/KR_wind_all_stations.csv --out service --freq annual
```

## 🚀 Training

## 🖥️ Streamlit App

The dashboard lets you

- Pick a date range and compare model forecasts with actual observations
- Visualise uncertainty bands & residuals
- Download predictions as CSV

Run locally:

```bash
streamlit run service/app.py
```

Deploy effortlessly to **Streamlit Community Cloud** (or any Docker‑ready host).
