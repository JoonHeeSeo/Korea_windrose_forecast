# Korea Wind Rose

[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-blue)](https://korea-windrose-forecast.streamlit.app/)

Wind rose visualization tool for Korean weather stations using historical data from Meteostat.

**Live demo:** https://korea-windrose-forecast.streamlit.app/

## Features

- Wind data from Korean weather stations via Meteostat API
- Hourly or daily resolution
- Annual or monthly aggregation
- Weibull distribution fitting and mean power density calculation
- 16-direction wind rose frequencies
- Year-to-year comparison at selected stations

## Installation

```bash
# Python 3.13+ recommended
uv sync
```

## Data Pipeline

Weather data is fetched from [Meteostat](https://dev.meteostat.net/) and cached locally.

```bash
# Download wind data
uv run python data/download_weather.py --start 2013-01-01 --end 2013-12-31 --interval hourly --limit 50 --out_dir korea_wind --merge

# Generate wind rose statistics
uv run python data/wind_rose.py --input korea_wind/KR_wind_all_stations.csv --out service --freq annual
```

## Running the App

```bash
uv run streamlit run service/app.py
```

The app allows you to:
- Select a station and view wind rose diagrams
- Compare wind patterns across different years
- Check wind resource statistics (Weibull parameters, power density)
- Download data as CSV

Works on Streamlit Community Cloud or any Docker host.
