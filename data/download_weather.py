#!/usr/bin/env python3
"""kr_wind_download.py

한국 관측소(ISO‑3166 = "KR")의 바람(wdir, wspd, wpgt) 자료를 Meteostat API에서
가져와 CSV 파일로 저장하고, 옵션에 따라 하나의 통합 CSV 로 병합한다.

Usage:
    python kr_wind_download.py --start 2024-01-01 --end 2024-12-31 \
        --interval hourly --out_dir data --merge

Arguments
---------
--start YYYY-MM-DD   : 시작 날짜 (UTC)
--end   YYYY-MM-DD   : 종료 날짜 (UTC)
--interval           : hourly | daily (기본 hourly)
--out_dir            : 개별·통합 CSV 저장 디렉터리 (기본 ./output)
--limit N            : 관측소 개수 제한 (기본 None → 전부)
--bbox               : 경계 박스 (min_lat max_lat min_lon max_lon) 4개 값 입력
--merge              : 플래그 지정 시 모든 관측소 데이터를 하나로 병합해 저장
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys

import pandas as pd
from meteostat import Stations, Hourly, Daily, Point

# Meteostat 컬럼(2025‑07 기준) 중 "바람" 관련만 추림
WIND_COLS: list[str] = ["wdir", "wspd", "wpgt"]  # wind direction (°), speed (m/s), gust (m/s)


def get_korean_stations(limit: int | None = None, bbox: tuple[float, float, float, float] | None = None) -> pd.DataFrame:
    """한국(KR) 관측소 메타데이터를 반환한다."""
    qs = Stations().region("KR")
    if bbox:
        qs = qs.bounds(*bbox)
    return qs.fetch(limit)


def fetch_wind_data(point: Point, start: datetime, end: datetime, *, interval: str = "hourly") -> pd.DataFrame:
    """지점(Point)의 바람 자료를 Hourly 또는 Daily 해상도로 수집한다."""
    fetcher = Hourly if interval == "hourly" else Daily
    df = fetcher(point, start, end).fetch()
    # WIND_COLS 중 존재하는 컬럼만 골라 공백(NaN) 행은 제거
    cols = [c for c in WIND_COLS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].dropna(how="all")


def save_station_csv(station_id: str, df: pd.DataFrame, out_dir: Path) -> Path:
    """관측소별 CSV 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{station_id}.csv"
    df.to_csv(fp, index=False)
    return fp


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download wind data from Meteostat (Korea)")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", choices=["hourly", "daily"], default="hourly")
    p.add_argument("--out_dir", default="output")
    p.add_argument("--limit", type=int, default=None, help="관측소 개수 제한 (None → 전부)")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"), help="경계 박스 (min_lat max_lat min_lon max_lon)")
    p.add_argument("--merge", action="store_true", help="모든 관측소를 하나의 CSV로 병합 저장")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        sys.exit(f"❌ 날짜 형식 오류: {e}")

    stations = get_korean_stations(limit=args.limit, bbox=tuple(args.bbox) if args.bbox else None)
    if stations.empty:
        sys.exit("❌ 조건에 맞는 관측소를 찾지 못했습니다.")

    out_dir = Path(args.out_dir)
    master_frames: list[pd.DataFrame] = []

    for _, row in stations.iterrows():
        station_id = row.name  # WMO or ICAO ID
        point = Point(row.latitude, row.longitude, row.elevation)

        df = fetch_wind_data(point, start, end, interval=args.interval)
        if df.empty:
            print(f"⚠️  {station_id}: {args.start}–{args.end} 기간에 자료 없음")
            continue

        # UTC datetime 인덱스를 열로 변환 후 station 컬럼 추가
        df = df.reset_index().rename(columns={"time": "datetime"})
        df.insert(0, "station", station_id)

        # 개별 CSV 저장
        path = save_station_csv(station_id, df, out_dir)
        print(f"✅ {station_id}: {len(df):,} rows → {path}")

        if args.merge:
            master_frames.append(df)

    # 통합 CSV 저장
    if args.merge and master_frames:
        merged = pd.concat(master_frames, ignore_index=True).sort_values(["datetime", "station"])
        out_fp = out_dir / "KR_wind_all_stations.csv"
        merged.to_csv(out_fp, index=False)
        print(f"📦 병합 데이터셋 → {out_fp}  (총 {len(merged):,} rows)")


if __name__ == "__main__":
    main()
