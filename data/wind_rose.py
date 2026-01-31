#!/usr/bin/env python3
"""build_wind_rose.py
──────────────────────
주어진 통합 CSV(`KR_wind_all_stations.csv`)로부터 관측소별 풍황(風況)
통계치와 풍향 장미(rose) 그래프, Weibull 분포 파라미터 등을 계산해
간이 **Wind Rose** 패키지를 만들어 저장합니다.

Usage
-----
1) 기본 실행 예
   python build_wind_rose.py \
       --input data/KR_wind_all_stations.csv \
       --meta  data/KR_wind_metadata.csv \
       --out   rose

2) 옵션
   --freq monthly|annual : 집계 주기 (기본 annual)
   --plot_rose          : 풍향장미 PNG 저장(각 관측소)
   --rho 1.225          : 공기밀도(kg/m³) (기본 해수면 표준)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

from crosswind import calc_crosswind, calc_headwind, load_airports

# ──────────────────────────────────────────────────────────────
# 통계 함수
# ──────────────────────────────────────────────────────────────

def fit_weibull(speeds: np.ndarray):
    """최대우도법으로 Weibull k(형상), c(Scale) 추정"""
    speeds = speeds[~np.isnan(speeds) & (speeds > 0)]
    if len(speeds) < 20:
        return np.nan, np.nan
    c, loc, k = weibull_min.fit(speeds, floc=0)  # scipy는 (c=shape, scale=k)
    return c, k


def mean_power_density(speeds: np.ndarray, rho: float = 1.225):
    """0.5 * rho * v^3 의 평균 (W/m²)"""
    speeds = speeds[~np.isnan(speeds)]
    return 0.5 * rho * np.mean(speeds ** 3)


def direction_bins(deg, bins: int = 16):
    """풍향을 16방위 등으로 빈도 집계 (결과: dict[label] = frequency)"""
    labels = np.arange(0, 360, 360 / bins)  # 0,22.5,...
    idx = np.floor((deg % 360) / (360 / bins)).astype(int)
    counts = pd.Series(idx).value_counts(normalize=True, sort=False)
    return {int(labels[i]): counts.get(i, 0.0) for i in range(bins)}


# ──────────────────────────────────────────────────────────────
# 메인 프로시저
# ──────────────────────────────────────────────────────────────

def build_rose(
    df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    *,
    freq: str = "annual",
    plot_rose: bool = False,
    out_dir: str | Path = "rose",
    rho: float = 1.225,
    airports_df: pd.DataFrame | None = None,
    station_icao_map: dict | None = None,
):
    """관측소별 풍향장미 및 풍황 통계 테이블 생성

    Parameters
    ----------
    df : pd.DataFrame
        통합 바람 데이터 (datetime, station, wspd, wdir 컬럼 필수)
    meta_df : pd.DataFrame, optional
        관측소 메타데이터
    freq : str
        집계 주기 ('monthly' 또는 'annual')
    plot_rose : bool
        풍향장미 PNG 저장 여부
    out_dir : str | Path
        출력 폴더
    rho : float
        공기밀도 (kg/m³)
    airports_df : pd.DataFrame, optional
        공항/활주로 정보. 제공 시 교차풍 통계 계산
    station_icao_map : dict, optional
        관측소 → ICAO 코드 매핑
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 공항별 활주로 heading 매핑 구축 (교차풍 계산용)
    airport_headings = {}
    if airports_df is not None:
        for icao in airports_df["icao"].unique():
            # 각 공항의 주 활주로 heading (첫 번째 활주로 사용)
            airport_headings[icao] = airports_df[airports_df["icao"] == icao]["heading"].iloc[0]

    # 집계 주기별 그룹 키 설정
    if freq == "monthly":
        df["period"] = df["datetime"].dt.to_period("M")
    else:
        df["period"] = df["datetime"].dt.to_period("Y")

    summaries: list[dict] = []

    for (station, period), g in df.groupby(["station", "period"]):
        wspd = g["wspd"].values.astype(float)
        wdir = g["wdir"].values.astype(float)

        mean = np.nanmean(wspd)
        p50 = np.nanpercentile(wspd, 50)
        p90 = np.nanpercentile(wspd, 90)
        shape, scale = fit_weibull(wspd)
        mpd = mean_power_density(wspd, rho)
        rose = direction_bins(wdir)

        rec = {
            "station": station,
            "period": str(period),
            "n": len(g),
            "mean": mean,
            "p50": p50,
            "p90": p90,
            "weibull_k": shape,
            "weibull_c": scale,
            "power_density": mpd,
            **{f"dir_{k}": v for k, v in rose.items()},
        }

        # 교차풍 통계 추가 (공항 데이터가 있는 경우)
        if airports_df is not None:
            # station → ICAO 매핑
            if station_icao_map:
                icao = station_icao_map.get(station)
            else:
                icao = station  # station 값이 ICAO 코드라고 가정

            if icao and icao in airport_headings:
                heading = airport_headings[icao]
                valid = ~np.isnan(wspd) & ~np.isnan(wdir)
                wspd_valid = wspd[valid]
                wdir_valid = wdir[valid]

                if len(wspd_valid) > 0:
                    crosswind = calc_crosswind(wspd_valid, wdir_valid, heading)
                    headwind = calc_headwind(wspd_valid, wdir_valid, heading)

                    rec["runway_heading"] = heading
                    rec["crosswind_mean"] = np.mean(crosswind)
                    rec["crosswind_abs_mean"] = np.mean(np.abs(crosswind))
                    rec["crosswind_abs_max"] = np.max(np.abs(crosswind))
                    rec["crosswind_p90"] = np.percentile(np.abs(crosswind), 90)
                    rec["headwind_mean"] = np.mean(headwind)
                    rec["tailwind_max"] = -np.min(headwind)
                    rec["exceed_15kt_pct"] = 100 * np.mean(np.abs(crosswind) > 15)

        summaries.append(rec)

        # 풍향장미 그래프 (옵션)
        if plot_rose:
            try:
                from windrose import WindroseAxes  # pip install windrose

                fig = plt.figure(figsize=(6, 6))
                ax = WindroseAxes.from_ax(fig=fig)
                ax.bar(wdir, wspd, normed=True, opening=0.9, edgecolor="white")
                ax.set_legend()
                fig.suptitle(f"Station {station} {period}")
                fig.savefig(out_dir / f"rose_{station}_{period}.png", dpi=150)
                plt.close(fig)
            except ImportError:
                print("✖ windrose 패키지 없음: rose plot 스킵")

    rose_df = pd.DataFrame(summaries)

    # 메타데이터 병합
    if meta_df is not None:
        rose_df = rose_df.merge(meta_df, on="station", how="left")

    rose_path = out_dir / f"wind_rose_{freq}.csv"
    rose_df.to_csv(rose_path, index=False)
    print(f"✅ Wind rose saved → {rose_path}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="통합 CSV 파일 경로")
    ap.add_argument("--meta", help="관측소 메타데이터 CSV(선택)")
    ap.add_argument("--out", default="rose", help="출력 폴더")
    ap.add_argument("--freq", choices=["monthly", "annual"], default="annual")
    ap.add_argument("--plot_rose", action="store_true", help="풍향장미 PNG 저장")
    ap.add_argument("--rho", type=float, default=1.225, help="공기밀도 kg/m³")
    ap.add_argument("--airports", help="공항/활주로 CSV (교차풍 통계 계산용)")
    ap.add_argument("--station_icao", help="관측소→ICAO 매핑 JSON 파일")
    args = ap.parse_args()

    # 데이터 로드
    df = pd.read_csv(args.input, parse_dates=["datetime"])
    meta_df = pd.read_csv(args.meta) if args.meta else None

    # 공항 데이터 로드
    airports_df = None
    if args.airports:
        airports_df = pd.read_csv(args.airports)
    else:
        # 기본 경로에서 시도
        default_airports = Path(__file__).parent / "airports.csv"
        if default_airports.exists():
            airports_df = load_airports(default_airports)

    # station → ICAO 매핑 로드
    station_icao_map = None
    if args.station_icao:
        import json
        with open(args.station_icao) as f:
            station_icao_map = json.load(f)

    build_rose(
        df,
        meta_df,
        freq=args.freq,
        plot_rose=args.plot_rose,
        out_dir=args.out,
        rho=args.rho,
        airports_df=airports_df,
        station_icao_map=station_icao_map,
    )
