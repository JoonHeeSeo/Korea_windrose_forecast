#!/usr/bin/env python3
"""build_wind_atlas.py
──────────────────────
주어진 통합 CSV(`KR_wind_all_stations.csv`)로부터 관측소별 풍황(風況)
통계치와 풍향 장미(rose) 그래프, Weibull 분포 파라미터 등을 계산해
간이 Wind Atlas를 만들어 저장합니다.

Usage
-----
1) 기본 실행 예
   python build_wind_atlas.py \
       --input data/KR_wind_all_stations.csv \
       --meta  data/KR_wind_metadata.csv \
       --out   atlas

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


def mean_power_density(speeds: np.ndarray, rho=1.225):
    """0.5 * rho * v^3 의 평균 (W/m²)"""
    speeds = speeds[~np.isnan(speeds)]
    return 0.5 * rho * np.mean(speeds ** 3)


def direction_bins(deg, bins=16):
    """풍향을 16방위 등으로 빈도 집계 (결과: dict[label] = frequency)"""
    labels = np.arange(0, 360, 360 / bins)  # 0,22.5,...
    idx = np.floor((deg % 360) / (360 / bins)).astype(int)
    counts = pd.Series(idx).value_counts(normalize=True, sort=False)
    return {int(labels[i]): counts.get(i, 0.0) for i in range(bins)}


# ──────────────────────────────────────────────────────────────
# 메인 프로시저
# ──────────────────────────────────────────────────────────────

def build_atlas(df, meta_df=None, freq="annual", plot_rose=False, out_dir="atlas", rho=1.225):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 집계 주기별 그룹 키 설정
    if freq == "monthly":
        df["period"] = df["datetime"].dt.to_period("M")
    else:
        df["period"] = df["datetime"].dt.to_period("Y")

    summaries = []

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
            **{f"dir_{k}": v for k, v in rose.items()}
        }
        summaries.append(rec)

        # 풍향장미 그래프 (옵션)
        if plot_rose:
            try:
                from windrose import WindroseAxes  # pip install windrose
                fig = plt.figure(figsize=(6,6))
                ax = WindroseAxes.from_ax(fig=fig)
                ax.bar(wdir, wspd, normed=True, opening=0.9, edgecolor='white')
                ax.set_legend()
                fig.suptitle(f"Station {station} {period}")
                fig.savefig(out_dir / f"rose_{station}_{period}.png", dpi=150)
                plt.close(fig)
            except ImportError:
                print("✖ windrose 패키지 없음: rose plot 스킵")

    atlas_df = pd.DataFrame(summaries)

    # 메타데이터 병합
    if meta_df is not None:
        atlas_df = atlas_df.merge(meta_df, on="station", how="left")

    atlas_path = out_dir / f"wind_atlas_{freq}.csv"
    atlas_df.to_csv(atlas_path, index=False)
    print(f"✅ Wind atlas saved → {atlas_path}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="통합 CSV 파일 경로")
    ap.add_argument("--meta",  help="관측소 메타데이터 CSV(선택)")
    ap.add_argument("--out", default="atlas", help="출력 폴더")
    ap.add_argument("--freq", choices=["monthly", "annual"], default="annual")
    ap.add_argument("--plot_rose", action="store_true", help="풍향장미 PNG 저장")
    ap.add_argument("--rho", type=float, default=1.225, help="공기밀도 kg/m³")
    args = ap.parse_args()

    # 데이터 로드
    df = pd.read_csv(args.input, parse_dates=["datetime"])
    meta_df = pd.read_csv(args.meta) if args.meta else None

    build_atlas(df, meta_df, freq=args.freq, plot_rose=args.plot_rose, out_dir=args.out, rho=args.rho)
