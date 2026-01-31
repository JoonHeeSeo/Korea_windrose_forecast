#!/usr/bin/env python3
"""crosswind.py
───────────────
활주로 기준 교차풍(crosswind) 및 정풍/배풍(headwind) 성분 계산 모듈

교차풍 계산 공식:
    crosswind = 풍속 × sin(바람방향 - 활주로방향)
    headwind  = 풍속 × cos(바람방향 - 활주로방향)

부호 규칙:
    - 교차풍 양수: 좌측에서 우측으로 부는 바람
    - 교차풍 음수: 우측에서 좌측으로 부는 바람
    - 정풍 양수: 활주로 정면에서 부는 바람 (headwind)
    - 정풍 음수: 활주로 뒤에서 부는 바람 (tailwind)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ──────────────────────────────────────────────────────────────
# 리스크 기준 상수 (단위: kt)
# ──────────────────────────────────────────────────────────────
CROSSWIND_LIMITS = {
    "LOW": 10,        # 10kt 미만: 안전
    "MODERATE": 15,   # 10-15kt: 주의
    "HIGH": 20,       # 15-20kt: 경고
    "CRITICAL": 20,   # 20kt 이상: 위험
}

TAILWIND_LIMITS = {
    "LOW": 5,         # 5kt 미만: 안전
    "MODERATE": 10,   # 5-10kt: 주의
    "HIGH": 15,       # 10-15kt: 경고
    "CRITICAL": 15,   # 15kt 이상: 위험
}


def calc_crosswind(wspd: np.ndarray, wdir: np.ndarray, runway_heading: float) -> np.ndarray:
    """교차풍 성분 계산

    Parameters
    ----------
    wspd : np.ndarray
        풍속 (m/s 또는 kt)
    wdir : np.ndarray
        바람이 불어오는 방향 (도, 0-360)
    runway_heading : float
        활주로 방향 (도, 0-360)

    Returns
    -------
    np.ndarray
        교차풍 성분 (양수: 좌→우, 음수: 우→좌)
    """
    angle_diff = np.radians(wdir - runway_heading)
    return wspd * np.sin(angle_diff)


def calc_headwind(wspd: np.ndarray, wdir: np.ndarray, runway_heading: float) -> np.ndarray:
    """정풍/배풍 성분 계산

    Parameters
    ----------
    wspd : np.ndarray
        풍속 (m/s 또는 kt)
    wdir : np.ndarray
        바람이 불어오는 방향 (도, 0-360)
    runway_heading : float
        활주로 방향 (도, 0-360)

    Returns
    -------
    np.ndarray
        정풍/배풍 성분 (양수: 정풍, 음수: 배풍)
    """
    angle_diff = np.radians(wdir - runway_heading)
    return wspd * np.cos(angle_diff)


def get_crosswind_risk_level(crosswind_kt: float) -> str:
    """교차풍 리스크 레벨 분류

    Parameters
    ----------
    crosswind_kt : float
        교차풍 절대값 (kt)

    Returns
    -------
    str
        리스크 레벨 ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
    """
    xw = abs(crosswind_kt)
    if xw < CROSSWIND_LIMITS["LOW"]:
        return "LOW"
    elif xw < CROSSWIND_LIMITS["MODERATE"]:
        return "MODERATE"
    elif xw < CROSSWIND_LIMITS["HIGH"]:
        return "HIGH"
    else:
        return "CRITICAL"


def get_tailwind_risk_level(headwind_kt: float) -> str:
    """배풍 리스크 레벨 분류 (headwind가 음수일 때 배풍)

    Parameters
    ----------
    headwind_kt : float
        정풍/배풍 성분 (kt). 음수면 배풍

    Returns
    -------
    str
        리스크 레벨 ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
    """
    tailwind = -headwind_kt if headwind_kt < 0 else 0
    if tailwind < TAILWIND_LIMITS["LOW"]:
        return "LOW"
    elif tailwind < TAILWIND_LIMITS["MODERATE"]:
        return "MODERATE"
    elif tailwind < TAILWIND_LIMITS["HIGH"]:
        return "HIGH"
    else:
        return "CRITICAL"


def calc_risk_score(crosswind_kt: float, headwind_kt: float) -> Tuple[int, str]:
    """교차풍 + 배풍 종합 리스크 점수 계산

    리스크 점수 기준:
    - 0-25: LOW (안전)
    - 26-50: MODERATE (주의)
    - 51-75: HIGH (경고)
    - 76-100: CRITICAL (위험)

    Parameters
    ----------
    crosswind_kt : float
        교차풍 절대값 (kt)
    headwind_kt : float
        정풍/배풍 성분 (kt). 음수면 배풍

    Returns
    -------
    Tuple[int, str]
        (리스크 점수 0-100, 리스크 레벨)
    """
    xw = abs(crosswind_kt)
    tailwind = -headwind_kt if headwind_kt < 0 else 0

    # 교차풍 점수 (0-100)
    if xw < CROSSWIND_LIMITS["LOW"]:
        xw_score = (xw / CROSSWIND_LIMITS["LOW"]) * 25
    elif xw < CROSSWIND_LIMITS["MODERATE"]:
        xw_score = 25 + ((xw - 10) / 5) * 25
    elif xw < CROSSWIND_LIMITS["HIGH"]:
        xw_score = 50 + ((xw - 15) / 5) * 25
    else:
        xw_score = min(75 + ((xw - 20) / 10) * 25, 100)

    # 배풍 점수 (0-100)
    if tailwind < TAILWIND_LIMITS["LOW"]:
        tw_score = (tailwind / TAILWIND_LIMITS["LOW"]) * 25
    elif tailwind < TAILWIND_LIMITS["MODERATE"]:
        tw_score = 25 + ((tailwind - 5) / 5) * 25
    elif tailwind < TAILWIND_LIMITS["HIGH"]:
        tw_score = 50 + ((tailwind - 10) / 5) * 25
    else:
        tw_score = min(75 + ((tailwind - 15) / 10) * 25, 100)

    # 종합 점수 (교차풍 70%, 배풍 30%)
    total_score = int(xw_score * 0.7 + tw_score * 0.3)
    total_score = min(max(total_score, 0), 100)

    # 레벨 결정
    if total_score <= 25:
        level = "LOW"
    elif total_score <= 50:
        level = "MODERATE"
    elif total_score <= 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return total_score, level


def calc_risk_score_array(
    crosswind: np.ndarray,
    headwind: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """배열 데이터에 대한 리스크 점수 계산

    Parameters
    ----------
    crosswind : np.ndarray
        교차풍 배열 (kt)
    headwind : np.ndarray
        정풍/배풍 배열 (kt)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (리스크 점수 배열, 리스크 레벨 배열)
    """
    scores = np.zeros(len(crosswind), dtype=int)
    levels = np.empty(len(crosswind), dtype=object)

    for i in range(len(crosswind)):
        scores[i], levels[i] = calc_risk_score(crosswind[i], headwind[i])

    return scores, levels


def get_risk_statistics(
    crosswind: np.ndarray,
    headwind: np.ndarray,
) -> dict:
    """리스크 통계 계산

    Parameters
    ----------
    crosswind : np.ndarray
        교차풍 배열 (kt)
    headwind : np.ndarray
        정풍/배풍 배열 (kt)

    Returns
    -------
    dict
        리스크 통계 딕셔너리
    """
    scores, levels = calc_risk_score_array(crosswind, headwind)

    n = len(scores)
    level_counts = {
        "LOW": np.sum(levels == "LOW"),
        "MODERATE": np.sum(levels == "MODERATE"),
        "HIGH": np.sum(levels == "HIGH"),
        "CRITICAL": np.sum(levels == "CRITICAL"),
    }

    return {
        "risk_score_mean": np.mean(scores),
        "risk_score_max": np.max(scores),
        "risk_score_p90": np.percentile(scores, 90),
        "risk_score_p95": np.percentile(scores, 95),
        "pct_low": 100 * level_counts["LOW"] / n,
        "pct_moderate": 100 * level_counts["MODERATE"] / n,
        "pct_high": 100 * level_counts["HIGH"] / n,
        "pct_critical": 100 * level_counts["CRITICAL"] / n,
    }


def load_airports(csv_path: str | Path = None) -> pd.DataFrame:
    """공항/활주로 CSV 로드

    Parameters
    ----------
    csv_path : str | Path, optional
        CSV 파일 경로. None이면 기본 경로(data/airports.csv) 사용

    Returns
    -------
    pd.DataFrame
        공항/활주로 정보 DataFrame
        columns: icao, name, lat, lon, runway, heading
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "airports.csv"
    return pd.read_csv(csv_path)


def analyze_crosswind(
    wind_df: pd.DataFrame,
    airports_df: pd.DataFrame,
    station_icao_map: dict = None,
) -> pd.DataFrame:
    """바람 데이터와 공항 정보를 결합하여 교차풍 통계 계산

    Parameters
    ----------
    wind_df : pd.DataFrame
        바람 데이터. 필수 컬럼: station, wspd, wdir
    airports_df : pd.DataFrame
        공항/활주로 정보. 필수 컬럼: icao, runway, heading
    station_icao_map : dict, optional
        관측소(station) → ICAO 코드 매핑.
        None이면 station 값이 ICAO 코드라고 가정

    Returns
    -------
    pd.DataFrame
        공항/활주로별 교차풍 통계
        columns: icao, runway, heading, n,
                 crosswind_mean, crosswind_abs_mean, crosswind_max, crosswind_p90,
                 headwind_mean, headwind_max, tailwind_max
    """
    results = []

    for _, airport in airports_df.iterrows():
        icao = airport["icao"]
        runway = airport["runway"]
        heading = airport["heading"]

        # 관측소 매핑
        if station_icao_map:
            stations = [k for k, v in station_icao_map.items() if v == icao]
        else:
            stations = [icao]

        # 해당 관측소 데이터 필터링
        mask = wind_df["station"].isin(stations)
        if mask.sum() == 0:
            continue

        subset = wind_df[mask].copy()
        wspd = subset["wspd"].values.astype(float)
        wdir = subset["wdir"].values.astype(float)

        # 유효 데이터만 사용
        valid = ~np.isnan(wspd) & ~np.isnan(wdir)
        wspd = wspd[valid]
        wdir = wdir[valid]

        if len(wspd) == 0:
            continue

        # 교차풍/정풍 계산
        crosswind = calc_crosswind(wspd, wdir, heading)
        headwind = calc_headwind(wspd, wdir, heading)

        # 통계 계산
        rec = {
            "icao": icao,
            "name": airport.get("name", ""),
            "runway": runway,
            "heading": heading,
            "n": len(wspd),
            # 교차풍 통계
            "crosswind_mean": np.mean(crosswind),
            "crosswind_abs_mean": np.mean(np.abs(crosswind)),
            "crosswind_std": np.std(crosswind),
            "crosswind_max": np.max(crosswind),
            "crosswind_min": np.min(crosswind),
            "crosswind_abs_max": np.max(np.abs(crosswind)),
            "crosswind_p90": np.percentile(np.abs(crosswind), 90),
            "crosswind_p95": np.percentile(np.abs(crosswind), 95),
            # 정풍/배풍 통계
            "headwind_mean": np.mean(headwind),
            "headwind_max": np.max(headwind),
            "tailwind_max": -np.min(headwind),  # 배풍 최대값 (양수로 표시)
            # 교차풍 초과 빈도 (kt 기준: 10kt, 15kt, 20kt)
            "exceed_10kt_pct": 100 * np.mean(np.abs(crosswind) > 10),
            "exceed_15kt_pct": 100 * np.mean(np.abs(crosswind) > 15),
            "exceed_20kt_pct": 100 * np.mean(np.abs(crosswind) > 20),
        }

        # 리스크 통계 추가
        risk_stats = get_risk_statistics(crosswind, headwind)
        rec.update(risk_stats)

        results.append(rec)

    return pd.DataFrame(results)


def get_crosswind_for_station(
    wind_df: pd.DataFrame,
    runway_heading: float,
) -> pd.DataFrame:
    """단일 관측소 데이터에 교차풍/정풍 컬럼 추가

    Parameters
    ----------
    wind_df : pd.DataFrame
        바람 데이터. 필수 컬럼: wspd, wdir
    runway_heading : float
        활주로 방향 (도)

    Returns
    -------
    pd.DataFrame
        crosswind, headwind 컬럼이 추가된 DataFrame
    """
    result = wind_df.copy()
    wspd = result["wspd"].values.astype(float)
    wdir = result["wdir"].values.astype(float)

    result["crosswind"] = calc_crosswind(wspd, wdir, runway_heading)
    result["headwind"] = calc_headwind(wspd, wdir, runway_heading)

    return result


# ──────────────────────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("교차풍 계산 모듈 테스트")
    print("=" * 60)

    # 테스트 1: 기본 계산 검증
    print("\n[테스트 1] 기본 계산 검증")
    print("-" * 40)

    # 활주로 방향 180도, 바람 270도(서풍), 풍속 10
    # 교차풍 = 10 * sin(270-180) = 10 * sin(90) = 10 (좌→우)
    # 정풍 = 10 * cos(270-180) = 10 * cos(90) = 0
    wspd = np.array([10.0])
    wdir = np.array([270.0])
    heading = 180.0

    xwind = calc_crosswind(wspd, wdir, heading)
    hwind = calc_headwind(wspd, wdir, heading)
    print(f"활주로: {heading}°, 바람: {wdir[0]}° @ {wspd[0]}m/s")
    print(f"교차풍: {xwind[0]:.2f} (예상: 10.0)")
    print(f"정풍:   {hwind[0]:.2f} (예상: 0.0)")

    # 테스트 2: 정풍 상황
    print("\n[테스트 2] 정풍 상황")
    print("-" * 40)

    # 활주로 방향 90도, 바람 90도(동풍), 풍속 15
    # 교차풍 = 15 * sin(90-90) = 0
    # 정풍 = 15 * cos(90-90) = 15
    wspd = np.array([15.0])
    wdir = np.array([90.0])
    heading = 90.0

    xwind = calc_crosswind(wspd, wdir, heading)
    hwind = calc_headwind(wspd, wdir, heading)
    print(f"활주로: {heading}°, 바람: {wdir[0]}° @ {wspd[0]}m/s")
    print(f"교차풍: {xwind[0]:.2f} (예상: 0.0)")
    print(f"정풍:   {hwind[0]:.2f} (예상: 15.0)")

    # 테스트 3: 배풍 상황
    print("\n[테스트 3] 배풍 상황")
    print("-" * 40)

    # 활주로 방향 90도, 바람 270도(서풍), 풍속 20
    # 교차풍 = 20 * sin(270-90) = 20 * sin(180) = 0
    # 정풍 = 20 * cos(270-90) = 20 * cos(180) = -20 (배풍)
    wspd = np.array([20.0])
    wdir = np.array([270.0])
    heading = 90.0

    xwind = calc_crosswind(wspd, wdir, heading)
    hwind = calc_headwind(wspd, wdir, heading)
    print(f"활주로: {heading}°, 바람: {wdir[0]}° @ {wspd[0]}m/s")
    print(f"교차풍: {xwind[0]:.2f} (예상: 0.0)")
    print(f"정풍:   {hwind[0]:.2f} (예상: -20.0, 배풍)")

    # 테스트 4: 리스크 점수 계산
    print("\n[테스트 4] 리스크 점수 계산")
    print("-" * 40)

    test_cases = [
        (5, 10, "LOW"),        # 교차풍 5kt, 정풍 10kt → LOW (score ~8)
        (14, -6, "MODERATE"),  # 교차풍 14kt, 배풍 6kt → MODERATE (score ~38)
        (18, -8, "HIGH"),      # 교차풍 18kt, 배풍 8kt → HIGH (score ~57)
        (25, -12, "CRITICAL"), # 교차풍 25kt, 배풍 12kt → CRITICAL (score ~79)
    ]

    print(f"{'Crosswind':>10} {'Head/Tail':>12} {'Score':>6} {'Level':>10} {'Expected':>10}")
    print("-" * 55)
    for xw, hw, expected in test_cases:
        score, level = calc_risk_score(xw, hw)
        status = "OK" if level == expected else "FAIL"
        hw_str = f"{hw}kt" if hw >= 0 else f"{hw}kt(tail)"
        print(f"{xw:>8}kt {hw_str:>12} {score:>6} {level:>10} {expected:>10} {status:>6}")

    # 테스트 5: 공항 데이터 로드
    print("\n[테스트 5] 공항 데이터 로드")
    print("-" * 40)

    try:
        airports = load_airports()
        print(f"로드된 공항/활주로 수: {len(airports)}")
        print("\n공항 목록:")
        for icao in airports["icao"].unique():
            name = airports[airports["icao"] == icao]["name"].iloc[0]
            runways = airports[airports["icao"] == icao]["runway"].tolist()
            print(f"  {icao} ({name}): {', '.join(runways)}")
    except FileNotFoundError:
        print("  airports.csv 파일을 찾을 수 없습니다.")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
