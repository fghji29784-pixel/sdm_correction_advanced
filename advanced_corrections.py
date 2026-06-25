# ─────────────────────────────────────────────────────────────
#  advanced_corrections.py  —  SDM 보정 고도화 모듈
#
#  구현된 보정법 (보정법_후보_조사.md 1차 도입 묶음):
#    A-1  Arrhenius 곱셈형 온도 보정  (물리 · ★★★)
#    C-1  Rwiring → τ 채널별 보정     (물리 · ★★★)
#    D-2  공간 추세면(2차 다항)        (통계 · ★★★)
#    D-1  혼합효과 모델(MixedLM)      (통계 · ★★★)
#
#  사용법:
#    >>> import sys; sys.path.insert(0, r"C:\Users\fghji\Desktop\sdm_logic")
#    >>> from advanced_corrections import run_analysis_advanced, compare_with_baseline
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# ── sdm_logic 경로 자동 추가 ──────────────────────────────────
_DESKTOP    = pathlib.Path.home() / "Desktop"
_SDM_LOGIC  = _DESKTOP / "sdm_logic"
if _SDM_LOGIC.exists() and str(_SDM_LOGIC) not in sys.path:
    sys.path.insert(0, str(_SDM_LOGIC))

from constants import (
    NUM_LAYERS, REF_V_INIT, REF_T_FINAL, REF_DELTA_T,
    PROCESS_COL_GRADE,
)
from analysis import (
    resolve_ocv_columns,
    remove_outliers,
    prepare_y,
    add_layer_dummies,
    correct_values,
    standardize,
    compute_metrics,
    docv_surrogate_analysis,
    run_analysis_per_tray,
)


# ══════════════════════════════════════════════════════════════
#  A-1. Arrhenius 곱셈형 온도 보정
# ══════════════════════════════════════════════════════════════

# k_B in eV/K (볼츠만 상수)
_KB_EV = 8.617333e-5

# ⚠ 보정법_후보_조사.md 의 exp(k*(1/T_ref - 1/T)) 는 부호가 반대임.
# 물리 검증: 고온(30°C)→기준(25°C) 환산 시 ISD 가 낮아져야 하므로 factor < 1 이어야 함.
# 올바른 식: ISD(T_ref) = ISD(T) * exp[(Eact/k_B) * (1/T_meas - 1/T_ref)]
#            T_meas > T_ref → (1/T_meas - 1/T_ref) < 0 → factor < 1  ✓
# CLAUDE.md 기재 형식과 동일: ISD(T2) = ISD(T1) * exp[(Eact/R)*(1/T1 - 1/T2)]

def arrhenius_normalize(
    i_meas: pd.Series,
    t_meas_c: pd.Series,
    t_ref_c: float = 25.0,
    eact_ev: float = 0.94,
) -> pd.Series:
    """
    측정 전류를 기준 온도로 Arrhenius 환산 (A-1).

    I_corr = I_meas * exp[(Eact/k_B) * (1/T_meas_K - 1/T_ref_K)]

    T_meas > T_ref (셀이 더 따뜻함) → factor < 1 (기준 온도에서의 ISD로 낮춤)
    T_meas < T_ref (셀이 더 차가움) → factor > 1

    Parameters
    ----------
    i_meas   : 측정 전류 (단위 무관 — A 또는 µA 모두 가능)
    t_meas_c : 측정 온도 Series (°C)
    t_ref_c  : 기준 온도 (°C), 기본값 25°C
    eact_ev  : 활성화에너지 (eV), R-Smith 2023 기준 0.94 eV
    """
    T_meas_K = pd.to_numeric(t_meas_c, errors="coerce") + 273.15
    T_ref_K  = t_ref_c + 273.15
    factor   = np.exp((eact_ev / _KB_EV) * (1.0 / T_meas_K - 1.0 / T_ref_K))
    return i_meas * factor


# ══════════════════════════════════════════════════════════════
#  C-1. Rwiring → τ 채널별 시간상수 보정
# ══════════════════════════════════════════════════════════════

def rwiring_tau_normalize(
    i_meas: pd.Series,
    rwiring: pd.Series,
    rout: float,
    rser: float = 0.05,
    rwiring_ref: float | None = None,
) -> pd.Series:
    """
    채널별 Rwiring 편차로 인한 τ 차이를 보정 (C-1).

    선형 구간: I(t) ≈ ISD * t/τ_ch,  τ_ch = (Rout + Rser + Rwiring_ch) * Ceff
    기준 τ_ref = (Rout + Rser + Rwiring_ref) * Ceff

    → I_corr = I_meas * (τ_ch / τ_ref)
               = I_meas * (Rout + Rser + Rwiring_ch) / (Rout + Rser + Rwiring_ref)

    Rwiring_ch > Rwiring_ref → τ_ch 크다 → I_meas 가 억제됨 → 스케일 업(ratio > 1)
    Rwiring_ch < Rwiring_ref → τ_ch 작다 → I_meas 가 과대됨 → 스케일 다운(ratio < 1)

    Parameters
    ----------
    i_meas       : 측정 전류 Series
    rwiring      : 채널별 배선 저항 Series (Ω)
    rout         : 설정 출력 저항 (Ω) — KSS TestSetupPerChan 에서 읽거나 직접 지정
    rser         : 내부 직렬 저항 추정값 (Ω), 기본 0.05 Ω
    rwiring_ref  : 기준 Rwiring (Ω). None 이면 트레이 중앙값 사용.
    """
    rw = pd.to_numeric(rwiring, errors="coerce")
    if rw.isna().all():
        return i_meas  # Rwiring 정보 없으면 보정 생략

    rw = rw.fillna(rw.median())
    if rwiring_ref is None:
        rwiring_ref = float(rw.median())

    tau_ch  = rout + rser + rw
    tau_ref = rout + rser + rwiring_ref
    ratio   = tau_ch / tau_ref

    return i_meas * ratio


# ══════════════════════════════════════════════════════════════
#  D-2. 공간 추세면 — 레이어 더미 대체
# ══════════════════════════════════════════════════════════════

SPATIAL_FEATURES = ["sp_row", "sp_col", "sp_row2", "sp_col2", "sp_rowcol"]


def add_spatial_poly_features(
    df: pd.DataFrame,
    cell_col: str = "cell_no",
    tray_rows: int = 12,
) -> pd.DataFrame:
    """
    셀 번호 → (row, col) 좌표 변환 후 2차 다항 공간 특성 5개 추가 (D-2).

    추가 컬럼: sp_row, sp_col, sp_row², sp_col², sp_row·col
    (중앙 정렬: 0-based 0~11 → -5.5~5.5)

    레이어 이산 더미 5개 대신 이 5개로 연속 온도 구배를 포착.
    """
    df = df.copy()
    cell = pd.to_numeric(df[cell_col], errors="coerce").fillna(1).astype(int)

    row = ((cell - 1) // tray_rows).astype(float) - 5.5   # -5.5 ~ 5.5
    col = ((cell - 1) %  tray_rows).astype(float) - 5.5

    df["sp_row"]    = row
    df["sp_col"]    = col
    df["sp_row2"]   = row ** 2
    df["sp_col2"]   = col ** 2
    df["sp_rowcol"] = row * col

    return df


# ══════════════════════════════════════════════════════════════
#  D-1. 혼합효과 모델 (MixedLM)
# ══════════════════════════════════════════════════════════════

def fit_mixed_lm(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> dict:
    """
    트레이 랜덤절편 혼합효과 모델 (D-1).

    I_ij = β·x_ij + u_tray[j] + ε_ij,   u_tray ~ N(0, σ²_tray)

    공통 기울기(고정효과)를 전 트레이에서 공유(부분 풀링) →
    소표본 트레이 계수 불안정 해소, 트레이 레벨차는 랜덤절편이 흡수.

    트레이 수가 2~3개로 매우 적을 경우 σ²_tray 추정 불안정 가능 →
    수렴 실패 시 OLS 로 자동 폴백.

    Parameters
    ----------
    X      : 고정효과 독립변수 DataFrame
    y      : 종속변수 Series (µA)
    groups : 트레이 ID Series (랜덤절편 그룹)

    Returns
    -------
    dict with keys: model, params(고정효과), random_effects, converged, type
    """
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(f"MixedLM은 트레이 ≥ 2개 필요 (현재 {n_groups}개).")

    Xc = sm.add_constant(X, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm_model = sm.MixedLM(y, Xc, groups=groups)
            result    = lmm_model.fit(reml=True, method="lbfgs")
            converged = bool(result.converged)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lmm_model = sm.MixedLM(y, Xc, groups=groups)
                result    = lmm_model.fit(reml=False, method="nm")
            converged = False
        except Exception as e:
            raise RuntimeError(f"MixedLM 수렴 실패: {e}") from e

    params = result.fe_params.drop("const", errors="ignore")

    return {
        "type"           : "MixedLM",
        "model"          : result,
        "params"         : params,
        "random_effects" : result.random_effects,   # {tray_id: {'Group': intercept}}
        "converged"      : converged,
        "aic"            : float(result.aic),
        "n_groups"       : n_groups,
    }


# ══════════════════════════════════════════════════════════════
#  통합 고도화 파이프라인
# ══════════════════════════════════════════════════════════════

def run_analysis_advanced(
    df_meta: pd.DataFrame,
    n_minutes: int,
    dep_type: str = "single",
    use_arrhenius: bool = True,
    use_rwiring_tau: bool = True,
    use_spatial_poly: bool = True,
    use_mixed_lm: bool = True,
    rout: float = 2.0,
    rser: float = 0.05,
    eact_ev: float = 0.94,
    t_ref_c: float = 25.0,
    rwiring_threshold: float | None = None,
    ref_conditions: dict | None = None,
) -> dict:
    """
    A-1 + C-1 (물리 전처리) → D-2 (공간 다항) + D-1 (MixedLM) 통합 파이프라인.

    흐름:
      1. OCV 컬럼 정규화 + 이상치 제거
      2. 종속변수 준비 (→ µA)
      3. [A-1] Arrhenius 온도 보정 적용
      4. [C-1] Rwiring τ 보정 적용
      5. [D-2] 공간 다항 특성 추가 (레이어 더미 대체)
      6. [D-1] MixedLM 적합 (폴백: 트레이별 OLS)
      7. 보정값 계산 → z-score 표준화 → 평가지표

    Parameters
    ----------
    df_meta          : parser.py 가 생성한 셀 메타 DataFrame
    n_minutes        : 판정 시간 (분)
    dep_type         : 'single' 또는 'slope'
    use_arrhenius    : A-1 Arrhenius 보정 활성화
    use_rwiring_tau  : C-1 Rwiring τ 보정 활성화
    use_spatial_poly : D-2 공간 다항 특성 활성화 (False 시 레이어 더미 유지)
    use_mixed_lm     : D-1 MixedLM 활성화 (False 시 트레이별 OLS)
    rout             : 출력 저항 (Ω) — C-1 에 사용
    rser             : 내부 직렬 저항 추정값 (Ω) — C-1 에 사용
    eact_ev          : 활성화에너지 (eV) — A-1 에 사용
    t_ref_c          : 기준 온도 (°C) — A-1 에 사용
    rwiring_threshold: 이상치 제거 Rwiring 임계값 (Ω)
    ref_conditions   : 보정 기준 조건 dict (v_init, t_final, delta_t 등)

    Returns
    -------
    dict with keys:
      feature_cols, df_valid, model_info,
      y_raw (물리 보정 전), y_phys (물리 보정 후),
      corrected, z_scores, metrics, corrections_applied
    """
    # 1. 전처리
    df = resolve_ocv_columns(df_meta.copy())
    df = remove_outliers(df, rwiring_threshold=rwiring_threshold)

    # 2. 종속변수 (µA)
    y_raw = prepare_y(df, n_minutes, dep_type)
    df["y_raw"] = y_raw

    # 3. A-1: Arrhenius 온도 보정
    y_phys = y_raw.copy()
    arrhenius_applied = False
    if use_arrhenius and "t_final" in df.columns:
        y_phys = arrhenius_normalize(
            y_phys,
            t_meas_c=df["t_final"],
            t_ref_c=t_ref_c,
            eact_ev=eact_ev,
        )
        arrhenius_applied = True
    df["y_phys"] = y_phys

    # 4. C-1: Rwiring τ 보정
    tau_applied = False
    if use_rwiring_tau and "rwiring" in df.columns:
        y_phys = rwiring_tau_normalize(
            y_phys,
            rwiring=df["rwiring"],
            rout=rout,
            rser=rser,
        )
        tau_applied = True
    df["y_phys"] = y_phys

    # 5. D-2: 공간 특성 / 레이어 더미
    if use_spatial_poly and "cell_no" in df.columns:
        df = add_spatial_poly_features(df)
        spatial_feats = SPATIAL_FEATURES
    else:
        df = add_layer_dummies(df)
        spatial_feats = [f"dummy_L{i}" for i in range(2, NUM_LAYERS + 1)]

    # 독립변수 구성:
    # A-1 으로 온도가 물리적으로 제거됐으면 t_final/delta_t 를 회귀에서 제외.
    # (회귀의 선형 온도항이 이미 보정된 잔차를 흡수할 수 있으므로 유지도 가능하나,
    #  물리적 해석 일관성을 위해 제외가 권장.)
    base_sdm = ["v_init"]
    if not arrhenius_applied:
        base_sdm += ["t_final", "delta_t"]

    feature_cols = [f for f in (base_sdm + spatial_feats) if f in df.columns]

    y = df["y_phys"]
    X = df[feature_cols]
    valid = X.notna().all(axis=1) & y.notna()
    Xv, yv, dfv = X[valid], y[valid], df[valid]

    if len(yv) < 10:
        raise ValueError(f"유효 셀 수 부족: {len(yv)}개")

    # 6. 모델 적합
    has_tray     = "tray_id" in dfv.columns and dfv["tray_id"].nunique() >= 2
    mixed_applied = False

    if use_mixed_lm and has_tray:
        try:
            lmm       = fit_mixed_lm(Xv, yv, groups=dfv["tray_id"])
            params    = lmm["params"]
            model_info = lmm
            mixed_applied = True
        except Exception as e:
            # 수렴 실패 → OLS 폴백
            ols       = sm.OLS(yv, sm.add_constant(Xv, has_constant="add"),
                               missing="drop").fit()
            params    = ols.params.drop("const", errors="ignore")
            model_info = {"type": "OLS_fallback", "model": ols, "fallback_reason": str(e)}
    else:
        ols       = sm.OLS(yv, sm.add_constant(Xv, has_constant="add"),
                           missing="drop").fit()
        params    = ols.params.drop("const", errors="ignore")
        model_info = {"type": "OLS", "model": ols}

    # 7. 보정값 계산 (기준: 모든 공간 특성 = 중앙(0), v_init = 기준값)
    rc    = ref_conditions or {}
    ref_default: dict = {
        "v_init"  : rc.get("v_init",   REF_V_INIT),
        "t_final" : rc.get("t_final",  REF_T_FINAL),
        "delta_t" : rc.get("delta_t",  REF_DELTA_T),
        **{f: 0.0 for f in SPATIAL_FEATURES},
    }

    corrected = correct_values(yv, Xv, params, ref=ref_default)
    z_scores  = standardize(corrected)

    # 평가지표
    grade_col   = PROCESS_COL_GRADE
    true_labels = dfv[grade_col] if grade_col in dfv.columns else None
    metrics     = compute_metrics(z_scores, true_labels)

    return {
        "feature_cols"      : feature_cols,
        "df_valid"          : dfv,
        "model_info"        : model_info,
        "params"            : params,
        "y_raw"             : y_raw[valid],
        "y_phys"            : yv,
        "corrected"         : corrected,
        "z_scores"          : z_scores,
        "metrics"           : metrics,
        "corrections_applied": {
            "A1_arrhenius"   : arrhenius_applied,
            "C1_rwiring_tau" : tau_applied,
            "D2_spatial_poly": use_spatial_poly,
            "D1_mixed_lm"    : mixed_applied,
        },
    }


# ══════════════════════════════════════════════════════════════
#  기존 분석과 비교
# ══════════════════════════════════════════════════════════════

def compare_with_baseline(
    df_meta: pd.DataFrame,
    n_minutes: int = 15,
    option_baseline: int = 3,
    print_summary: bool = True,
    **advanced_kwargs,
) -> dict:
    """
    기존 파이프라인(run_analysis_per_tray) vs 고도화 결과 비교.

    검증 지표: d_prime, SDM↔dOCV Pearson/Spearman (양품만 포함)
    CLAUDE.md v3: d'/AUC 보다 dOCV 연속 상관이 핵심 지표임에 유의.

    Returns
    -------
    dict: baseline, advanced, dp_delta, docv_corr_delta
    """
    baseline = run_analysis_per_tray(
        df_meta, option=option_baseline, n_minutes=n_minutes
    )
    advanced = run_analysis_advanced(df_meta, n_minutes=n_minutes, **advanced_kwargs)

    dp_base = baseline["metrics"].get("d_prime", float("nan"))
    dp_adv  = advanced["metrics"].get("d_prime", float("nan"))

    # dOCV 상관 비교
    from constants import PROCESS_COL_DOCV
    docv_base = docv_surrogate_analysis(
        baseline["df_valid"], baseline["corrected"], PROCESS_COL_DOCV
    )
    docv_adv  = docv_surrogate_analysis(
        advanced["df_valid"], advanced["corrected"], PROCESS_COL_DOCV
    )

    r_base = docv_base.get("pearson_normal", float("nan")) if docv_base else float("nan")
    r_adv  = docv_adv.get("pearson_normal", float("nan")) if docv_adv  else float("nan")

    result = {
        "baseline"         : baseline,
        "advanced"         : advanced,
        "dp_baseline"      : dp_base,
        "dp_advanced"      : dp_adv,
        "dp_delta"         : dp_adv - dp_base,
        "docv_r_baseline"  : r_base,
        "docv_r_advanced"  : r_adv,
        "docv_r_delta"     : r_adv - r_base,
        "corrections"      : advanced.get("corrections_applied", {}),
    }

    if print_summary:
        _print_comparison(result, n_minutes, option_baseline)

    return result


def _print_comparison(result: dict, n_minutes: int, option_baseline: int) -> None:
    sep = "-" * 52
    print(sep)
    print(f"  SDM 보정 고도화 비교  ({n_minutes}분 판정)")
    print(sep)

    corr = result["corrections"]
    print("  적용 보정:")
    for key, val in corr.items():
        mark = "[O]" if val else "[X]"
        print(f"    {mark}  {key}")

    print()
    label_baseline = f"기존(옵션{option_baseline})"
    print(f"  {'지표':<18} {label_baseline:<14} {'고도화':<14} {'delta'}")
    dp_b = result["dp_baseline"]
    dp_a = result["dp_advanced"]
    r_b  = result["docv_r_baseline"]
    r_a  = result["docv_r_advanced"]
    print(f"  {'d_prime':<18} {dp_b:<14.3f} {dp_a:<14.3f} {dp_a - dp_b:+.3f}")
    print(f"  {'dOCV r(양품만)':<18} {r_b:<14.3f} {r_a:<14.3f} {r_a - r_b:+.3f}")
    print(sep)
