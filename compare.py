# ─────────────────────────────────────────────────────────────
#  compare.py  —  보정 구성별 파이프라인 + 비교
#
#  보정법_후보_조사.md 종합 권장안을 단계적으로 켜가며 비교:
#    [0] Baseline      : 보정 없음 (원시 N분 전류)
#    [1] +A1           : Arrhenius 온도 보정만
#    [2] +A1+C1        : 물리 보정 (온도 + Rwiring τ)
#    [3] +A1+C1+D2     : 물리 + 공간 추세면 회귀
#    [4] +A1+C1+D2+D1  : 물리 + 공간 + 혼합효과 (권장 1차 도입 묶음)
#
#  비교 지표:
#    d_prime          — 불량/양품 분리도 (불량 라벨 있을 때)
#    dOCV r(양품만)    — SDM↔dOCV 연속 상관 (CLAUDE.md v3 핵심 지표)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from constants import (
    DEFAULT_ROUT, DEFAULT_RSER, EACT_EV,
    REF_V_INIT, REF_T_FINAL, REF_DELTA_T,
    PROCESS_COL_GRADE, PROCESS_COL_DOCV, NUM_LAYERS,
)
from corrections import (
    prepare_y, arrhenius_normalize, rwiring_tau_normalize,
    add_spatial_poly_features, add_layer_dummies, SPATIAL_FEATURES,
    fit_mixed_lm, correct_values, standardize, standardize_per_tray,
    compute_metrics, docv_surrogate,
)


# 보정 구성 정의 (이름, A1, C1, D2공간, D1혼합)
CONFIGS = [
    ('[0] Baseline',         False, False, False, False),
    ('[1] +A1',              True,  False, False, False),
    ('[2] +A1+C1',           True,  True,  False, False),
    ('[3] +A1+C1+D2',        True,  True,  True,  False),
    ('[4] +A1+C1+D2+D1',     True,  True,  True,  True),
]


def run_config(df_meta: pd.DataFrame, n_minutes: int,
               use_a1: bool, use_c1: bool, use_d2: bool, use_d1: bool,
               dep_type: str = 'single',
               rout: float = DEFAULT_ROUT, rser: float = DEFAULT_RSER,
               eact_ev: float = EACT_EV, t_ref_c: float = REF_T_FINAL,
               ref_conditions: dict | None = None) -> dict:
    """단일 보정 구성 실행 → 보정값·z-score·지표."""
    df = df_meta.copy()

    # OCV 컬럼 정규화 (회귀 미사용이지만 호환용)
    y = prepare_y(df, n_minutes, dep_type)

    # ── A-1: Arrhenius 온도 보정 ──
    if use_a1 and 't_final' in df.columns:
        y = arrhenius_normalize(y, df['t_final'], t_ref_c=t_ref_c, eact_ev=eact_ev)

    # ── C-1: Rwiring τ 보정 ──
    if use_c1 and 'rwiring' in df.columns:
        y = rwiring_tau_normalize(y, df['rwiring'], rout=rout, rser=rser)

    df['_y'] = y

    # ── 회귀 독립변수 구성 ──
    # A-1로 온도가 물리 제거됐으면 회귀에서 온도항 제외 (해석 일관성)
    base_feats = ['v_init']
    if not use_a1:
        base_feats += ['t_final', 'delta_t']

    if use_d2:
        df = add_spatial_poly_features(df)
        spatial_feats = [f for f in SPATIAL_FEATURES if f in df.columns]
    else:
        df = add_layer_dummies(df)
        spatial_feats = [f'dummy_L{i}' for i in range(2, NUM_LAYERS + 1) if f'dummy_L{i}' in df.columns]

    feature_cols = [c for c in (base_feats + spatial_feats) if c in df.columns]

    yv_all = df['_y']
    X = df[feature_cols] if feature_cols else pd.DataFrame(index=df.index)
    valid = yv_all.notna() & (X.notna().all(axis=1) if feature_cols else True)
    Xv, yv, dfv = X[valid], yv_all[valid], df[valid]

    has_tray = 'tray_id' in dfv.columns and dfv['tray_id'].nunique() >= 2

    # ── 회귀 적합 ──
    model_type = 'none'
    if not feature_cols:
        # Baseline: 회귀 없이 측정값 그대로
        corrected = yv.copy()
        model_type = 'baseline'
    else:
        if use_d1 and has_tray:
            try:
                lmm = fit_mixed_lm(Xv, yv, groups=dfv['tray_id'])
                params = lmm['params']
                model_type = 'MixedLM' if lmm['converged'] else 'MixedLM(non-converged→OLS-like)'
            except Exception:
                ols = sm.OLS(yv, sm.add_constant(Xv, has_constant='add'), missing='drop').fit()
                params = ols.params.drop('const', errors='ignore')
                model_type = 'OLS(fallback)'
        else:
            ols = sm.OLS(yv, sm.add_constant(Xv, has_constant='add'), missing='drop').fit()
            params = ols.params.drop('const', errors='ignore')
            model_type = 'OLS'

        ref = {'v_init': (ref_conditions or {}).get('v_init', REF_V_INIT),
               't_final': REF_T_FINAL, 'delta_t': REF_DELTA_T,
               **{f: 0.0 for f in SPATIAL_FEATURES}}
        corrected = correct_values(yv, Xv, params, ref=ref)

    # ── 표준화 (트레이별 중심) ──
    if has_tray:
        z = standardize_per_tray(corrected, dfv['tray_id'])
    else:
        z = standardize(corrected)

    # ── 지표 ──
    labels  = dfv[PROCESS_COL_GRADE] if PROCESS_COL_GRADE in dfv.columns else None
    metrics = compute_metrics(z, labels)
    docv    = docv_surrogate(dfv, corrected, PROCESS_COL_DOCV)

    return {
        'corrected': corrected, 'z_scores': z, 'df_valid': dfv,
        'metrics': metrics, 'docv': docv, 'model_type': model_type,
        'feature_cols': feature_cols,
    }


def compare_all(df_meta: pd.DataFrame, n_minutes: int = 15,
                dep_type: str = 'single',
                rout: float = DEFAULT_ROUT, rser: float = DEFAULT_RSER,
                eact_ev: float = EACT_EV, t_ref_c: float = REF_T_FINAL) -> pd.DataFrame:
    """모든 보정 구성 실행 → 비교 DataFrame 반환."""
    rows = []
    results = {}
    for name, a1, c1, d2, d1 in CONFIGS:
        try:
            res = run_config(df_meta, n_minutes, a1, c1, d2, d1,
                             dep_type=dep_type, rout=rout, rser=rser,
                             eact_ev=eact_ev, t_ref_c=t_ref_c)
            results[name] = res
            dp   = res['metrics'].get('d_prime', np.nan)
            auc  = res['metrics'].get('auc', np.nan)
            docv = res['docv'] or {}
            rows.append({
                '구성': name,
                'd_prime': dp,
                'AUC': auc,
                'dOCV_r_전체': docv.get('pearson', np.nan),
                'dOCV_r_양품만': docv.get('pearson_normal', np.nan),
                'dOCV_spearman_양품만': docv.get('spearman_normal', np.nan),
                '모델': res['model_type'],
            })
        except Exception as e:
            rows.append({'구성': name, 'd_prime': np.nan, 'AUC': np.nan,
                         'dOCV_r_전체': np.nan, 'dOCV_r_양품만': np.nan,
                         'dOCV_spearman_양품만': np.nan, '모델': f'실패: {e}'})

    table = pd.DataFrame(rows)
    table.attrs['results'] = results
    return table


def print_table(table: pd.DataFrame, n_minutes: int) -> None:
    sep = '=' * 78
    print(sep)
    print(f'  SDM 보정법 비교  ({n_minutes}분 판정)   [출처: 보정법_후보_조사.md 종합권장안]')
    print(sep)
    fmt = lambda v: f'{v:7.3f}' if isinstance(v, (int, float)) and not pd.isna(v) else '   nan'
    hdr = f'{"구성":<20}{"d_prime":>9}{"AUC":>8}{"dOCV_r전체":>12}{"dOCV_r양품":>12}'
    print(hdr)
    print('-' * 78)
    for _, r in table.iterrows():
        print(f'{r["구성"]:<20}{fmt(r["d_prime"]):>9}{fmt(r["AUC"]):>8}'
              f'{fmt(r["dOCV_r_전체"]):>12}{fmt(r["dOCV_r_양품만"]):>12}   {r["모델"]}')
    print(sep)
    print('  해석 가이드:')
    print('   · d_prime ↑  = 불량/양품 분리 우수 (불량 라벨 있을 때만 유효)')
    print('   · dOCV_r양품 ↑ = 양품 범위 안에서도 dOCV 추종 → 미세불량 선별 가능(진짜 대체재)')
    print('   · CLAUDE.md v3: d_prime/AUC 는 불량 소수 의존 → dOCV_r양품 을 우선 지표로 볼 것')
    print(sep)
