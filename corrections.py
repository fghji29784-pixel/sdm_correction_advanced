# ─────────────────────────────────────────────────────────────
#  corrections.py  —  보정법 구현 + 평가 헬퍼 (자체 포함)
#
#  보정법_후보_조사.md 종합 권장안(1차 도입 묶음):
#    A-1  Arrhenius 곱셈형 온도 보정   (물리 · ★★★)
#    C-1  Rwiring → τ 채널별 보정      (물리 · ★★★)
#    D-2  공간 추세면(2차 다항)         (통계 · ★★★)
#    D-1  혼합효과 모델(MixedLM)       (통계 · ★★★)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from constants import (
    NUM_LAYERS, EACT_EV, KB_EV, DEFAULT_RSER,
    REF_V_INIT, REF_T_FINAL, REF_DELTA_T, DOCV_OFFSET,
    PROCESS_COL_GRADE,
)

warnings.filterwarnings('ignore', category=FutureWarning)


# ══════════════════════════════════════════════════════════════
#  종속변수 준비
# ══════════════════════════════════════════════════════════════

def prepare_y(df: pd.DataFrame, n_minutes: int, dep_type: str = 'single') -> pd.Series:
    """N분 전류(single) 또는 0~N분 기울기(slope) → µA 단위 Series."""
    col = f'i_{n_minutes}min' if dep_type == 'single' else f'slope_0_{n_minutes}'
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors='coerce') * 1e6   # A → µA


# ══════════════════════════════════════════════════════════════
#  A-1. Arrhenius 곱셈형 온도 보정
# ══════════════════════════════════════════════════════════════

def arrhenius_normalize(i_meas: pd.Series, t_meas_c: pd.Series,
                        t_ref_c: float = 25.0, eact_ev: float = EACT_EV) -> pd.Series:
    """
    I_corr = I_meas * exp[(Eact/k_B) * (1/T_meas_K - 1/T_ref_K)]

    T_meas > T_ref → factor < 1  (고온 셀을 기준온도 ISD로 낮춤)
    부호 검증: CLAUDE.md ISD(T2)=ISD(T1)*exp[(Eact/R)(1/T1-1/T2)] 와 동일.
    """
    T_meas = pd.to_numeric(t_meas_c, errors='coerce') + 273.15
    T_ref  = t_ref_c + 273.15
    factor = np.exp((eact_ev / KB_EV) * (1.0 / T_meas - 1.0 / T_ref))
    return i_meas * factor


# ══════════════════════════════════════════════════════════════
#  C-1. Rwiring → τ 채널별 시간상수 보정
# ══════════════════════════════════════════════════════════════

def rwiring_tau_normalize(i_meas: pd.Series, rwiring: pd.Series, rout: float,
                          rser: float = DEFAULT_RSER,
                          rwiring_ref: float | None = None) -> pd.Series:
    """
    I_corr = I_meas * (Rout+Rser+Rwiring_ch) / (Rout+Rser+Rwiring_ref)

    Rwiring 큼 → τ 큼 → 같은 시간에 전류 작게 측정됨 → 스케일 업(ratio>1).
    """
    rw = pd.to_numeric(rwiring, errors='coerce')
    if rw.isna().all():
        return i_meas
    rw = rw.fillna(rw.median())
    if rwiring_ref is None:
        rwiring_ref = float(rw.median())
    ratio = (rout + rser + rw) / (rout + rser + rwiring_ref)
    return i_meas * ratio


# ══════════════════════════════════════════════════════════════
#  D-2. 공간 추세면 (레이어 더미 대체)
# ══════════════════════════════════════════════════════════════

SPATIAL_FEATURES = ['sp_row', 'sp_col', 'sp_row2', 'sp_col2', 'sp_rowcol']


def add_spatial_poly_features(df: pd.DataFrame, cell_col: str = 'cell_no',
                              tray_rows: int = 12) -> pd.DataFrame:
    """셀 번호 → (row,col) 중앙정렬 좌표 → 2차 다항 5개 추가."""
    df = df.copy()
    cell = pd.to_numeric(df[cell_col], errors='coerce').fillna(1).astype(int)
    row = ((cell - 1) // tray_rows).astype(float) - 5.5
    col = ((cell - 1) %  tray_rows).astype(float) - 5.5
    df['sp_row'], df['sp_col'] = row, col
    df['sp_row2'], df['sp_col2'] = row ** 2, col ** 2
    df['sp_rowcol'] = row * col
    return df


def add_layer_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(2, NUM_LAYERS + 1):
        df[f'dummy_L{i}'] = (df['layer'] == i).astype(float)
    return df


# ══════════════════════════════════════════════════════════════
#  D-1. 혼합효과 모델 (MixedLM)
# ══════════════════════════════════════════════════════════════

def fit_mixed_lm(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    """트레이 랜덤절편 MixedLM. 수렴 실패/singular → converged=False."""
    if groups.nunique() < 2:
        raise ValueError(f'MixedLM은 트레이 ≥ 2개 필요 (현재 {groups.nunique()}개).')
    Xc = sm.add_constant(X, has_constant='add')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.MixedLM(y, Xc, groups=groups).fit(reml=True, method='lbfgs')
        converged = bool(res.converged)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.MixedLM(y, Xc, groups=groups).fit(reml=False, method='nm')
        converged = False
    return {
        'type': 'MixedLM', 'model': res,
        'params': res.fe_params.drop('const', errors='ignore'),
        'converged': converged, 'aic': float(res.aic),
    }


# ══════════════════════════════════════════════════════════════
#  보정값 / 표준화 / 평가
# ══════════════════════════════════════════════════════════════

def correct_values(y: pd.Series, X: pd.DataFrame, params: pd.Series,
                   ref: dict | None = None) -> pd.Series:
    """I_corrected = I - Σ β_k (x_k - x_k_ref)"""
    default_ref = {
        'v_init': REF_V_INIT, 't_final': REF_T_FINAL, 'delta_t': REF_DELTA_T,
        **{f: 0.0 for f in SPATIAL_FEATURES},
        **{f'dummy_L{i}': 0.0 for i in range(2, NUM_LAYERS + 1)},
        **{f'OCV{i}': 0.0 for i in [1, 2, 3, 4, 7]},
    }
    if ref:
        default_ref.update(ref)
    corr = pd.Series(0.0, index=y.index)
    for feat in X.columns:
        if feat in params.index:
            corr += params[feat] * (X[feat] - default_ref.get(feat, 0.0))
    return y - corr


def standardize(corrected: pd.Series, center: str = 'median') -> pd.Series:
    loc   = corrected.median() if center == 'median' else corrected.mean()
    sigma = corrected.std(ddof=1)
    return (corrected - loc) / (sigma + 1e-10)


def standardize_per_tray(corrected: pd.Series, tray: pd.Series,
                         center: str = 'median') -> pd.Series:
    """트레이별 중심·분산으로 z-score (트레이 간 레벨차 흡수)."""
    out = pd.Series(index=corrected.index, dtype=float)
    for tid, idx in tray.groupby(tray).groups.items():
        seg = corrected.loc[idx]
        loc = seg.median() if center == 'median' else seg.mean()
        out.loc[idx] = (seg - loc) / (seg.std(ddof=1) + 1e-10)
    return out


def compute_metrics(z: pd.Series, labels: pd.Series | None) -> dict:
    """d_prime, AUC. 라벨/불량 없으면 빈 dict."""
    metrics: dict = {}
    if labels is None or labels.isna().all():
        return metrics
    binary = (labels.astype(str).str.upper() == 'E').astype(int)
    valid  = ~(z.isna() | binary.isna())
    zz, bb = z[valid], binary[valid]
    if bb.sum() == 0:
        metrics['note'] = '불량셀(E) 없음'
        return metrics
    g, b = zz[bb == 0], zz[bb == 1]
    sig_g = g.std(ddof=1) if len(g) > 1 else 1.0
    metrics.update({
        'd_prime': float((b.mean() - g.mean()) / (sig_g + 1e-10)),
        'n_good': int((bb == 0).sum()), 'n_bad': int(bb.sum()),
    })
    try:
        from sklearn.metrics import roc_auc_score
        metrics['auc'] = float(roc_auc_score(bb, zz))
    except Exception:
        metrics['auc'] = np.nan
    return metrics


def _pearson(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return float('nan')
    return float(np.corrcoef(a.values, b.values)[0, 1])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return _pearson(a.rank(), b.rank())


def docv_surrogate(df_valid: pd.DataFrame, corrected: pd.Series,
                   docv_col: str, docv_offset: float = DOCV_OFFSET) -> dict | None:
    """
    SDM_corrected ↔ dOCV 연속 상관 (CLAUDE.md v3 핵심 검증 지표).
    트레이별 median 중심 정렬 후 Pearson/Spearman (전체 + 양품만).
    """
    if docv_col not in df_valid.columns:
        return None
    docv = pd.to_numeric(df_valid[docv_col], errors='coerce')
    sdm  = pd.to_numeric(corrected, errors='coerce')
    if 'tray_id' in df_valid.columns:
        tray = df_valid['tray_id']
        docv_c = docv - tray.map(docv.groupby(tray).median())
        sdm_c  = sdm  - tray.map(sdm.groupby(tray).median())
    else:
        docv_c, sdm_c = docv - docv.median(), sdm - sdm.median()

    m = docv_c.notna() & sdm_c.notna()
    if m.sum() < 5:
        return None
    dc, sc = docv_c[m], sdm_c[m]
    normal = ~(dc > docv_offset)
    out = {
        'n': int(m.sum()), 'n_docv_E': int((dc > docv_offset).sum()),
        'pearson': _pearson(sc, dc), 'spearman': _spearman(sc, dc),
    }
    if normal.sum() >= 5:
        out['pearson_normal']  = _pearson(sc[normal], dc[normal])
        out['spearman_normal'] = _spearman(sc[normal], dc[normal])
    else:
        out['pearson_normal'] = out['spearman_normal'] = float('nan')
    return out
