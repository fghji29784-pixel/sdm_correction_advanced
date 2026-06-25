# ─────────────────────────────────────────────────────────────
#  run_comparison.py  —  보정법 비교 실행 스크립트
#
#  사용법:
#    python run_comparison.py
#    → df_meta 를 직접 넣거나 아래 예시를 수정하여 사용.
# ─────────────────────────────────────────────────────────────

import sys
import pathlib

# sdm_logic 경로
_SDM_LOGIC = pathlib.Path.home() / "Desktop" / "sdm_logic"
if _SDM_LOGIC.exists() and str(_SDM_LOGIC) not in sys.path:
    sys.path.insert(0, str(_SDM_LOGIC))

from advanced_corrections import (
    compare_with_baseline,
    run_analysis_advanced,
    arrhenius_normalize,
    rwiring_tau_normalize,
)


# ─────────────────────────────────────────────────────────────
#  여기에 df_meta 를 불러오는 코드를 작성하세요.
#  (parser.py 로 파싱한 결과를 그대로 사용)
# ─────────────────────────────────────────────────────────────
# 예시:
#   from parser import parse_tray_folder
#   df_meta, df_ts = parse_tray_folder(r"C:\...\CFDD000001")

# 테스트 전용 더미 데이터 (실제 사용 시 아래 블록 제거)
def _make_dummy_df(n=144, seed=42):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    cell_no = np.arange(1, n + 1)
    layer   = np.clip((np.minimum(
        np.minimum(
            (cell_no - 1) // 12,
            11 - (cell_no - 1) // 12),
        np.minimum(
            (cell_no - 1) % 12,
            11 - (cell_no - 1) % 12)
    ) // 2) + 1, 1, 6)

    df = pd.DataFrame({
        "cell_no"    : cell_no,
        "tray_id"    : ["TRAY_A"] * 72 + ["TRAY_B"] * 72,
        "layer"      : layer,
        "v_init"     : rng.normal(3590, 5, n),
        "t_init"     : rng.normal(25.0, 0.3, n),
        "t_final"    : rng.normal(25.2, 0.5, n),
        "delta_t"    : rng.normal(0.2, 0.2, n),
        "rwiring"    : rng.normal(0.62, 0.05, n).clip(0.3, 1.5),
        "i_15min"    : rng.exponential(7e-8, n),   # A 단위
        "판정등급"   : ["A"] * 140 + ["E"] * 4,
    })
    # E급 셀은 전류 크게
    df.loc[df["판정등급"] == "E", "i_15min"] *= 30
    return df


if __name__ == "__main__":
    import pandas as pd

    print("더미 데이터로 실행 중 (실제 df_meta 로 교체하세요)...")
    df_meta = _make_dummy_df()

    # ── 단일 보정법 개별 확인 ──────────────────────────────
    print("\n[A-1] Arrhenius 보정 factor 예시 (25°C 기준)")
    import pandas as pd
    temps = pd.Series([23.0, 25.0, 27.0, 30.0, 35.0])
    i_dummy = pd.Series([1.0] * 5)
    factors = arrhenius_normalize(i_dummy, temps, t_ref_c=25.0)
    for t, f in zip(temps, factors):
        print(f"  {t:.0f}°C → factor = {f:.4f}")

    print("\n[C-1] Rwiring τ 보정 ratio 예시 (Rout=2Ω, 기준Rwiring=0.62Ω)")
    rw = pd.Series([0.40, 0.62, 0.80, 1.20, 1.80])
    i_dummy2 = pd.Series([1.0] * 5)
    ratios = rwiring_tau_normalize(i_dummy2, rw, rout=2.0, rser=0.05, rwiring_ref=0.62)
    for r, ratio in zip(rw, ratios):
        print(f"  Rwiring={r:.2f}Ω → ratio = {ratio:.4f}")

    # ── 전체 비교 ──────────────────────────────────────────
    print()
    result = compare_with_baseline(
        df_meta,
        n_minutes=15,
        option_baseline=3,
        rout=2.0,
        rser=0.05,
        eact_ev=0.94,
        t_ref_c=25.0,
    )
