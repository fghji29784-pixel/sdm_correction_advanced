# ─────────────────────────────────────────────────────────────
#  main.py  —  SDM 보정법 비교 진입점 (독립 실행)
#
#  사용 예:
#    # 1) 원본 측정 폴더 + 공정 데이터
#    python main.py --data "C:\측정\상위폴더" --process "공정데이터.xlsx"
#
#    # 2) 이미 파싱된 df_meta CSV (sdm_logic 결과 테이블)
#    python main.py --meta-csv "df_meta.csv"
#
#    # 3) 데이터 없이 데모 실행 (동작 확인)
#    python main.py --demo
#
#  옵션:
#    --n 15         판정 시간(분), 기본 15
#    --dep slope    종속변수 slope (기본 single)
#    --rout 2.0     출력 저항 Ω (미지정 시 KSS에서 자동 감지, 실패 시 2.0)
#    --out 결과.csv 보정값/z-score CSV 저장 경로
#    --plot         비교 막대그래프 PNG 저장
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from constants import DEFAULT_ROUT
from data_loader import load_from_raw, load_from_meta_csv, make_demo_meta
from compare import compare_all, print_table, run_config, CONFIGS


def _build_meta(args) -> tuple[pd.DataFrame, float | None]:
    if args.demo:
        print('▶ 데모 데이터 생성 (4 트레이, 실측 특성 모사)')
        return make_demo_meta(), None
    if args.meta_csv:
        print(f'▶ df_meta 로드: {args.meta_csv}')
        return load_from_meta_csv(args.meta_csv), None
    if args.data:
        print(f'▶ 원본 폴더 파싱: {args.data}')
        return load_from_raw(args.data, process_filepath=args.process)
    raise SystemExit('입력이 필요합니다: --data / --meta-csv / --demo 중 하나')


def _save_results(table: pd.DataFrame, out_path: str) -> None:
    """가장 강한 구성([4])의 셀별 보정값/z-score 저장."""
    results = table.attrs.get('results', {})
    best_name = CONFIGS[-1][0]
    res = results.get(best_name)
    if res is None:
        print('[경고] 저장할 결과가 없습니다.')
        return
    df = res['df_valid'].copy()
    df['corrected'] = res['corrected']
    df['z_score']   = res['z_scores']
    keep = [c for c in ['tray_id', 'cell_no', 'layer', 'v_init', 't_final',
                        'rwiring', '판정등급', 'corrected', 'z_score'] if c in df.columns]
    out = df[keep]
    if out_path.lower().endswith(('.xlsx', '.xls')):
        out.to_excel(out_path, index=False)
    else:
        out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'▶ 결과 저장 ({best_name}): {out_path}  ({len(out)}행)')


def _plot(table: pd.DataFrame, n_minutes: int, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[건너뜀] matplotlib 미설치 → 그래프 생략')
        return
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = table['구성'].tolist()
    ax[0].bar(range(len(names)), table['d_prime'], color='#4C72B0')
    ax[0].set_title(f'd_prime ({n_minutes}min)')
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax[1].bar(range(len(names)), table['dOCV_r_양품만'], color='#C44E52')
    ax[1].set_title('dOCV r (normal cells only)')
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f'▶ 그래프 저장: {path}')


def main():
    ap = argparse.ArgumentParser(description='SDM 보정법 비교 (독립 실행)')
    ap.add_argument('--data',     help='원본 측정 상위 폴더')
    ap.add_argument('--process',  help='공정 데이터 파일 (CSV/Excel)')
    ap.add_argument('--meta-csv', dest='meta_csv', help='파싱된 df_meta CSV/Excel')
    ap.add_argument('--demo',     action='store_true', help='데모 데이터로 실행')
    ap.add_argument('--n',        type=int, default=15, help='판정 시간(분)')
    ap.add_argument('--dep',      default='single', choices=['single', 'slope'])
    ap.add_argument('--rout',     type=float, default=None, help='출력 저항 Ω')
    ap.add_argument('--out',      help='결과 CSV/Excel 저장 경로')
    ap.add_argument('--plot',     help='비교 그래프 PNG 저장 경로')
    args = ap.parse_args()

    df_meta, rout_detected = _build_meta(args)
    rout = args.rout if args.rout is not None else (rout_detected or DEFAULT_ROUT)
    print(f'▶ 셀 {len(df_meta)}개 / 트레이 {df_meta["tray_id"].nunique() if "tray_id" in df_meta else 1}개'
          f' / Rout={rout}Ω '
          f'({"지정" if args.rout else ("KSS감지" if rout_detected else "기본값")})')

    table = compare_all(df_meta, n_minutes=args.n, dep_type=args.dep, rout=rout)
    print()
    print_table(table, args.n)

    if args.out:
        _save_results(table, args.out)
    if args.plot:
        _plot(table, args.n, args.plot)


if __name__ == '__main__':
    main()
