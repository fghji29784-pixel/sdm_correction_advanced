# ─────────────────────────────────────────────────────────────
#  data_loader.py  —  원본 측정 파일 파싱 (자체 포함)
#
#  입력 (둘 중 하나):
#    (1) 원본 측정 폴더 (하위 트레이 폴더 + KSS/TEMP 파일) + 공정 데이터 파일
#    (2) 이미 파싱된 df_meta CSV/Excel (sdm_logic 통합 결과 테이블)
#
#  출력: df_meta  (셀당 1행)
#
#  ※ sdm_logic/parser.py 의 검증된 파싱 로직을 자체 포함하여 독립 실행.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    DEVICE_OFFSET, TOTAL_CELLS, TRAY_ROWS, TRAY_COLS,
    PROCESS_COL_OCV, PROCESS_COL_DOCV,
    PROCESS_COL_GRADE, PROCESS_COL_CELL_ID, PROCESS_COL_LOT_ID,
    get_layer,
)


# ══════════════════════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════════════════════

def cell_no_from(device_no: int, channel_no: int) -> int:
    return DEVICE_OFFSET[device_no] + channel_no


def detect_device_no(filename: str) -> int | None:
    stem = Path(filename).stem
    for part in reversed(stem.split('_')):
        if re.fullmatch(r'[1-5]', part):
            return int(part)
    return None


def _split_line(line: str) -> list[str]:
    if '\t' in line:
        sep = '\t'
    elif ',' in line:
        sep = ','
    else:
        sep = '|'
    return [p.strip() for p in line.split(sep)]


def _read_text(filepath: str) -> str:
    for enc in ('utf-8', 'euc-kr', 'cp949'):
        try:
            return Path(filepath).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(filepath).read_text(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════
#  KSS 파일 파싱
# ══════════════════════════════════════════════════════════════

def _split_sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in content.splitlines():
        line = raw.strip()
        m = re.match(r'^\[?Cat[\s_]+(.+?)\]?$', line)
        if m:
            current = m.group(1).strip()
            sections[current] = []
        elif current is not None and line:
            sections[current].append(line)
    return sections


def _line_key_value(line: str) -> tuple[str, str] | None:
    if ':' in line:
        k, v = line.split(':', 1)
        return k.strip(), v.strip()
    parts = _split_line(line)
    if len(parts) >= 2 and parts[0]:
        return parts[0].strip(), parts[1].strip()
    return None


def _parse_app_info(lines: list[str]) -> dict:
    out = {}
    for line in lines:
        kv = _line_key_value(line)
        if kv is not None:
            out[kv[0]] = kv[1]
    return out


def _parse_global(lines: list[str]) -> dict:
    out = {'test_duration': 15, 'tint': 10, 'rout': None}
    for line in lines:
        kv = _line_key_value(line)
        if kv is None:
            continue
        k, v = kv
        if k == 'TestDuration':
            try: out['test_duration'] = int(float(v))
            except ValueError: pass
        elif k == 'Tint':
            try: out['tint'] = int(float(v))
            except ValueError: pass
        elif k.lower() == 'rout':
            try: out['rout'] = float(v)
            except ValueError: pass
    return out


def _parse_per_chan(lines: list[str]) -> dict[int, dict]:
    """채널별 {enabled, rwiring, rout} 파싱"""
    channels: dict[int, dict] = {}
    if not lines:
        return channels

    header_idx = None
    for i, line in enumerate(lines):
        if any('channel' in p.lower() for p in _split_line(line)):
            header_idx = i
            break
    if header_idx is None:
        return channels

    headers = _split_line(lines[header_idx])

    def idx(keyword):
        for i, h in enumerate(headers):
            if keyword.lower() in h.lower():
                return i
        return None

    i_ch = idx('Channel');  i_ch = 0 if i_ch is None else i_ch
    i_en = idx('IsChanEnabled') or idx('IsChanCheck') or idx('Enabled')
    i_en = 1 if i_en is None else i_en
    i_rw = idx('Rwiring')
    if i_rw is None:
        i_rw = idx('Rwire')
    i_ro = idx('Rout')

    for line in lines[header_idx + 1:]:
        parts = _split_line(line)
        if len(parts) < 2:
            continue
        try:
            ch      = int(parts[i_ch])
            enabled = parts[i_en].strip().upper() in ('TRUE', '1', 'Y', 'YES')
            rwiring = float(parts[i_rw]) if (i_rw is not None and i_rw < len(parts)) else 0.0
            rout    = float(parts[i_ro]) if (i_ro is not None and i_ro < len(parts)) else np.nan
            channels[ch] = {'enabled': enabled, 'rwiring': rwiring, 'rout': rout}
        except (ValueError, IndexError):
            continue
    return channels


def _parse_measurements(lines: list[str]) -> pd.DataFrame | None:
    if not lines:
        return None
    header_idx = None
    for i, line in enumerate(lines):
        lowered = [p.lower() for p in _split_line(line)]
        if lowered and lowered[0] == 'time' and any(re.fullmatch(r'[iv]\d+', p) for p in lowered[1:]):
            header_idx = i
            break
    if header_idx is None:
        return None

    headers = _split_line(lines[header_idx])
    rows = [_split_line(line) for line in lines[header_idx + 1:] if line]
    if not rows:
        return None

    max_len = max(len(r) for r in rows)
    headers = (headers + [''] * max_len)[:max_len]
    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_kss_file(filepath: str) -> dict:
    content  = _read_text(filepath)
    sections = _split_sections(content)
    app    = _parse_app_info(sections.get('AppInfo', []))
    glb    = _parse_global(sections.get('TestSetupGlobal', []))
    chans  = _parse_per_chan(sections.get('TestSetupPerChan', []))
    meas   = _parse_measurements(sections.get('Measurements', []))
    return {
        'serial_num'   : app.get('SerialNum'),
        'test_duration': glb['test_duration'],
        'tint'         : glb['tint'],
        'rout_global'  : glb['rout'],
        'channels'     : chans,
        'measurements' : meas,
    }


# ══════════════════════════════════════════════════════════════
#  TEMP 파일 파싱
# ══════════════════════════════════════════════════════════════

def parse_temp_data(filepath: str) -> pd.DataFrame:
    fp = Path(filepath)
    try:
        if fp.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(fp, header=0)
        else:
            df = pd.read_csv(fp, sep=None, engine='python', header=0)
        n_cols = len(df.columns)
        df.columns = ['t_sec'] + [f'T_{i}' for i in range(1, n_cols)]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f'[TEMP 파싱 오류] {fp.name}: {e}')
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  트레이 폴더 파싱
# ══════════════════════════════════════════════════════════════

def parse_tray_folder(folder_path: str) -> tuple[pd.DataFrame, float | None]:
    """
    트레이 폴더 1개 파싱 → (df_meta, rout_detected)
    rout_detected: KSS에서 읽어낸 Rout (없으면 None)
    """
    folder = Path(folder_path)
    _m = re.search(r'CFDD\S+', folder.name, re.IGNORECASE)
    tray_id = (_m.group(0).upper() if _m
               else re.sub(r'^\d+\.\s*', '', folder.name).strip() or folder.name)

    kss_files = list(folder.glob('*.kss')) + list(folder.glob('*.KSS'))
    device_kss: dict[int, str] = {}
    for f in kss_files:
        dev = detect_device_no(f.name)
        if dev is not None:
            device_kss[dev] = str(f)

    temp_files = list(folder.glob('*TEMP_DATA*')) or list(folder.glob('*TEMP*'))
    temp_df = parse_temp_data(str(temp_files[0])) if temp_files else pd.DataFrame()

    meta_rows: list[dict] = []
    rout_detected: float | None = None
    N_ALL = range(5, 31)

    for device_no, kss_path in sorted(device_kss.items()):
        kss  = parse_kss_file(kss_path)
        meas = kss['measurements']
        if meas is None or meas.empty:
            continue

        if rout_detected is None and kss.get('rout_global') is not None:
            rout_detected = kss['rout_global']

        t_sec_arr = meas[meas.columns[0]].values
        active    = [ch for ch, info in kss['channels'].items() if info['enabled']]

        for ch in active:
            cell_no = cell_no_from(device_no, ch)
            if not (1 <= cell_no <= TOTAL_CELLS):
                continue
            i_col, v_col = f'I{ch}', f'V{ch}'
            if i_col not in meas.columns or v_col not in meas.columns:
                continue

            cur_arr = meas[i_col].values.astype(float)
            vol_arr = meas[v_col].values.astype(float)
            v_init  = vol_arr[0] * 1000.0 if len(vol_arr) > 0 else np.nan

            t_key = f'T_{cell_no}'
            t_init = t_final = np.nan
            if not temp_df.empty and t_key in temp_df.columns:
                temps  = temp_df[t_key].values
                t_init  = float(temps[0])  if len(temps) > 0 else np.nan
                t_final = float(temps[-1]) if len(temps) > 0 else np.nan

            ch_rout = kss['channels'][ch].get('rout', np.nan)
            if rout_detected is None and not np.isnan(ch_rout):
                rout_detected = ch_rout

            row: dict = {
                'tray_id'   : tray_id,
                'cell_no'   : cell_no,
                'device_no' : device_no,
                'channel_no': ch,
                'rwiring'   : kss['channels'][ch]['rwiring'],
                'rout'      : ch_rout,
                'v_init'    : v_init,
                't_init'    : t_init,
                't_final'   : t_final,
                'delta_t'   : (t_final - t_init)
                              if not (np.isnan(t_init) or np.isnan(t_final)) else np.nan,
                'layer'     : get_layer(cell_no),
            }
            for n in N_ALL:
                n_sec = n * 60
                idx_n = min(int(np.searchsorted(t_sec_arr, n_sec)), len(t_sec_arr) - 1)
                row[f'i_{n}min']    = cur_arr[idx_n] if idx_n < len(cur_arr) else np.nan
                row[f'slope_0_{n}'] = ((cur_arr[idx_n] - cur_arr[0]) / n_sec) if n_sec > 0 else np.nan
            meta_rows.append(row)

    return pd.DataFrame(meta_rows), rout_detected


# ══════════════════════════════════════════════════════════════
#  공정 데이터 병합
# ══════════════════════════════════════════════════════════════

def parse_process_data(filepath: str) -> pd.DataFrame:
    fp = Path(filepath)
    try:
        if fp.suffix.lower() in ('.xlsx', '.xls'):
            return pd.read_excel(fp, header=0)
        return pd.read_csv(fp, sep=None, engine='python', header=0)
    except Exception as e:
        print(f'[공정 데이터 파싱 오류] {fp.name}: {e}')
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, keyword: str) -> str | None:
    for col in df.columns:
        if keyword in str(col):
            return col
    return None


def merge_process_data(df_meta: pd.DataFrame, process_df: pd.DataFrame) -> pd.DataFrame:
    if process_df.empty or df_meta.empty:
        return df_meta
    tray_col = _find_col(process_df, 'TRAY ID')
    cell_col = _find_col(process_df, 'CELL NO')
    if tray_col is None or cell_col is None:
        print('[경고] 공정 데이터에서 TRAY ID / CELL NO 컬럼을 찾을 수 없습니다.')
        return df_meta

    want = [tray_col, cell_col]
    for col in PROCESS_COL_OCV.values():
        if col in process_df.columns:
            want.append(col)
    for col in [PROCESS_COL_DOCV, PROCESS_COL_GRADE, PROCESS_COL_CELL_ID, PROCESS_COL_LOT_ID]:
        if col in process_df.columns:
            want.append(col)

    proc = process_df[want].copy().rename(columns={tray_col: '_tk', cell_col: '_ck'})
    proc['_tk'] = proc['_tk'].astype(str).str.strip().str.upper()
    proc['_ck'] = pd.to_numeric(proc['_ck'], errors='coerce')

    meta = df_meta.copy()
    meta['_tk'] = meta['tray_id'].astype(str).str.strip().str.upper()
    meta['_ck'] = pd.to_numeric(meta['cell_no'], errors='coerce')

    return meta.merge(proc, on=['_tk', '_ck'], how='left').drop(columns=['_tk', '_ck'])


# ══════════════════════════════════════════════════════════════
#  통합 로더 (진입점에서 호출)
# ══════════════════════════════════════════════════════════════

def load_from_raw(data_root: str,
                  process_filepath: str | None = None) -> tuple[pd.DataFrame, float | None]:
    """
    상위 폴더 아래 트레이 폴더들을 자동 감지하여 일괄 파싱.
    반환: (df_meta, rout_detected)
    """
    root = Path(data_root)
    # 하위에 KSS가 직접 있으면 단일 트레이, 아니면 하위 폴더들이 트레이
    has_kss_here = any(root.glob('*.kss')) or any(root.glob('*.KSS'))
    tray_folders = [root] if has_kss_here else [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (any(d.glob('*.kss')) or any(d.glob('*.KSS')))
    ]
    if not tray_folders:
        raise FileNotFoundError(f'KSS 파일이 있는 트레이 폴더를 찾을 수 없습니다: {data_root}')

    metas, rout = [], None
    for tf in tray_folders:
        dm, r = parse_tray_folder(str(tf))
        if not dm.empty:
            metas.append(dm)
        if rout is None and r is not None:
            rout = r

    if not metas:
        raise ValueError('파싱된 셀 데이터가 없습니다.')
    df_meta = pd.concat(metas, ignore_index=True)

    if process_filepath:
        df_meta = merge_process_data(df_meta, parse_process_data(process_filepath))

    return df_meta, rout


def load_from_meta_csv(meta_path: str) -> pd.DataFrame:
    """이미 파싱된 df_meta CSV/Excel 로드 (sdm_logic 결과 테이블)."""
    fp = Path(meta_path)
    if fp.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(fp, header=0)
    else:
        df = pd.read_csv(fp, sep=None, engine='python', header=0)
    if 'cell_no' not in df.columns:
        raise ValueError("df_meta CSV에 'cell_no' 컬럼이 필요합니다.")
    if 'layer' not in df.columns and 'cell_no' in df.columns:
        df['layer'] = df['cell_no'].apply(get_layer)
    return df


# ══════════════════════════════════════════════════════════════
#  데모 데이터 생성기 (데이터 없이 동작 확인용)
# ══════════════════════════════════════════════════════════════

def make_demo_meta(n_trays: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    실측 특성을 모사한 데모 df_meta 생성:
      - 트레이 간 레벨차 (bimodal)
      - 중앙 고온 / 가장자리 저온 공간 구배
      - 채널별 Rwiring 편차
      - 트레이당 평균 0~1개 미세 불량(E)
    """
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_trays):
        cell_no = np.arange(1, TOTAL_CELLS + 1)
        row = (cell_no - 1) // TRAY_COLS
        col = (cell_no - 1) %  TRAY_COLS
        # 중앙 고온 구배 (가장자리 24.8℃ → 중앙 25.6℃)
        dist_c  = np.sqrt((row - 5.5) ** 2 + (col - 5.5) ** 2)
        t_base  = 25.6 - 0.10 * dist_c + rng.normal(0, 0.05, TOTAL_CELLS)
        tray_level = 0.3 * t                      # 트레이 간 레벨차
        rwiring = np.clip(rng.normal(0.62, 0.06, TOTAL_CELLS), 0.35, 1.6)

        # 양품 ISD: 온도 의존(Arrhenius) + τ(Rwiring) 영향 반영한 15분 측정 전류
        from constants import EACT_EV, KB_EV, DEFAULT_ROUT, DEFAULT_RSER
        isd_true = rng.normal(5e-6, 1e-6, TOTAL_CELLS).clip(2e-6, None)  # A
        T_K = t_base + tray_level + 273.15
        arr = np.exp((EACT_EV / KB_EV) * (1 / (25 + 273.15) - 1 / T_K))
        tau = (DEFAULT_ROUT + DEFAULT_RSER + rwiring)
        i15 = isd_true * arr * (15 * 60) / (tau * 1e5)   # 임의 스케일

        grade = np.array(['A'] * TOTAL_CELLS, dtype=object)
        # 트레이당 0~1개 미세불량
        n_bad = rng.integers(0, 2)
        if n_bad:
            bad_idx = rng.choice(TOTAL_CELLS, n_bad, replace=False)
            i15[bad_idx] *= rng.uniform(4, 8, n_bad)   # 미세불량 (z≈2~5 수준)
            grade[bad_idx] = 'E'

        df = pd.DataFrame({
            'tray_id' : f'CFDD{t+1:06d}',
            'cell_no' : cell_no,
            'rwiring' : rwiring,
            'rout'    : DEFAULT_ROUT,
            'v_init'  : rng.normal(3590, 4, TOTAL_CELLS),
            't_init'  : t_base + tray_level - 0.1,
            't_final' : t_base + tray_level,
            'delta_t' : rng.normal(0.1, 0.05, TOTAL_CELLS),
            'layer'   : [get_layer(c) for c in cell_no],
            'i_15min' : i15,
            'OCV_OCV #01'           : rng.normal(3.590, 0.004, TOTAL_CELLS),
            'OCV_OCV #03'           : rng.normal(3.588, 0.004, TOTAL_CELLS),
            PROCESS_COL_DOCV        : None,
            PROCESS_COL_GRADE       : grade,
        })
        # dOCV(mV) = 양품 기준선 + ISD 비례 + 불량 가산
        docv = rng.normal(1.8, 0.15, TOTAL_CELLS) + (i15 / i15.mean() - 1) * 0.3
        docv[grade == 'E'] += rng.uniform(1.0, 2.0, (grade == 'E').sum())
        df[PROCESS_COL_DOCV] = docv
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
