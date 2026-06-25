# ─────────────────────────────────────────────────────────────
#  constants.py  —  전역 상수 (자체 포함, sdm_logic 불필요)
# ─────────────────────────────────────────────────────────────

# 설비번호 → 셀 번호 오프셋 (셀 1~144)
DEVICE_OFFSET = {5: 0, 4: 16, 3: 48, 2: 80, 1: 112}
DEVICE_CELL_RANGE = {
    5: range(1, 17),
    4: range(17, 49),
    3: range(49, 81),
    2: range(81, 113),
    1: range(113, 145),
}

TOTAL_CELLS = 144
TRAY_ROWS   = 12
TRAY_COLS   = 12
NUM_LAYERS  = 6   # 가장자리(1) → 중앙(6)

# ── 공정 데이터 컬럼명 ──────────────────────────────────────────
PROCESS_COL_TRAY_ID = 'TRAY ID_OCV #01'
PROCESS_COL_CELL_NO = 'CELL NO_OCV #01'

PROCESS_COL_OCV = {
    'OCV1'        : 'OCV_OCV #01',
    'OCV2'        : 'OCV_OCV #02',
    'OCV3'        : 'OCV_OCV #03',
    'OCV4'        : 'OCV_OCV #04',
    'OCV7'        : 'OCV_OCV #07',
    'CHARGE_END_V': 'End Voltage_Charge #01',
}

# 검증용 dOCV (3일 기준)
PROCESS_COL_DOCV    = 'Delta OCV_Delta OCV #07'
PROCESS_COL_GRADE   = '판정등급'
PROCESS_COL_CELL_ID = 'Cell ID'
PROCESS_COL_LOT_ID  = 'Lot ID'

# ── 보정 기준 조건 ─────────────────────────────────────────────
REF_V_INIT  = 3590.0   # mV
REF_T_FINAL = 25.0     # °C
REF_DELTA_T = 0.0      # °C

# ── 물리 보정 기본 파라미터 ────────────────────────────────────
EACT_EV      = 0.94    # 활성화에너지 (eV), R-Smith 2023
KB_EV        = 8.617333e-5   # 볼츠만 상수 (eV/K)
DEFAULT_ROUT = 2.0     # 출력 저항 (Ω) — 실제 KSS TestSetupGlobal 값으로 교체 권장
DEFAULT_RSER = 0.05    # 내부 직렬 저항 추정값 (Ω)

# dOCV 규칙 오프셋 (mV) — 트레이 median + offset 초과 → E
DOCV_OFFSET = 0.8


def get_layer(cell_no: int) -> int:
    """셀 번호(1~144) → 트레이 레이어(1=가장자리, 6=중앙)"""
    row = (cell_no - 1) // TRAY_COLS + 1
    col = (cell_no - 1) % TRAY_COLS + 1
    return int(min(row, TRAY_ROWS + 1 - row, col, TRAY_COLS + 1 - col))


def cell_to_label(cell_no: int) -> str:
    """셀 번호(1~144) → 위치 레이블 (A01~L12)"""
    row_letter = chr(ord('A') + (cell_no - 1) // 12)
    col_number = (cell_no - 1) % 12 + 1
    return f'{row_letter}{col_number:02d}'
