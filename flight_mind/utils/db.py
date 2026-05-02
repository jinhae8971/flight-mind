"""
DuckDB Feature Store — Single Source of Truth for All Tiers
============================================================
모든 Tier(1~4)가 공유하는 데이터 저장소.
Parquet 콜드 스토리지에서 데이터를 읽어와 인덱스/뷰로 빠른 조회 제공.

Schema:
  - ohlcv          : raw 5분봉 (BTC, ETH)
  - features_5m    : RSI, MA, ATR 등 사전 계산된 지표
  - features_1h    : 1시간봉 리샘플링 + 지표
  - features_1d    : 일봉 + 시장 국면 지표
  - regimes        : Tier 4용 regime 라벨
  - tier1_signals  : Tier 1 룰 출력 캐시
  - tier_outputs   : 모든 Tier의 출력 시계열 (백테스트 + 라이브 공통)
  - decisions      : Fusion Layer의 최종 의사결정 로그
  - trades         : 실제 진입/청산 기록
  - pnl_daily      : 일일 손익 집계 (Kill-Switch가 참조)

Why DuckDB?
  - Embedded (별도 서버 없음) — 영길님의 GitHub Actions 패턴과 호환
  - Columnar — 시계열 집계 쿼리 매우 빠름
  - SQL 표준 — 영길님께서 직접 분석 쿼리 작성 가능
  - Parquet native — 콜드 스토리지와 zero-copy 연동
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from flight_mind.config import DATA_DIR


DUCKDB_PATH = DATA_DIR / "flight_mind.duckdb"


# =============================================================================
# Schema Definition
# =============================================================================
SCHEMA_DDL = """
-- 1) Raw OHLCV — 5분봉 (BTC, ETH)
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      VARCHAR NOT NULL,
    interval    VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    volume      DOUBLE NOT NULL,
    trades      INTEGER,
    PRIMARY KEY (symbol, interval, open_time)
);

-- 2) 사전 계산된 5분봉 지표 (Tier 1 입력)
CREATE TABLE IF NOT EXISTS features_5m (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    close       DOUBLE,
    rsi_14      DOUBLE,
    ma_7        DOUBLE,
    ma_30       DOUBLE,
    ma_120      DOUBLE,
    atr_14      DOUBLE,
    volume_ma20 DOUBLE,
    volume_zscore DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 3) 1시간봉 + 지표 (멀티 타임프레임 분석용)
CREATE TABLE IF NOT EXISTS features_1h (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    close       DOUBLE,
    rsi_14      DOUBLE,
    ma_50       DOUBLE,
    ma_200      DOUBLE,
    atr_14      DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 4) 일봉 + 시장 국면 지표 (Tier 4 입력)
CREATE TABLE IF NOT EXISTS features_1d (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    close       DOUBLE,
    return_1d   DOUBLE,
    volatility_30d DOUBLE,
    rsi_14      DOUBLE,
    ma_50       DOUBLE,
    ma_200      DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 5) Tier 4 regime 라벨 (학습용)
CREATE TABLE IF NOT EXISTS regimes (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    regime      VARCHAR NOT NULL,    -- Bull-Trending, Bear-Trending, etc.
    confidence  DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 6) Tier 1 시그널 캐시
CREATE TABLE IF NOT EXISTS tier1_signals (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    score       DOUBLE NOT NULL,
    direction   VARCHAR NOT NULL,
    active_rules INTEGER,
    metadata    JSON,
    PRIMARY KEY (symbol, open_time)
);

-- 7) 모든 Tier 출력 시계열
CREATE TABLE IF NOT EXISTS tier_outputs (
    symbol      VARCHAR NOT NULL,
    open_time   TIMESTAMP NOT NULL,
    tier        VARCHAR NOT NULL,    -- T1, T2, T3, T4
    score       DOUBLE NOT NULL,
    direction   VARCHAR NOT NULL,
    metadata    JSON,
    PRIMARY KEY (symbol, open_time, tier)
);

-- 8) Fusion 의사결정 로그
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY,
    symbol      VARCHAR NOT NULL,
    decided_at  TIMESTAMP NOT NULL,
    action      VARCHAR NOT NULL,    -- open_long, open_short, hold, close_position
    confluence  DOUBLE NOT NULL,
    direction   VARCHAR NOT NULL,
    position_size_usdt DOUBLE,
    leverage    INTEGER,
    veto_reason VARCHAR,
    tier_outputs JSON
);
CREATE SEQUENCE IF NOT EXISTS decisions_seq START 1;

-- 9) 실제 진입/청산 거래 기록
CREATE TABLE IF NOT EXISTS trades (
    trade_id    INTEGER PRIMARY KEY,
    decision_id INTEGER,
    symbol      VARCHAR NOT NULL,
    side        VARCHAR NOT NULL,    -- long, short
    entry_time  TIMESTAMP NOT NULL,
    entry_price DOUBLE NOT NULL,
    size_usdt   DOUBLE NOT NULL,
    leverage    INTEGER,
    exit_time   TIMESTAMP,
    exit_price  DOUBLE,
    exit_reason VARCHAR,             -- tp, sl, time, manual, kill_switch
    pnl_usdt    DOUBLE,
    pnl_pct     DOUBLE,
    fees_usdt   DOUBLE
);
CREATE SEQUENCE IF NOT EXISTS trades_seq START 1;

-- 10) 일일 PnL — Kill-Switch가 참조
CREATE TABLE IF NOT EXISTS pnl_daily (
    date        DATE PRIMARY KEY,
    trades_count INTEGER NOT NULL DEFAULT 0,
    realized_pnl_usdt DOUBLE NOT NULL DEFAULT 0,
    realized_pnl_pct  DOUBLE NOT NULL DEFAULT 0,
    fees_usdt   DOUBLE NOT NULL DEFAULT 0,
    end_balance_usdt DOUBLE,
    max_dd_pct  DOUBLE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_features5m_symbol_time ON features_5m(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON decisions(decided_at);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
"""


# =============================================================================
# Connection Management
# =============================================================================
@contextmanager
def get_conn(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """DuckDB 연결을 컨텍스트 매니저로 제공."""
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """스키마 생성 — 멱등(idempotent)"""
    with get_conn() as conn:
        # DuckDB는 multi-statement 지원
        for statement in SCHEMA_DDL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
    print(f"[DuckDB] Schema initialized at {DUCKDB_PATH}")


def db_stats() -> dict[str, int]:
    """각 테이블 row 수 — 진단용"""
    tables = [
        "ohlcv", "features_5m", "features_1h", "features_1d",
        "regimes", "tier1_signals", "tier_outputs",
        "decisions", "trades", "pnl_daily",
    ]
    stats = {}
    with get_conn(read_only=True) as conn:
        for t in tables:
            try:
                stats[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                stats[t] = -1
    return stats


# =============================================================================
# Bulk Load Helpers
# =============================================================================
def load_parquet_to_ohlcv(
    parquet_path: Path | str,
    symbol: str,
    interval: str = "5m",
) -> int:
    """Parquet 파일을 ohlcv 테이블에 적재 (idempotent: PK 충돌 무시)."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    sql = f"""
    INSERT OR IGNORE INTO ohlcv (symbol, interval, open_time, open, high, low, close, volume, trades)
    SELECT
        '{symbol}' AS symbol,
        '{interval}' AS interval,
        open_time,
        CAST(open AS DOUBLE),
        CAST(high AS DOUBLE),
        CAST(low AS DOUBLE),
        CAST(close AS DOUBLE),
        CAST(volume AS DOUBLE),
        CAST(trades AS INTEGER)
    FROM read_parquet('{parquet_path}')
    """
    with get_conn() as conn:
        before = conn.execute(
            f"SELECT COUNT(*) FROM ohlcv WHERE symbol='{symbol}' AND interval='{interval}'"
        ).fetchone()[0]
        conn.execute(sql)
        after = conn.execute(
            f"SELECT COUNT(*) FROM ohlcv WHERE symbol='{symbol}' AND interval='{interval}'"
        ).fetchone()[0]
    return after - before


def fetch_ohlcv(
    symbol: str,
    interval: str = "5m",
    start: str | None = None,
    end: str | None = None,
):
    """Pandas DataFrame으로 OHLCV 조회"""
    where = [f"symbol = '{symbol}'", f"interval = '{interval}'"]
    if start:
        where.append(f"open_time >= '{start}'")
    if end:
        where.append(f"open_time <= '{end}'")
    where_clause = " AND ".join(where)

    sql = f"""
    SELECT open_time, open, high, low, close, volume, trades
    FROM ohlcv
    WHERE {where_clause}
    ORDER BY open_time
    """
    with get_conn(read_only=True) as conn:
        return conn.execute(sql).fetch_df().set_index("open_time")


if __name__ == "__main__":
    init_db()
    stats = db_stats()
    print("\nTable Statistics:")
    for table, count in stats.items():
        print(f"  {table:20s}: {count:>10,}")
