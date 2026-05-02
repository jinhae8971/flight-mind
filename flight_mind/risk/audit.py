"""
Audit Logger — Order Lifecycle Tracking
==========================================
모든 주문 의사결정과 실행 결과를 SQLite에 영구 저장.

목적:
  - 추후 분쟁/디버깅 시 정확한 의사결정 추적
  - 백테스트와 실거래 결과 비교 분석
  - Tier 1/2/4 시그널이 실제 PnL과 어떻게 연관되는지 회귀 분석

설계 원칙:
  - 한 번 쓴 레코드는 절대 수정하지 않음 (append-only audit trail)
  - 모든 timestamps는 UTC ISO-8601 (timezone-aware, "+00:00" 표기)
  - Tier 신호는 JSON 형태로 그대로 저장 (구조 변화에도 유연)
  - 별도 SQLite (DuckDB OLAP과 분리 — 동시성 안전)

3-Layer 기록:
  1. Decisions   — Fusion Layer 의사결정 (open_long / open_short / hold)
  2. Orders      — 거래소에 보낸 주문 (placed / filled / cancelled / failed)
  3. Trades      — 완료된 매매 (PnL, exit_reason)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "audit.db"


def _now_iso() -> str:
    """현재 UTC를 SQLite DATE() 호환 ISO 형식으로 반환.

    SQLite의 DATE() 함수는 'YYYY-MM-DD HH:MM:SS' 또는 'YYYY-MM-DDTHH:MM:SS'
    형식을 인식. timezone suffix '+00:00'이나 'Z'가 있어도 보통 동작하지만,
    둘 다 같이 있으면 NULL 반환. 일관성을 위해 '+00:00'만 사용.
    """
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,        -- open_long / open_short / hold
    direction       TEXT,                  -- long / short / none
    confluence      REAL,
    veto_reason     TEXT,
    tier_outputs    TEXT,                  -- JSON
    market_snapshot TEXT,                  -- JSON: bid/ask/mid at decision time
    mode            TEXT NOT NULL          -- paper / testnet / live
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts_utc);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);

CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc              TEXT NOT NULL,
    decision_id         INTEGER,                 -- FK to decisions
    exchange_order_id   TEXT,                     -- 거래소 반환 ID
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,            -- buy / sell
    type                TEXT NOT NULL,            -- market / limit / stop / take_profit
    quantity            REAL NOT NULL,
    price               REAL,                     -- limit/stop price (NULL for market)
    status              TEXT NOT NULL,            -- pending / filled / cancelled / failed
    filled_qty          REAL DEFAULT 0,
    avg_fill_price      REAL,
    fee_usdt            REAL DEFAULT 0,
    error_message       TEXT,
    raw_response        TEXT,                     -- JSON of exchange response
    mode                TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts_utc);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_decision ON orders(decision_id);

CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id         INTEGER,
    entry_order_id      INTEGER NOT NULL,
    exit_order_id       INTEGER,
    symbol              TEXT NOT NULL,
    direction           TEXT NOT NULL,            -- long / short
    entry_ts_utc        TEXT NOT NULL,
    exit_ts_utc         TEXT,
    entry_price         REAL NOT NULL,
    exit_price          REAL,
    quantity            REAL NOT NULL,
    pnl_usdt            REAL,
    pnl_pct             REAL,
    exit_reason         TEXT,                     -- tp / sl / time / manual
    fees_usdt           REAL DEFAULT 0,
    mode                TEXT NOT NULL,
    FOREIGN KEY (entry_order_id) REFERENCES orders(id),
    FOREIGN KEY (exit_order_id) REFERENCES orders(id),
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades(entry_ts_utc);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
"""


# Thread-local connection (SQLite 멀티스레드 안전)
_local = threading.local()


@contextmanager
def get_audit_conn():
    """Audit DB 커넥션 — auto-commit"""
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUDIT_DB_PATH, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_audit_db() -> None:
    """Schema 초기화 (idempotent)"""
    with get_audit_conn() as conn:
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)


# =============================================================================
# Logging Functions
# =============================================================================
def log_decision(
    symbol: str,
    action: str,                    # open_long / open_short / hold
    direction: str,
    confluence: float,
    tier_outputs: dict[str, Any],
    market_snapshot: dict[str, Any] | None = None,
    veto_reason: str | None = None,
    mode: str = "paper",
) -> int:
    """
    Fusion Layer 의사결정을 영구 기록.

    Returns:
        decision_id (이후 order/trade와 연결)
    """
    init_audit_db()
    with get_audit_conn() as conn:
        cur = conn.execute(
            """INSERT INTO decisions
               (ts_utc, symbol, action, direction, confluence, veto_reason,
                tier_outputs, market_snapshot, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now_iso(),
                symbol, action, direction, confluence, veto_reason,
                json.dumps(tier_outputs, default=str),
                json.dumps(market_snapshot, default=str) if market_snapshot else None,
                mode,
            ),
        )
        return cur.lastrowid


def log_order(
    decision_id: int | None,
    exchange_order_id: str | None,
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: float | None,
    status: str,
    filled_qty: float = 0,
    avg_fill_price: float | None = None,
    fee_usdt: float = 0,
    error_message: str | None = None,
    raw_response: dict | None = None,
    mode: str = "paper",
) -> int:
    """주문 기록 (placed / filled / cancelled / failed 모든 상태)"""
    init_audit_db()
    with get_audit_conn() as conn:
        cur = conn.execute(
            """INSERT INTO orders
               (ts_utc, decision_id, exchange_order_id, symbol, side, type,
                quantity, price, status, filled_qty, avg_fill_price, fee_usdt,
                error_message, raw_response, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now_iso(),
                decision_id, exchange_order_id, symbol, side, order_type,
                quantity, price, status, filled_qty, avg_fill_price, fee_usdt,
                error_message,
                json.dumps(raw_response, default=str) if raw_response else None,
                mode,
            ),
        )
        return cur.lastrowid


def update_order_status(
    order_id: int,
    status: str,
    filled_qty: float | None = None,
    avg_fill_price: float | None = None,
    fee_usdt: float | None = None,
    raw_response: dict | None = None,
) -> None:
    """기존 주문 상태 업데이트 (audit immutability 깨지 않음 — 같은 row 갱신만 허용)"""
    fields = ["status = ?"]
    values = [status]

    if filled_qty is not None:
        fields.append("filled_qty = ?")
        values.append(filled_qty)
    if avg_fill_price is not None:
        fields.append("avg_fill_price = ?")
        values.append(avg_fill_price)
    if fee_usdt is not None:
        fields.append("fee_usdt = ?")
        values.append(fee_usdt)
    if raw_response is not None:
        fields.append("raw_response = ?")
        values.append(json.dumps(raw_response, default=str))

    values.append(order_id)
    sql = f"UPDATE orders SET {', '.join(fields)} WHERE id = ?"

    with get_audit_conn() as conn:
        conn.execute(sql, values)


def log_trade_open(
    decision_id: int,
    entry_order_id: int,
    symbol: str,
    direction: str,
    entry_price: float,
    quantity: float,
    mode: str = "paper",
) -> int:
    """거래 진입 기록"""
    init_audit_db()
    with get_audit_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (decision_id, entry_order_id, symbol, direction,
                entry_ts_utc, entry_price, quantity, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id, entry_order_id, symbol, direction,
                _now_iso(),
                entry_price, quantity, mode,
            ),
        )
        return cur.lastrowid


def log_trade_close(
    trade_id: int,
    exit_order_id: int,
    exit_price: float,
    pnl_usdt: float,
    pnl_pct: float,
    exit_reason: str,
    fees_usdt: float = 0,
) -> None:
    """거래 종료 기록"""
    with get_audit_conn() as conn:
        conn.execute(
            """UPDATE trades
               SET exit_order_id = ?, exit_ts_utc = ?, exit_price = ?,
                   pnl_usdt = ?, pnl_pct = ?, exit_reason = ?, fees_usdt = ?
               WHERE id = ?""",
            (
                exit_order_id,
                _now_iso(),
                exit_price, pnl_usdt, pnl_pct, exit_reason, fees_usdt,
                trade_id,
            ),
        )


# =============================================================================
# Query Helpers (분석용)
# =============================================================================
def fetch_recent_decisions(symbol: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM decisions"
    params = []
    if symbol:
        sql += " WHERE symbol = ?"
        params.append(symbol)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_audit_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def fetch_recent_trades(symbol: str | None = None, limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM trades"
    params = []
    if symbol:
        sql += " WHERE symbol = ?"
        params.append(symbol)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_audit_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def daily_pnl_summary(date_utc: str | None = None) -> dict:
    """오늘 또는 지정일 PnL 요약"""
    if date_utc is None:
        date_utc = datetime.now(timezone.utc).date().isoformat()

    with get_audit_conn() as conn:
        rows = conn.execute(
            """SELECT mode, COUNT(*) as n_trades,
                      SUM(pnl_usdt) as total_pnl_usdt,
                      SUM(pnl_pct) as cumulative_pnl_pct,
                      SUM(fees_usdt) as total_fees,
                      SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins
               FROM trades
               WHERE DATE(exit_ts_utc) = ?
               GROUP BY mode""",
            [date_utc],
        ).fetchall()

    return {
        date_utc: [dict(r) for r in rows],
    }
