"""
Position Tracker
=================
오픈 포지션을 메모리 + DB로 이중 추적하며 실시간 unrealized PnL 계산.

설계 원칙:
  - Single source of truth: audit DB의 trades 테이블 (exit_ts_utc IS NULL = 오픈)
  - 메모리 캐시는 빠른 조회용, DB가 항상 최종 진실
  - 거래소 포지션과 정기 reconcile (drift 감지)
  - 모든 PnL 계산은 fee 차감 포함

핵심 책임:
  1. 오픈 포지션 조회 (어떤 심볼에 어느 방향 얼마)
  2. Unrealized PnL 실시간 (현재가 입력 시)
  3. 청산 시 trade record 완료
  4. 거래소 잔고와 reconcile (drift > 1% 시 경고)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from rich.console import Console

from flight_mind.risk.audit import get_audit_conn, log_trade_close


CONSOLE = Console()


@dataclass
class OpenPosition:
    """오픈 상태인 단일 포지션"""
    trade_id: int
    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    quantity: float
    entry_ts_utc: str
    fees_paid_usdt: float = 0.0       # 진입 수수료

    # 진단용 (DB에는 없음)
    decision_id: int | None = None
    entry_order_id: int | None = None

    # 실시간 갱신 필드
    current_price: float | None = None
    unrealized_pnl_usdt: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    def update_market_price(self, current_price: float, fee_rate: float = 0.0005) -> None:
        """현재가로 unrealized PnL 갱신"""
        self.current_price = current_price

        # 진입 수수료 + 청산 예상 수수료
        exit_fee = current_price * self.quantity * fee_rate
        total_fees = self.fees_paid_usdt + exit_fee

        if self.is_long:
            gross_pnl = (current_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - current_price) * self.quantity

        net_pnl = gross_pnl - total_fees

        self.unrealized_pnl_usdt = net_pnl
        self.unrealized_pnl_pct = (net_pnl / (self.entry_price * self.quantity)) * 100

    def hold_duration_minutes(self) -> float:
        """진입 이후 경과 시간 (분)"""
        try:
            entry_dt = datetime.fromisoformat(self.entry_ts_utc.rstrip("Z"))
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - entry_dt).total_seconds() / 60
        except Exception:
            return 0.0


class PositionTracker:
    """오픈 포지션 통합 관리"""

    def __init__(self):
        self._cache: dict[int, OpenPosition] = {}  # trade_id → position
        self._dirty: bool = True

    def refresh_from_db(self) -> int:
        """DB에서 오픈 포지션 다시 로드 (cold start 또는 reconcile 시)"""
        with get_audit_conn() as conn:
            rows = conn.execute(
                """SELECT id, decision_id, entry_order_id, symbol, direction,
                          entry_ts_utc, entry_price, quantity, fees_usdt
                   FROM trades
                   WHERE exit_ts_utc IS NULL
                   ORDER BY entry_ts_utc"""
            ).fetchall()

        self._cache.clear()
        for row in rows:
            self._cache[row["id"]] = OpenPosition(
                trade_id=row["id"],
                decision_id=row["decision_id"],
                entry_order_id=row["entry_order_id"],
                symbol=row["symbol"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                quantity=row["quantity"],
                entry_ts_utc=row["entry_ts_utc"],
                fees_paid_usdt=row["fees_usdt"] or 0.0,
            )
        self._dirty = False
        return len(self._cache)

    def get_open_positions(self, symbol: str | None = None) -> list[OpenPosition]:
        """오픈 포지션 조회 (필요 시 자동 refresh)"""
        if self._dirty:
            self.refresh_from_db()

        positions = list(self._cache.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def has_open_position(self, symbol: str) -> bool:
        return len(self.get_open_positions(symbol)) > 0

    def add_position(self, position: OpenPosition) -> None:
        """새 포지션 등록 (ExecutionEngine에서 진입 성공 시 호출)"""
        self._cache[position.trade_id] = position

    def update_all_prices(self, prices: dict[str, float],
                          fee_rate: float = 0.0005) -> dict[str, float]:
        """
        모든 오픈 포지션의 현재가 일괄 갱신.

        Args:
            prices: {"BTCUSDT": 70000.0, "ETHUSDT": 3500.0}

        Returns:
            {"BTCUSDT_trade_5": +12.34, ...} unrealized PnL by trade
        """
        result = {}
        for pos in self.get_open_positions():
            price = prices.get(pos.symbol)
            if price is None:
                continue
            pos.update_market_price(price, fee_rate)
            result[f"{pos.symbol}_trade_{pos.trade_id}"] = pos.unrealized_pnl_usdt
        return result

    def total_unrealized_pnl_usdt(self) -> float:
        return sum(p.unrealized_pnl_usdt for p in self.get_open_positions())

    def close_position(
        self,
        trade_id: int,
        exit_order_id: int,
        exit_price: float,
        exit_reason: str,
        fee_rate: float = 0.0005,
    ) -> dict:
        """
        포지션 청산 + audit DB 업데이트.

        Returns:
            {"realized_pnl_usdt", "realized_pnl_pct", "fees_total_usdt"}
        """
        if trade_id not in self._cache:
            self.refresh_from_db()

        pos = self._cache.get(trade_id)
        if pos is None:
            raise ValueError(f"Trade {trade_id} not found in open positions")

        # PnL calculation (final, with both-side fees)
        exit_fee = exit_price * pos.quantity * fee_rate
        total_fees = pos.fees_paid_usdt + exit_fee

        if pos.is_long:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_fees
        net_pnl_pct = (net_pnl / (pos.entry_price * pos.quantity)) * 100

        log_trade_close(
            trade_id=trade_id,
            exit_order_id=exit_order_id,
            exit_price=exit_price,
            pnl_usdt=net_pnl,
            pnl_pct=net_pnl_pct,
            exit_reason=exit_reason,
            fees_usdt=total_fees,
        )

        # Remove from cache
        del self._cache[trade_id]

        return {
            "realized_pnl_usdt": net_pnl,
            "realized_pnl_pct": net_pnl_pct,
            "fees_total_usdt": total_fees,
        }

    def reconcile_with_exchange(self, exchange_positions: dict) -> dict:
        """
        거래소 실제 포지션과 DB 비교 — drift 감지.

        Args:
            exchange_positions: CCXT의 fetch_positions() 결과

        Returns:
            {"matches": int, "drifts": [...], "missing_in_db": [...]}
        """
        db_positions = {(p.symbol, p.direction): p for p in self.get_open_positions()}
        exchange_map = {}

        for ep in exchange_positions:
            symbol = ep.get("symbol", "").replace("/", "")
            contracts = ep.get("contracts", 0) or 0
            if contracts == 0:
                continue
            direction = "long" if contracts > 0 else "short"
            exchange_map[(symbol, direction)] = abs(contracts)

        matches = 0
        drifts = []
        missing_in_db = []

        for (symbol, direction), qty in exchange_map.items():
            db_pos = db_positions.get((symbol, direction))
            if db_pos is None:
                missing_in_db.append({"symbol": symbol, "direction": direction, "qty": qty})
                continue

            # qty mismatch (>1% diff)
            if abs(qty - db_pos.quantity) / max(db_pos.quantity, 1e-9) > 0.01:
                drifts.append({
                    "symbol": symbol, "direction": direction,
                    "db_qty": db_pos.quantity, "exchange_qty": qty,
                })
            else:
                matches += 1

        # DB에는 있지만 거래소에는 없는 경우 (수동 청산?)
        missing_in_exchange = []
        for (symbol, direction), pos in db_positions.items():
            if (symbol, direction) not in exchange_map:
                missing_in_exchange.append({
                    "symbol": symbol, "direction": direction,
                    "trade_id": pos.trade_id, "qty": pos.quantity,
                })

        return {
            "matches": matches,
            "drifts": drifts,
            "missing_in_db": missing_in_db,
            "missing_in_exchange": missing_in_exchange,
        }

    def summary(self) -> dict:
        """현재 포지션 요약 (Telegram 알림용)"""
        positions = self.get_open_positions()
        return {
            "n_positions": len(positions),
            "by_symbol": {
                s: sum(1 for p in positions if p.symbol == s)
                for s in set(p.symbol for p in positions)
            },
            "total_unrealized_pnl_usdt": self.total_unrealized_pnl_usdt(),
            "positions": [
                {
                    "trade_id": p.trade_id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "current_price": p.current_price,
                    "unrealized_pnl_usdt": p.unrealized_pnl_usdt,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "hold_minutes": round(p.hold_duration_minutes(), 1),
                }
                for p in positions
            ],
        }


# Module-level singleton (편의용)
_tracker: PositionTracker | None = None


def get_position_tracker() -> PositionTracker:
    global _tracker
    if _tracker is None:
        _tracker = PositionTracker()
    return _tracker
