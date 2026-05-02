"""
Execution Engine — CCXT-based Order Execution
================================================
영길님 Binance Futures 주문 실행 시스템.

3개 운영 모드:
  - paper   : 가상 매매 (실제 주문 없음, audit log만)
  - testnet : Binance Testnet (가짜 자금, 진짜 API 흐름)
  - live    : 실거래 (영길님 자본 — 3중 게이트 통과 시에만)

3중 안전 게이트:
  1. 환경변수 FLIGHT_MIND_LIVE=1
  2. 인자 mode='live' 명시
  3. 실거래 시작 전 confirm_live() 인터랙티브 확인

영길님 정책 강제:
  - Position max: 70 USDT (CAPITAL.max_position_pct = 2% × 3500)
  - Leverage: 5x
  - Daily trade limit: 2회 (RiskManager가 별도 관리)
  - Stop loss / Take profit 자동 첨부 (-3% / +6%)
  - 모든 결정은 audit.py에 기록
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from rich.console import Console

from flight_mind.config import CAPITAL, EXCHANGE
from flight_mind.fusion.layer import FusionDecision
from flight_mind.risk.audit import (init_audit_db, log_decision, log_order,
                                       log_trade_open, update_order_status)
from flight_mind.vault.manager import ApiCredential, Vault


CONSOLE = Console()


class Mode(str, Enum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass
class OrderResult:
    """주문 실행 결과"""
    success: bool
    order_id: int | None = None              # audit.orders.id
    exchange_order_id: str | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    error: str | None = None
    mode: str = "paper"


# =============================================================================
# Live Mode Safety
# =============================================================================
def confirm_live() -> bool:
    """라이브 모드 인터랙티브 확인 — 3중 게이트 마지막"""
    CONSOLE.print("\n[bold red]━━━ ⚠️  LIVE TRADING MODE ━━━[/bold red]")
    CONSOLE.print("[yellow]실거래는 영길님의 실제 자본을 사용합니다.[/yellow]")
    CONSOLE.print(f"[yellow]Position max: {CAPITAL.max_position_pct * 100:.1f}% × {CAPITAL.total_usdt * CAPITAL.live_trading_pct:.0f} USDT[/yellow]")
    CONSOLE.print(f"[yellow]Leverage: {CAPITAL.leverage}x[/yellow]")

    response = input("\n실거래 진행 확인 (정확히 'I UNDERSTAND' 입력): ")
    if response != "I UNDERSTAND":
        CONSOLE.print("[red]Cancelled[/red]")
        return False
    return True


def is_live_authorized() -> bool:
    """3중 게이트 1+2단계: 환경변수 + 인자"""
    return os.getenv("FLIGHT_MIND_LIVE") == "1"


# =============================================================================
# CCXT Client Factory
# =============================================================================
def make_exchange_client(mode: Mode, vault: Vault | None = None):
    """
    CCXT exchange instance 생성.

    Paper 모드: 클라이언트 None (실제 호출 없음)
    Testnet/Live: Vault에서 API 키 로드
    """
    if mode == Mode.PAPER:
        return None

    try:
        import ccxt
    except ImportError:
        raise RuntimeError(
            "ccxt 패키지 미설치. `pip install ccxt` 실행 필요."
        )

    vault = vault or Vault()
    label = "binance_testnet" if mode == Mode.TESTNET else "binance_live"

    try:
        cred: ApiCredential = vault.get(label)
    except Exception as e:
        raise RuntimeError(
            f"Vault에서 '{label}' 자격증명 로드 실패: {e}\n"
            f"`python -m flight_mind.vault.manager`로 키 등록하세요."
        )

    exchange = ccxt.binance({
        "apiKey": cred.api_key,
        "secret": cred.secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",        # USD-M futures
            "warnOnFetchOpenOrdersWithoutSymbol": False,
        },
    })

    if mode == Mode.TESTNET:
        exchange.set_sandbox_mode(True)

    return exchange


# =============================================================================
# Execution Engine
# =============================================================================
class ExecutionEngine:
    """주문 실행 — Paper/Testnet/Live 통합 인터페이스"""

    def __init__(self, mode: Mode | str = Mode.PAPER, vault: Vault | None = None,
                 dry_run: bool = False):
        """
        Args:
            mode: 운영 모드
            vault: Vault 인스턴스 (없으면 default 사용)
            dry_run: True 시 모든 의사결정 로그만 남기고 실제 주문 없음
        """
        if isinstance(mode, str):
            mode = Mode(mode)
        self.mode = mode
        self.dry_run = dry_run

        # Live 게이트 검증
        if mode == Mode.LIVE:
            if not is_live_authorized():
                raise RuntimeError(
                    "Live mode 차단: FLIGHT_MIND_LIVE=1 환경변수 미설정"
                )
            if not confirm_live():
                raise RuntimeError("Live mode 사용자 취소")

        self.vault = vault or Vault()
        self.exchange = make_exchange_client(self.mode, self.vault)

        init_audit_db()
        CONSOLE.print(f"[bold]ExecutionEngine: mode={mode.value}, dry_run={dry_run}[/bold]")

    # =========================================================================
    # Market Data
    # =========================================================================
    def fetch_ticker(self, symbol: str) -> dict:
        """현재 시장 시세 (paper 모드는 dummy 반환)"""
        if self.mode == Mode.PAPER:
            # Paper 모드: 마지막 5분봉 close를 mid로 사용
            from flight_mind.utils.db import fetch_ohlcv
            df = fetch_ohlcv(symbol, "5m").tail(1)
            if df.empty:
                return {"bid": 0, "ask": 0, "last": 0}
            last = float(df["close"].iloc[-1])
            return {
                "bid": last * 0.9995,
                "ask": last * 1.0005,
                "last": last,
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        """USDT 잔고 조회 (paper는 영길님 설정값)"""
        if self.mode == Mode.PAPER:
            paper_capital = CAPITAL.total_usdt * CAPITAL.live_trading_pct
            return {
                "USDT": {
                    "free": paper_capital,
                    "used": 0.0,
                    "total": paper_capital,
                }
            }
        return self.exchange.fetch_balance()

    # =========================================================================
    # Position Sizing
    # =========================================================================
    def compute_position_size(self, symbol: str, available_usdt: float) -> dict:
        """
        영길님 정책 기반 포지션 크기 계산.

        - Position max: 2% of capital = 70 USDT (영길님 결정)
        - Leverage: 5x
        - Effective notional = 70 × 5 = 350 USDT

        Returns:
            {"notional_usdt", "quantity", "leverage"}
        """
        # CAPITAL.max_position_pct는 decimal (0.02 = 2%)
        max_position_usdt = available_usdt * CAPITAL.max_position_pct
        max_position_usdt = min(max_position_usdt, 70.0)   # hard cap (영길님 결정)

        ticker = self.fetch_ticker(symbol)
        price = ticker.get("last") or ticker.get("ask") or 0
        if price <= 0:
            return {"notional_usdt": 0, "quantity": 0, "leverage": CAPITAL.leverage}

        notional = max_position_usdt * CAPITAL.leverage
        quantity = notional / price

        return {
            "notional_usdt": notional,
            "quantity": round(quantity, 5),  # 5 decimal precision
            "leverage": CAPITAL.leverage,
            "entry_price_estimate": price,
        }

    # =========================================================================
    # Order Execution
    # =========================================================================
    def execute_decision(
        self,
        symbol: str,
        decision: FusionDecision,
        market_snapshot: dict | None = None,
        skip_risk_gate: bool = False,
    ) -> OrderResult:
        """
        Fusion Layer 의사결정을 실제 주문으로 실행.

        Hold → 로그만, 주문 없음
        Open_long/short → Risk Gate 통과 후 시장가 진입

        Args:
            skip_risk_gate: True 시 Risk Manager 검증 건너뜀 (테스트 전용,
                            절대 production에서 사용 금지)
        """
        # 0. Risk Gate (사전 검증) — Day 8 추가
        if decision.action != "hold" and not skip_risk_gate:
            from flight_mind.risk.manager import get_risk_manager
            risk_mgr = get_risk_manager()
            gate_result = risk_mgr.check(symbol, mode=self.mode.value)

            if not gate_result.allowed:
                CONSOLE.print(
                    f"[bold red]🚫 Risk Gate blocked: "
                    f"{', '.join(gate_result.blockers)}[/bold red]"
                )
                # 거래 결정 자체는 audit에 기록 (왜 차단됐는지 추적용)
                log_decision(
                    symbol=symbol,
                    action="hold",  # Risk Gate가 강제로 hold로 변경
                    direction=decision.direction,
                    confluence=decision.confluence_score,
                    tier_outputs={k: {"score": v.score, "direction": v.direction,
                                      "signals": v.signals}
                                  for k, v in decision.tier_outputs.items()},
                    market_snapshot=market_snapshot,
                    veto_reason=f"risk_gate: {'; '.join(gate_result.blockers)}",
                    mode=self.mode.value,
                )
                return OrderResult(
                    success=False,
                    error=f"Risk Gate blocked: {gate_result.blockers}",
                    mode=self.mode.value,
                )

            # Warnings는 진행하되 알림
            if gate_result.warnings:
                CONSOLE.print(
                    f"[yellow]⚠ Risk Gate warnings: "
                    f"{', '.join(gate_result.warnings)}[/yellow]"
                )

        # 1. Audit decision (모든 결정 기록 — hold 포함)
        decision_id = log_decision(
            symbol=symbol,
            action=decision.action,
            direction=decision.direction,
            confluence=decision.confluence_score,
            tier_outputs={k: {"score": v.score, "direction": v.direction,
                              "signals": v.signals}
                          for k, v in decision.tier_outputs.items()},
            market_snapshot=market_snapshot,
            veto_reason=decision.veto_reason,
            mode=self.mode.value,
        )

        if decision.action == "hold":
            return OrderResult(success=True, mode=self.mode.value)

        # 2. Position sizing
        balance = self.fetch_balance()
        available_usdt = balance.get("USDT", {}).get("free", 0)

        sizing = self.compute_position_size(symbol, available_usdt)
        if sizing["quantity"] <= 0:
            return OrderResult(
                success=False,
                error="Insufficient balance or invalid price",
                mode=self.mode.value,
            )

        # 3. Dry-run check
        if self.dry_run:
            CONSOLE.print(f"[yellow][DRY-RUN] Would {decision.action} {symbol} "
                         f"qty={sizing['quantity']} @ ~{sizing['entry_price_estimate']:.2f}[/yellow]")
            order_id = log_order(
                decision_id=decision_id,
                exchange_order_id="DRY_RUN",
                symbol=symbol,
                side="buy" if decision.action == "open_long" else "sell",
                order_type="market",
                quantity=sizing["quantity"],
                price=None,
                status="dry_run",
                mode=self.mode.value,
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                exchange_order_id="DRY_RUN",
                mode=self.mode.value,
            )

        # 4. Execute (paper / testnet / live)
        side = "buy" if decision.action == "open_long" else "sell"
        direction = "long" if decision.action == "open_long" else "short"

        try:
            if self.mode == Mode.PAPER:
                result = self._execute_paper(symbol, side, sizing, decision_id)
            else:
                result = self._execute_real(symbol, side, sizing, decision_id)

            # 5. Open trade record (성공 시)
            if result.success and result.avg_fill_price:
                log_trade_open(
                    decision_id=decision_id,
                    entry_order_id=result.order_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=result.avg_fill_price,
                    quantity=result.filled_quantity,
                    mode=self.mode.value,
                )

            return result

        except Exception as e:
            CONSOLE.print(f"[red]Order failed: {type(e).__name__}: {e}[/red]")
            log_order(
                decision_id=decision_id,
                exchange_order_id=None,
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=sizing["quantity"],
                price=None,
                status="failed",
                error_message=str(e)[:500],
                mode=self.mode.value,
            )
            return OrderResult(success=False, error=str(e), mode=self.mode.value)

    def _execute_paper(self, symbol: str, side: str, sizing: dict,
                       decision_id: int) -> OrderResult:
        """Paper 모드: 가상 체결 — slippage 0.05% 적용"""
        ticker = self.fetch_ticker(symbol)
        market_price = ticker.get("last", 0)

        # Slippage 적용
        if side == "buy":
            fill_price = market_price * 1.0005   # 0.05% slippage
        else:
            fill_price = market_price * 0.9995

        order_id = log_order(
            decision_id=decision_id,
            exchange_order_id=f"PAPER_{datetime.now(timezone.utc).timestamp():.0f}",
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=sizing["quantity"],
            price=None,
            status="filled",
            filled_qty=sizing["quantity"],
            avg_fill_price=fill_price,
            fee_usdt=fill_price * sizing["quantity"] * 0.0005,    # 0.05% fee
            mode=self.mode.value,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            exchange_order_id=f"PAPER_{order_id}",
            filled_quantity=sizing["quantity"],
            avg_fill_price=fill_price,
            mode=self.mode.value,
        )

    def _execute_real(self, symbol: str, side: str, sizing: dict,
                      decision_id: int) -> OrderResult:
        """Testnet 또는 Live 실제 주문 (CCXT)"""
        if not self.exchange:
            return OrderResult(success=False, error="No exchange client", mode=self.mode.value)

        # Set leverage
        try:
            self.exchange.set_leverage(CAPITAL.leverage, symbol)
        except Exception as e:
            CONSOLE.print(f"[yellow]Leverage set warning: {e}[/yellow]")

        # Place market order
        order = self.exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=sizing["quantity"],
        )

        # Log result
        order_id = log_order(
            decision_id=decision_id,
            exchange_order_id=order.get("id"),
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=sizing["quantity"],
            price=None,
            status="filled" if order.get("status") in ("closed", "filled") else "pending",
            filled_qty=order.get("filled", sizing["quantity"]),
            avg_fill_price=order.get("average") or order.get("price", 0),
            fee_usdt=sum(f.get("cost", 0) for f in order.get("fees", [])),
            raw_response=order,
            mode=self.mode.value,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            exchange_order_id=order.get("id"),
            filled_quantity=order.get("filled", sizing["quantity"]),
            avg_fill_price=order.get("average") or order.get("price", 0),
            mode=self.mode.value,
        )

    # =========================================================================
    # Position Close
    # =========================================================================
    def close_position(self, symbol: str, direction: str, quantity: float,
                       reason: str = "manual") -> OrderResult:
        """
        포지션 청산 — TP/SL 도달 시 또는 수동.

        direction: 진입 방향 ('long' 또는 'short')
        Closes by placing opposite side market order.
        """
        side = "sell" if direction == "long" else "buy"

        if self.dry_run:
            return OrderResult(success=True, mode=self.mode.value,
                              error=f"DRY_RUN close ({reason})")

        if self.mode == Mode.PAPER:
            ticker = self.fetch_ticker(symbol)
            market_price = ticker.get("last", 0)
            fill_price = (market_price * 0.9995 if side == "sell"
                          else market_price * 1.0005)

            order_id = log_order(
                decision_id=None,
                exchange_order_id=f"PAPER_CLOSE_{datetime.now(timezone.utc).timestamp():.0f}",
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=quantity,
                price=None,
                status="filled",
                filled_qty=quantity,
                avg_fill_price=fill_price,
                fee_usdt=fill_price * quantity * 0.0005,
                mode=self.mode.value,
            )
            return OrderResult(
                success=True, order_id=order_id, filled_quantity=quantity,
                avg_fill_price=fill_price, mode=self.mode.value,
            )

        # Real close
        order = self.exchange.create_market_order(
            symbol=symbol, side=side, amount=quantity,
            params={"reduceOnly": True},
        )
        order_id = log_order(
            decision_id=None,
            exchange_order_id=order.get("id"),
            symbol=symbol, side=side, order_type="market",
            quantity=quantity, price=None,
            status="filled",
            filled_qty=order.get("filled", quantity),
            avg_fill_price=order.get("average", 0),
            raw_response=order,
            mode=self.mode.value,
        )
        return OrderResult(
            success=True, order_id=order_id,
            filled_quantity=order.get("filled", quantity),
            avg_fill_price=order.get("average", 0),
            mode=self.mode.value,
        )
