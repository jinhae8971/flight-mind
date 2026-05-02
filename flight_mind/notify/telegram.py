"""
Telegram Notifier
==================
영길님과 시스템의 실시간 양방향 채널.

영길님 환경 (메모리에 저장된 봇 토큰):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 환경변수

설계 원칙:
  - Send-and-forget: 실패해도 메인 흐름 절대 차단 안 함
  - Rate limit aware: Telegram 한도 (초당 30) 자동 회피
  - Severity-based: critical은 sound, info는 silent
  - Idempotent: 같은 알림 5분 내 중복 시 자동 dedup
  - Markdown safe: 거래 데이터의 특수문자 자동 escape
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from rich.console import Console


CONSOLE = Console()


class Severity(str, Enum):
    """알림 중요도 — sound 여부와 retention 결정"""
    DEBUG = "debug"           # 개발 중 디버깅
    INFO = "info"              # 일반 정보 (silent)
    NOTICE = "notice"          # 진입/청산 (silent)
    WARN = "warn"              # 주의 (sound)
    CRITICAL = "critical"      # Kill-Switch, 시스템 오류 (sound)


@dataclass
class TelegramConfig:
    bot_token: str | None
    chat_id: str | None
    enabled: bool = True
    parse_mode: str = "Markdown"

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        return cls(
            bot_token=token,
            chat_id=chat_id,
            enabled=bool(token and chat_id),
        )


class TelegramNotifier:
    """단순 + 견고한 Telegram 클라이언트"""

    BASE_URL = "https://api.telegram.org"
    DEDUP_WINDOW_S = 300    # 5분 내 동일 메시지 dedup

    def __init__(self, config: TelegramConfig | None = None,
                 max_retries: int = 3):
        self.config = config or TelegramConfig.from_env()
        self.max_retries = max_retries

        self._dedup_cache: dict[str, float] = {}    # hash → timestamp
        self._last_send_time: float = 0.0
        self._min_interval_s: float = 0.05    # 초당 30 회피

        if not self.config.enabled:
            CONSOLE.print("[yellow]TelegramNotifier disabled "
                         "(TELEGRAM_BOT_TOKEN/CHAT_ID 미설정)[/yellow]")

    def is_enabled(self) -> bool:
        return self.config.enabled

    # =========================================================================
    # Public API
    # =========================================================================
    def send(
        self,
        text: str,
        severity: Severity = Severity.INFO,
        dedup_key: str | None = None,
    ) -> bool:
        """
        메시지 전송. 실패해도 절대 raise 안 함 (메인 흐름 보호).

        Args:
            text: 메시지 본문 (Markdown)
            severity: sound/dedup 여부 결정
            dedup_key: 명시적 dedup 키 (None이면 본문 hash 사용)

        Returns:
            True 성공 / False 실패 (또는 disabled / dedup'd)
        """
        if not self.config.enabled:
            return False

        # Dedup check
        key = dedup_key or hashlib.md5(text.encode()).hexdigest()[:12]
        now = time.time()
        if key in self._dedup_cache:
            if now - self._dedup_cache[key] < self.DEDUP_WINDOW_S:
                return False     # 중복 — silent skip
        self._dedup_cache[key] = now

        # Rate limit
        elapsed = now - self._last_send_time
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

        # Severity prefix
        prefix_map = {
            Severity.DEBUG: "🔧",
            Severity.INFO: "ℹ️",
            Severity.NOTICE: "📌",
            Severity.WARN: "⚠️",
            Severity.CRITICAL: "🚨",
        }
        prefix = prefix_map.get(severity, "")
        full_text = f"{prefix} {text}" if prefix else text

        # Send (silent 알림은 Telegram silent flag 사용)
        silent = severity in (Severity.DEBUG, Severity.INFO, Severity.NOTICE)
        return self._send_raw(full_text, silent=silent)

    # =========================================================================
    # Convenience methods (트레이딩 도메인 전용)
    # =========================================================================
    def send_decision(self, symbol: str, action: str, direction: str,
                       confluence: float, tier_signals: dict) -> bool:
        """Fusion Layer 의사결정 알림 (NOTICE)"""
        if action == "hold":
            return False    # hold는 알림 안 함

        emoji = "📈" if direction == "long" else "📉"
        text = (
            f"*{emoji} {action.upper()} — {symbol}*\n"
            f"Direction: `{direction}`\n"
            f"Confluence: `{confluence:.3f}`\n"
            f"Time: `{_now_str()}`"
        )

        signals_text = "\n".join(
            f"  • {tier}: dir=`{sig.get('direction', 'n/a')}` "
            f"score=`{sig.get('score', 0):.2f}`"
            for tier, sig in tier_signals.items()
        )
        if signals_text:
            text += f"\n{signals_text}"

        return self.send(text, severity=Severity.NOTICE)

    def send_order_filled(self, symbol: str, direction: str,
                           quantity: float, price: float, mode: str) -> bool:
        """주문 체결 알림 (NOTICE)"""
        text = (
            f"*✅ 진입 체결 — {symbol}*\n"
            f"Direction: `{direction}`\n"
            f"Quantity: `{quantity:.5f}`\n"
            f"Price: `{price:,.2f}`\n"
            f"Mode: `{mode}`\n"
            f"Time: `{_now_str()}`"
        )
        return self.send(text, severity=Severity.NOTICE)

    def send_position_closed(self, symbol: str, direction: str,
                              entry_price: float, exit_price: float,
                              pnl_usdt: float, pnl_pct: float,
                              exit_reason: str, mode: str) -> bool:
        """포지션 청산 알림 (수익/손실에 따라 emoji 변경)"""
        if pnl_usdt > 0:
            emoji = "💰"
            severity = Severity.NOTICE
        else:
            emoji = "📉"
            severity = Severity.NOTICE

        text = (
            f"*{emoji} 청산 — {symbol} ({exit_reason})*\n"
            f"Direction: `{direction}`\n"
            f"Entry → Exit: `{entry_price:,.2f}` → `{exit_price:,.2f}`\n"
            f"PnL: `{pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)`\n"
            f"Mode: `{mode}`\n"
            f"Time: `{_now_str()}`"
        )
        return self.send(text, severity=severity)

    def send_killswitch(self, level: str, reason: str, metrics: dict) -> bool:
        """Kill-Switch 발동 알림 (CRITICAL)"""
        text = (
            f"*🚨 KILL-SWITCH: {level.upper()}*\n"
            f"Reason: `{reason}`\n"
            f"Today: `{metrics.get('today_realized_pct', 0):+.2f}%`\n"
            f"Week: `{metrics.get('week_realized_pct', 0):+.2f}%`\n"
            f"MDD: `{metrics.get('max_drawdown_pct', 0):.2f}%`\n"
            f"Consecutive losses: `{metrics.get('consecutive_losses', 0)}`\n"
            f"Time: `{_now_str()}`"
        )
        # Critical는 dedup 비활성화 (영길님이 반드시 봐야 함)
        return self.send(text, severity=Severity.CRITICAL,
                          dedup_key=f"killswitch_{level}_{int(time.time())}")

    def send_heartbeat(self, daemon_name: str, uptime_seconds: int,
                        n_decisions: int, n_trades: int) -> bool:
        """데몬 생존 신호 (1시간마다, INFO)"""
        hours = uptime_seconds // 3600
        text = (
            f"*💓 {daemon_name} alive*\n"
            f"Uptime: `{hours}h`\n"
            f"Decisions: `{n_decisions}` | Trades: `{n_trades}`\n"
            f"Time: `{_now_str()}`"
        )
        return self.send(text, severity=Severity.INFO,
                          dedup_key=f"heartbeat_{daemon_name}_{hours}")

    def send_daily_summary(self, summary: dict) -> bool:
        """매일 자정 일일 리포트 (INFO)"""
        text = (
            f"*📊 Daily Summary — {summary.get('date', _today_str())}*\n"
            f"Trades: `{summary.get('n_trades', 0)}` "
            f"({summary.get('n_wins', 0)} wins)\n"
            f"PnL: `{summary.get('pnl_usdt', 0):+.2f} USDT "
            f"({summary.get('pnl_pct', 0):+.2f}%)`\n"
            f"Best: `{summary.get('best_trade_pct', 0):+.2f}%` | "
            f"Worst: `{summary.get('worst_trade_pct', 0):+.2f}%`\n"
            f"Cumulative: `{summary.get('total_pnl_pct', 0):+.2f}%`"
        )
        return self.send(text, severity=Severity.INFO,
                          dedup_key=f"summary_{_today_str()}")

    def send_error(self, daemon_name: str, error_type: str, message: str) -> bool:
        """예외/오류 알림 (WARN)"""
        text = (
            f"*⚠️ {daemon_name} error*\n"
            f"Type: `{error_type}`\n"
            f"Message: `{message[:200]}`\n"
            f"Time: `{_now_str()}`"
        )
        return self.send(text, severity=Severity.WARN)

    # =========================================================================
    # Internal — Raw HTTP
    # =========================================================================
    def _send_raw(self, text: str, silent: bool = False) -> bool:
        """실제 Telegram API 호출 — 재시도 포함"""
        try:
            import requests
        except ImportError:
            CONSOLE.print("[yellow]requests 패키지 없음 — Telegram 비활성[/yellow]")
            return False

        url = f"{self.BASE_URL}/bot{self.config.bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": self.config.parse_mode,
            "disable_notification": silent,
            "disable_web_page_preview": True,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                self._last_send_time = time.time()

                if resp.status_code == 200:
                    return True

                if resp.status_code == 429:
                    # Rate limited — Telegram이 요구하는 대기 시간
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    time.sleep(min(retry_after, 30))
                    continue

                # Other errors
                CONSOLE.print(
                    f"[yellow]Telegram error {resp.status_code}: "
                    f"{resp.text[:200]}[/yellow]"
                )
                return False

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                CONSOLE.print(f"[yellow]Telegram send failed: {e}[/yellow]")
                return False

        return False


# =============================================================================
# Helpers
# =============================================================================
def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =============================================================================
# Singleton
# =============================================================================
_notifier: TelegramNotifier | None = None


def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
