"""
Mock vs Real Backtest Comparison
==================================
영길님이 PC에서 학습 완료 후 한 줄로 실행:
  python scripts/backtest_real_vs_mock.py --symbol BTCUSDT

이 스크립트의 가치:
  - Day 5에서 mock signal로 측정한 +1.93%가 실제 학습된 모델로는 어떻게 나오는가?
  - 학습 정확도 (Tier 2: ~70%, Tier 4: ~65%)가 mock의 'realistic' 가정과 얼마나 일치하는가?
  - 실제 모델이 mock의 어느 시나리오 (pessimistic/realistic/optimistic)에 가장 가까운가?

3가지 비교:
  1. Tier 1 단독 (Day 2 baseline): -72%
  2. 3-Tier with Mock signals (Day 5): +1.93% (realistic 기준)
  3. 3-Tier with REAL trained models (오늘 처음 측정 가능)

영길님 PC 학습 직후 실행 시간:
  - 30일 데이터: ~2분
  - 5년 데이터: ~30~60분 (캐싱으로 단축)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from flight_mind.utils.backtest_integrated import backtest_integrated
from flight_mind.utils.real_model_signals import check_model_status


CONSOLE = Console()


def show_model_status() -> bool:
    """학습된 모델 가용성 확인"""
    status = check_model_status()

    table = Table(title="Trained Model Status")
    table.add_column("Model", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Path / Test Acc")

    t2_status = "[green]✓ Available[/green]" if status.tier2_available else "[red]✗ Not trained[/red]"
    t2_detail = (
        f"{status.tier2_path}\nTest Acc: {status.tier2_test_acc:.4f}"
        if status.tier2_available else
        "Run: python -m flight_mind.tier2_pattern.train"
    )
    table.add_row("Tier 2 (CNN)", t2_status, t2_detail)

    t4_status = "[green]✓ Available[/green]" if status.tier4_available else "[red]✗ Not trained[/red]"
    t4_detail = (
        f"{status.tier4_path}\nTest Acc: {status.tier4_test_acc:.4f}"
        if status.tier4_available else
        "Run: python -m flight_mind.tier4_regime.train"
    )
    table.add_row("Tier 4 (Transformer)", t4_status, t4_detail)

    CONSOLE.print(table)

    if status.fallback_to_mock:
        CONSOLE.print(
            "\n[yellow]⚠ Both models not trained yet — comparison will use mock fallback[/yellow]"
        )
        CONSOLE.print(
            "[yellow]  영길님 PC에서 .\\setup_and_train.ps1 실행 후 다시 시도하세요.[/yellow]\n"
        )
        return False
    return True


def run_comparison(symbol: str, mock_mode: str = "realistic") -> dict:
    """Mock 'realistic' vs Real model 백테스트 비교"""
    CONSOLE.print(Panel(
        f"[bold]{symbol} — Mock 'realistic' vs Real Models[/bold]",
        expand=False,
    ))

    results = {}

    # 1. Mock backtest (baseline)
    CONSOLE.print("\n[cyan]━━━ Mock Signals (Day 5 baseline) ━━━[/cyan]")
    try:
        _, mock_metrics = backtest_integrated(
            symbol, mode=mock_mode, signal_source="mock",
        )
        results["mock"] = mock_metrics
    except Exception as e:
        CONSOLE.print(f"[red]Mock backtest failed: {e}[/red]")
        results["mock"] = {"n_trades": 0, "error": str(e)}

    # 2. Real model backtest
    CONSOLE.print("\n[cyan]━━━ Real Trained Models ━━━[/cyan]")
    try:
        _, real_metrics = backtest_integrated(
            symbol, mode=mock_mode, signal_source="real",
        )
        results["real"] = real_metrics
    except Exception as e:
        CONSOLE.print(f"[red]Real backtest failed: {e}[/red]")
        CONSOLE.print(
            "[yellow]Tip: 학습된 모델이 있는지 확인 후 다시 시도하세요.[/yellow]"
        )
        results["real"] = {"n_trades": 0, "error": str(e)}

    return results


def print_comparison_table(symbol: str, results: dict) -> None:
    """Mock vs Real 결과 비교 테이블"""
    mock = results.get("mock", {})
    real = results.get("real", {})

    table = Table(
        title=f"Mock vs Real Comparison — {symbol}",
        title_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Mock\n(Day 5 oracle)", justify="right", style="dim")
    table.add_column("Real\n(Trained models)", justify="right", style="bold yellow")
    table.add_column("Difference", justify="right", style="green")

    def fmt_or_dash(d, key, fmt="0", suffix=""):
        if d.get("n_trades", 0) == 0:
            return "—"
        v = d.get(key, 0)
        if isinstance(v, float):
            return f"{v:{fmt}}{suffix}"
        return str(v)

    def diff(a, b, fmt="+.2f"):
        if a is None or b is None:
            return "—"
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return "—"
        return f"{(b - a):{fmt}}"

    table.add_row(
        "거래 횟수",
        fmt_or_dash(mock, "n_trades"),
        fmt_or_dash(real, "n_trades"),
        diff(mock.get("n_trades"), real.get("n_trades"), fmt="+d"),
    )
    table.add_row(
        "승률 (%)",
        f"{mock.get('win_rate', 0) * 100:.1f}" if mock.get("n_trades") else "—",
        f"{real.get('win_rate', 0) * 100:.1f}" if real.get("n_trades") else "—",
        diff(
            mock.get("win_rate", 0) * 100 if mock.get("n_trades") else None,
            real.get("win_rate", 0) * 100 if real.get("n_trades") else None,
        ) + "pp" if mock.get("n_trades") and real.get("n_trades") else "—",
    )
    table.add_row(
        "Profit Factor",
        f"{mock.get('profit_factor', 0):.2f}" if mock.get("n_trades") else "—",
        f"{real.get('profit_factor', 0):.2f}" if real.get("n_trades") else "—",
        diff(
            mock.get("profit_factor") if mock.get("n_trades") else None,
            real.get("profit_factor") if real.get("n_trades") else None,
        ),
    )
    table.add_row(
        "총 수익률 (%)",
        f"{mock.get('total_return_pct', 0):+.2f}" if mock.get("n_trades") else "—",
        f"{real.get('total_return_pct', 0):+.2f}" if real.get("n_trades") else "—",
        diff(
            mock.get("total_return_pct") if mock.get("n_trades") else None,
            real.get("total_return_pct") if real.get("n_trades") else None,
        ),
    )
    table.add_row(
        "Avg Confluence",
        f"{mock.get('avg_confluence', 0):.3f}" if mock.get("n_trades") else "—",
        f"{real.get('avg_confluence', 0):.3f}" if real.get("n_trades") else "—",
        "—",
    )

    CONSOLE.print(table)

    # Real model 추가 진단
    if real.get("cache_stats"):
        cs = real["cache_stats"]
        CONSOLE.print(
            f"\n[dim]Real model cache — "
            f"T2 hit: {cs.get('t2_hit_rate', 0) * 100:.1f}%, "
            f"T4 hit: {cs.get('t4_hit_rate', 0) * 100:.1f}%, "
            f"T2 fail: {cs.get('t2_failure_rate', 0) * 100:.1f}%, "
            f"T4 fail: {cs.get('t4_failure_rate', 0) * 100:.1f}%[/dim]"
        )


def print_interpretation(results: dict) -> None:
    """결과 해석 — 영길님께 의미 설명"""
    mock = results.get("mock", {})
    real = results.get("real", {})

    if not real.get("n_trades", 0):
        CONSOLE.print("\n[yellow]Real model 결과가 없어 해석 생략[/yellow]")
        return

    if not mock.get("n_trades", 0):
        CONSOLE.print("\n[yellow]Mock 결과가 없어 비교 해석 생략[/yellow]")
        return

    real_pnl = real.get("total_return_pct", 0)
    mock_pnl = mock.get("total_return_pct", 0)

    CONSOLE.print("\n[bold]📊 결과 해석[/bold]\n")

    # 1. Real이 mock 대비 어디쯤 떨어졌는가
    if real_pnl > mock_pnl + 1:
        CONSOLE.print(
            f"  ✓ [green]Real model이 mock 'realistic'보다 우수[/green] "
            f"({real_pnl:+.2f}% vs {mock_pnl:+.2f}%)"
        )
        CONSOLE.print("    → 학계 SOTA 수준 달성. Optimistic 시나리오에 근접.")
    elif real_pnl > mock_pnl - 1:
        CONSOLE.print(
            f"  ✓ [yellow]Real model이 mock 'realistic'와 유사[/yellow] "
            f"({real_pnl:+.2f}% vs {mock_pnl:+.2f}%)"
        )
        CONSOLE.print("    → 합격선 근처. Paper trading 진행 가능.")
    else:
        CONSOLE.print(
            f"  ⚠ [red]Real model이 mock 'realistic' 미달[/red] "
            f"({real_pnl:+.2f}% vs {mock_pnl:+.2f}%)"
        )
        CONSOLE.print("    → 추가 학습 또는 하이퍼파라미터 튜닝 권장.")

    # 2. 거래 횟수 비교
    mock_n = mock.get("n_trades", 0)
    real_n = real.get("n_trades", 0)
    if abs(mock_n - real_n) > max(3, mock_n * 0.5):
        CONSOLE.print(
            f"\n  ⚠ 거래 횟수 큰 차이: mock={mock_n}, real={real_n}"
        )
        if real_n < mock_n * 0.3:
            CONSOLE.print("    → 모델이 너무 보수적 (대부분 hold) — Confluence threshold 검토")
        elif real_n > mock_n * 2:
            CONSOLE.print("    → 모델이 너무 공격적 — overfitting 가능성")

    # 3. 승률 비교
    if mock.get("win_rate") and real.get("win_rate"):
        mock_wr = mock["win_rate"] * 100
        real_wr = real["win_rate"] * 100
        diff_wr = real_wr - mock_wr
        if abs(diff_wr) > 10:
            direction = "낮음" if diff_wr < 0 else "높음"
            CONSOLE.print(
                f"\n  📊 승률 차이 큼: real {real_wr:.1f}% vs mock {mock_wr:.1f}% "
                f"({direction} {abs(diff_wr):.1f}%p)"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--mock-mode", default="realistic",
                   choices=["pessimistic", "realistic", "optimistic"],
                   help="Mock baseline 시나리오 (default: realistic)")
    p.add_argument("--save-to", default=None,
                   help="결과를 JSON으로 저장할 경로")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="여러 심볼 일괄 비교")
    args = p.parse_args()

    # 1. 모델 가용성 확인
    has_real = show_model_status()

    if not has_real:
        CONSOLE.print(
            Panel(
                "[bold yellow]학습된 모델이 없으므로 mock-only 백테스트로 진행합니다.[/bold yellow]\n"
                "영길님 PC에서 학습 완료 후 이 스크립트를 다시 실행하면\n"
                "Mock vs Real 직접 비교가 가능합니다.",
                title="Notice",
                expand=False,
            )
        )
        # mock만 실행 (Day 5와 동일)
        from flight_mind.utils.backtest_integrated import (
            print_comparison_table as print_legacy_table,
            print_pipeline_diagnostics)

        results = {}
        for mode in ["pessimistic", "realistic", "optimistic"]:
            CONSOLE.print(f"\n[cyan]━━━ Mock {mode.upper()} ━━━[/cyan]")
            try:
                _, m = backtest_integrated(
                    args.symbol, mode=mode, signal_source="mock",
                )
                results[mode] = m
            except Exception as e:
                CONSOLE.print(f"[red]Failed: {e}[/red]")
                results[mode] = {"n_trades": 0}

        print_legacy_table(results, args.symbol)
        print_pipeline_diagnostics(args.symbol, results)
        return 0

    # 2. Mock vs Real 비교 (모델 있을 때)
    symbols = args.symbols or [args.symbol]
    all_results = {}

    for symbol in symbols:
        results = run_comparison(symbol, mock_mode=args.mock_mode)
        all_results[symbol] = results

        CONSOLE.print()
        print_comparison_table(symbol, results)
        print_interpretation(results)

    # 3. 결과 저장
    if args.save_to:
        save_path = Path(args.save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(all_results, indent=2, default=str))
        CONSOLE.print(f"\n[dim]Results saved: {save_path}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
