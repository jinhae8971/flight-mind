"""
Environment Diagnostic
=======================
영길님 PC 환경을 점검하고 학습 파이프라인 실행 가능 여부를 판단.

Checks:
  1. Python 버전 (3.11+ 권장)
  2. 핵심 의존성 설치 여부
  3. CUDA / GPU 메모리
  4. 디스크 여유 공간 (5년 데이터 + 모델 체크포인트)
  5. 네트워크 연결 (Binance Vision, GitHub)
  6. 권장 batch_size 자동 산출 (GPU VRAM 기준)

Usage:
    python scripts/diagnose_env.py

Output:
    /tmp/flight_mind_env.json — 다른 스크립트가 사용
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table


CONSOLE = Console()


@dataclass
class EnvReport:
    """진단 결과 — JSON 직렬화 가능"""
    python_version: str = ""
    python_ok: bool = False

    deps_installed: dict = field(default_factory=dict)
    deps_ok: bool = False

    cuda_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    recommended_t2_batch: int = 32
    recommended_t4_batch: int = 64

    disk_free_gb: float = 0.0
    disk_ok: bool = False

    network_binance: bool = False
    network_github: bool = False
    network_ok: bool = False

    overall_ok: bool = False
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# =============================================================================
# Individual Checks
# =============================================================================
def check_python(report: EnvReport) -> None:
    v = sys.version_info
    report.python_version = f"{v.major}.{v.minor}.{v.micro}"
    report.python_ok = v.major == 3 and v.minor >= 11
    if not report.python_ok:
        report.blockers.append(
            f"Python {report.python_version} < 3.11 필요. "
            "pyenv 또는 Anaconda로 3.11+ 설치 권장."
        )


def check_dependencies(report: EnvReport) -> None:
    required = {
        "numpy": "1.26.0",
        "pandas": "2.0.0",
        "scipy": "1.13.0",
        "duckdb": "0.10.0",
        "pyarrow": "15.0.0",
        "torch": "2.0.0",
        "pyts": "0.13.0",
        "rich": "13.0.0",
        "requests": "2.30.0",
    }
    optional = {
        "timm": "1.0.0",
        "torchvision": "0.18.0",
    }

    deps = {}
    for pkg, min_ver in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            deps[pkg] = {"installed": True, "version": ver, "required": True}
        except ImportError:
            deps[pkg] = {"installed": False, "version": None, "required": True}
            report.blockers.append(f"필수 패키지 누락: {pkg} >= {min_ver}")

    for pkg, min_ver in optional.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            deps[pkg] = {"installed": True, "version": ver, "required": False}
        except ImportError:
            deps[pkg] = {"installed": False, "version": None, "required": False}
            report.warnings.append(
                f"선택 패키지 미설치: {pkg} (Tier 2 fallback 사용 — 성능 저하 가능)"
            )

    report.deps_installed = deps
    report.deps_ok = all(d["installed"] for p, d in deps.items() if d["required"])


def check_gpu(report: EnvReport) -> None:
    try:
        import torch
        report.cuda_available = torch.cuda.is_available()

        if report.cuda_available:
            report.gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            report.gpu_vram_gb = round(vram_bytes / (1024 ** 3), 1)

            # batch_size 추천 (heuristic from 학계 + 영길님 결정)
            # Tier 2 (ResNet-18): VRAM ~ batch * 130MB
            # Tier 4 (Transformer): VRAM ~ batch * 50MB
            usable_vram = report.gpu_vram_gb * 0.85   # 15% 안전 마진
            report.recommended_t2_batch = max(16, min(128, int(usable_vram * 1024 / 130)))
            report.recommended_t4_batch = max(32, min(256, int(usable_vram * 1024 / 50)))
        else:
            report.warnings.append(
                "CUDA 미탐지 — CPU 학습 가능하지만 매우 느림 "
                "(Tier 2: 약 80~120시간 예상)"
            )
            report.recommended_t2_batch = 16
            report.recommended_t4_batch = 32
    except ImportError:
        report.blockers.append("PyTorch 미설치")


def check_disk(report: EnvReport) -> None:
    """현재 디렉토리 기준 여유 공간 체크"""
    cwd = Path.cwd()
    stat = shutil.disk_usage(cwd)
    report.disk_free_gb = round(stat.free / (1024 ** 3), 1)

    # 필요 공간 추정:
    # - Binance Vision 5y BTC + ETH 5m: 약 4 GB
    # - DuckDB feature store: 약 2 GB
    # - Model checkpoints: 약 500 MB
    # - 학습 중간 산출물: 약 2 GB
    # - 안전 마진: 5 GB
    # 합계 약 15 GB
    REQUIRED_GB = 15.0

    report.disk_ok = report.disk_free_gb >= REQUIRED_GB
    if not report.disk_ok:
        report.blockers.append(
            f"디스크 여유 공간 부족: {report.disk_free_gb}GB / "
            f"권장 {REQUIRED_GB}GB. 다른 곳으로 이동 또는 정리 필요."
        )


def check_network(report: EnvReport) -> None:
    import socket

    def ping(host: str, port: int = 443, timeout: float = 5.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (OSError, socket.timeout):
            return False

    report.network_binance = ping("data.binance.vision")
    report.network_github = ping("github.com")
    report.network_ok = report.network_binance and report.network_github

    if not report.network_binance:
        report.blockers.append("Binance Vision 접속 실패 — 데이터 다운로드 불가")
    if not report.network_github:
        report.warnings.append("GitHub 접속 실패 — 레포 동기화 불가")


# =============================================================================
# Main
# =============================================================================
def diagnose() -> EnvReport:
    report = EnvReport()

    CONSOLE.print("\n[bold cyan]━━━ Flight-Mind 환경 진단 ━━━[/bold cyan]\n")

    with CONSOLE.status("환경 점검 중..."):
        check_python(report)
        check_dependencies(report)
        check_gpu(report)
        check_disk(report)
        check_network(report)

    # Final assessment
    report.overall_ok = (
        report.python_ok and report.deps_ok and report.disk_ok
        and report.network_ok and report.cuda_available
    )

    return report


def print_report(report: EnvReport) -> None:
    table = Table(title="Environment Diagnostic", title_style="bold")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    def row(name, ok, detail):
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(name, status, detail)

    row("Python ≥ 3.11", report.python_ok, report.python_version)
    row("Dependencies", report.deps_ok,
        f"{sum(1 for d in report.deps_installed.values() if d['installed'])} / "
        f"{len(report.deps_installed)} installed")
    row("CUDA / GPU", report.cuda_available,
        f"{report.gpu_name} ({report.gpu_vram_gb} GB VRAM)" if report.cuda_available else "Not available")
    row("Disk Space", report.disk_ok, f"{report.disk_free_gb} GB free / 15 GB required")
    row("Network", report.network_ok,
        f"Binance: {'✓' if report.network_binance else '✗'}, "
        f"GitHub: {'✓' if report.network_github else '✗'}")

    CONSOLE.print(table)

    if report.cuda_available:
        CONSOLE.print(f"\n[bold]권장 batch_size:[/bold]")
        CONSOLE.print(f"  Tier 2 (CNN):         {report.recommended_t2_batch}")
        CONSOLE.print(f"  Tier 4 (Transformer): {report.recommended_t4_batch}")

    if report.blockers:
        CONSOLE.print("\n[bold red]🚫 Blockers (반드시 해결):[/bold red]")
        for b in report.blockers:
            CONSOLE.print(f"  • {b}")

    if report.warnings:
        CONSOLE.print("\n[bold yellow]⚠️  Warnings (권장):[/bold yellow]")
        for w in report.warnings:
            CONSOLE.print(f"  • {w}")

    if report.overall_ok:
        CONSOLE.print("\n[bold green]✅ 모든 점검 통과 — 학습 시작 가능[/bold green]")
    else:
        CONSOLE.print("\n[bold red]❌ 일부 점검 실패 — Blocker 해결 필요[/bold red]")


def get_default_report_path() -> Path:
    """OS별 임시 디렉토리 — Windows는 %TEMP%, Linux/Mac은 /tmp"""
    import tempfile
    return Path(tempfile.gettempdir()) / "flight_mind_env.json"


def save_report(report: EnvReport, path: Path | None = None) -> None:
    if path is None:
        path = get_default_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    CONSOLE.print(f"\n[dim]Report saved to {path}[/dim]")


if __name__ == "__main__":
    report = diagnose()
    print_report(report)
    save_report(report)

    sys.exit(0 if report.overall_ok else 1)
