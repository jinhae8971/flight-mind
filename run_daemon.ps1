# Flight-Mind Paper Trading Daemon — Windows PowerShell
#
# 영길님 PC에서 paper trading 데몬을 백그라운드로 가동.
#
# Usage:
#   .\run_daemon.ps1                         # paper 모드 (기본)
#   .\run_daemon.ps1 -Mode testnet           # Binance Testnet
#   .\run_daemon.ps1 -Mode live              # 실거래 (3중 게이트)
#   .\run_daemon.ps1 -Background             # 백그라운드 실행 (PowerShell Job)
#   .\run_daemon.ps1 -Once                   # 단일 사이클 (테스트용)
#
# 환경 변수 (Telegram 알림):
#   $env:TELEGRAM_BOT_TOKEN = "..."
#   $env:TELEGRAM_CHAT_ID   = "..."
#
# Vault (testnet/live):
#   $env:VAULT_PASSPHRASE = "..."

param(
    [ValidateSet("paper", "testnet", "live")]
    [string]$Mode = "paper",

    [string[]]$Symbols = @(),

    [int]$CycleSeconds = 300,

    [switch]$Once,

    [switch]$Background    # PowerShell Job으로 분리 실행
)

$ErrorActionPreference = "Stop"
$VenvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "venv not found at $VenvPython. Run setup_and_train.ps1 first."
    exit 1
}

# Pre-flight checks
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Flight-Mind Paper Trading Daemon" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

Write-Host "Mode:           $Mode"
Write-Host "Cycle interval: $($CycleSeconds)s"

if ($Symbols.Count -gt 0) {
    Write-Host "Symbols:        $($Symbols -join ', ')"
}

# Telegram check
if ($env:TELEGRAM_BOT_TOKEN -and $env:TELEGRAM_CHAT_ID) {
    Write-Host "Telegram:       [OK] enabled" -ForegroundColor Green
} else {
    Write-Host "Telegram:       [INFO] disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)" -ForegroundColor Yellow
}

# Vault check (testnet/live)
if ($Mode -ne "paper") {
    if (-not $env:VAULT_PASSPHRASE) {
        Write-Error "VAULT_PASSPHRASE 필요 (testnet/live 모드). PowerShell:`n  `$env:VAULT_PASSPHRASE = 'your_password'"
        exit 1
    }
    Write-Host "Vault:          [OK] passphrase set" -ForegroundColor Green
}

# Live mode 추가 게이트
if ($Mode -eq "live") {
    if ($env:FLIGHT_MIND_LIVE -ne "1") {
        Write-Host ""
        Write-Host "[BLOCKED] Live mode requires:" -ForegroundColor Red
        Write-Host "  `$env:FLIGHT_MIND_LIVE = '1'"
        Write-Host ""
        Write-Host "또한 데몬 시작 시 인터랙티브 'I UNDERSTAND' 확인이 필요합니다."
        exit 1
    }
    Write-Host "Live Mode:      [WARNING] using REAL CAPITAL" -ForegroundColor Red
}

Write-Host ""

# Build args
$args = @(
    "-m", "flight_mind.daemon.paper_worker",
    "--mode", $Mode,
    "--cycle-s", $CycleSeconds
)

if ($Once) { $args += "--once" }
if ($Symbols.Count -gt 0) { $args += "--symbols"; $args += $Symbols }


# =============================================================================
# Execution
# =============================================================================
if ($Background) {
    Write-Host "Starting in background as PowerShell Job..." -ForegroundColor Cyan
    Write-Host ""

    $jobName = "FlightMindDaemon_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    $job = Start-Job -Name $jobName -ScriptBlock {
        param($pythonExe, $args)
        & $pythonExe @args
    } -ArgumentList $VenvPython, $args

    Write-Host "Job started: $jobName" -ForegroundColor Green
    Write-Host "Job ID: $($job.Id)"
    Write-Host ""
    Write-Host "View output:    Receive-Job -Id $($job.Id) -Keep"
    Write-Host "Stop job:       Stop-Job -Id $($job.Id); Remove-Job -Id $($job.Id)"
    Write-Host "List jobs:      Get-Job"
    Write-Host ""
    Write-Host "Status:"
    Write-Host "  python scripts\risk_status.py"
}
else {
    Write-Host "Running foreground (Ctrl+C to stop)..." -ForegroundColor Cyan
    Write-Host ""
    & $VenvPython @args
}
