# Flight-Mind Setup & Training — Windows PowerShell
#
# 영길님 PC에서 한 번에 실행:
#   PS> .\setup_and_train.ps1
#
# 단계:
#   1. Python venv 생성 + 의존성 설치
#   2. PyTorch CUDA 12.1 설치
#   3. 환경 진단
#   4. 마스터 학습 파이프라인 실행
#
# 환경 변수 (Telegram 알림용):
#   $env:TELEGRAM_BOT_TOKEN = "your_bot_token"
#   $env:TELEGRAM_CHAT_ID   = "your_chat_id"

param(
    [switch]$SkipSetup,                 # venv 이미 있음
    [switch]$DiagnoseOnly,
    [string]$StartFrom = "",            # 특정 단계부터 재시작
    [int]$T2Batch = 0,                  # 0 = auto (GPU 자동 감지)
    [int]$T2Epochs = 50,
    [int]$T4Batch = 0,                  # 0 = auto
    [int]$T4Epochs = 40,
    [switch]$Reset                      # 체크포인트 초기화
)

$ErrorActionPreference = "Stop"
$VenvPath = ".\.venv"
$PythonExe = "$VenvPath\Scripts\python.exe"

function Section($title) {
    Write-Host ""
    Write-Host ("━" * 60) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("━" * 60) -ForegroundColor Cyan
}


# =============================================================================
# Step 1: Setup
# =============================================================================
if (-not $SkipSetup) {
    Section "Step 1: Python venv + Dependencies"

    # Check Python availability
    $pyver = & python --version 2>&1
    Write-Host "System Python: $pyver"

    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating venv at $VenvPath..."
        & python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error "venv creation failed. 'python -m venv' available?"
            exit 1
        }
    } else {
        Write-Host "venv already exists — reusing"
    }

    Write-Host ""
    Write-Host "Upgrading pip..."
    & $PythonExe -m pip install --upgrade pip --quiet

    Write-Host "Installing core dependencies..."
    & $PythonExe -m pip install -e ".[dev]" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Core install failed. Check pyproject.toml"
        exit 1
    }

    Write-Host ""
    Write-Host "Installing PyTorch (CUDA 12.1)..."
    Write-Host "[NOTE] CPU-only PyTorch가 이미 설치된 경우 재설치 — 5분 정도 소요" -ForegroundColor Yellow
    & $PythonExe -m pip install --upgrade `
        torch torchvision `
        --index-url https://download.pytorch.org/whl/cu121 `
        --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "PyTorch CUDA install failed — falling back to CPU"
        & $PythonExe -m pip install torch torchvision --quiet
    }

    Write-Host ""
    Write-Host "[green]✓ Setup complete[/green]" -ForegroundColor Green
}


# =============================================================================
# Step 2: Environment Diagnostic
# =============================================================================
Section "Step 2: Environment Diagnostic"

& $PythonExe scripts\diagnose_env.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment check failed. Fix blockers and retry."
    Write-Host ""
    Write-Host "Common fixes:"
    Write-Host "  - PyTorch CUDA 미작동: NVIDIA driver 업데이트 (≥ 525.x)"
    Write-Host "  - 디스크 부족: 다른 드라이브로 레포 이동"
    exit 1
}

if ($DiagnoseOnly) {
    Write-Host ""
    Write-Host "[diagnose-only mode] — exiting" -ForegroundColor Yellow
    exit 0
}


# =============================================================================
# Step 3: Auto batch size from diagnostic report
# =============================================================================
$envJsonPath = Join-Path $env:TEMP "flight_mind_env.json"
if (-not (Test-Path $envJsonPath)) {
    $envJsonPath = "/tmp/flight_mind_env.json"   # WSL2 fallback
}
$envJson = Get-Content $envJsonPath -Raw -ErrorAction SilentlyContinue

if ($envJson) {
    $envData = $envJson | ConvertFrom-Json
    if ($T2Batch -eq 0) { $T2Batch = $envData.recommended_t2_batch }
    if ($T4Batch -eq 0) { $T4Batch = $envData.recommended_t4_batch }
    Write-Host ""
    Write-Host "Auto-detected batch sizes:" -ForegroundColor Cyan
    Write-Host "  Tier 2: $T2Batch"
    Write-Host "  Tier 4: $T4Batch"
} else {
    if ($T2Batch -eq 0) { $T2Batch = 32 }
    if ($T4Batch -eq 0) { $T4Batch = 64 }
    Write-Warning "Diagnostic report not found — using default batch sizes"
}


# =============================================================================
# Step 4: Master Training Pipeline
# =============================================================================
Section "Step 4: Master Training Pipeline"

# Telegram 알림 환경변수 확인
if (-not $env:TELEGRAM_BOT_TOKEN) {
    Write-Host "[INFO] TELEGRAM_BOT_TOKEN not set — 알림 비활성화" -ForegroundColor Yellow
    Write-Host "  활성화: `$env:TELEGRAM_BOT_TOKEN = 'YOUR_TOKEN'"
} else {
    Write-Host "[OK] Telegram notifications enabled" -ForegroundColor Green
}

$args = @(
    "scripts\run_full_training.py",
    "--t2-epochs", $T2Epochs,
    "--t2-batch", $T2Batch,
    "--t4-epochs", $T4Epochs,
    "--t4-batch", $T4Batch
)

if ($StartFrom) { $args += "--start-from", $StartFrom }
if ($Reset) { $args += "--reset" }

Write-Host ""
Write-Host "Command: $PythonExe $($args -join ' ')" -ForegroundColor Cyan
Write-Host ""
Write-Host "예상 시간:"
Write-Host "  - 데이터 다운로드 + 적재  : 1~2시간"
Write-Host "  - 피처 빌드               : 30분"
Write-Host "  - Tier 2 학습 (RTX 4090)  : 8시간"
Write-Host "  - Tier 4 학습             : 1~2시간"
Write-Host "  - 통합 백테스트           : 30분"
Write-Host "  ──────────────────────────────────"
Write-Host "  총 예상 시간              : 약 12~15시간" -ForegroundColor Yellow
Write-Host ""
Write-Host "[TIP] PowerShell 창 닫지 마세요. 백그라운드 실행하려면 Job 사용:"
Write-Host "  Start-Job -ScriptBlock { ./setup_and_train.ps1 -SkipSetup }"
Write-Host ""

& $PythonExe @args
$exitCode = $LASTEXITCODE

Section "Pipeline Result"

if ($exitCode -eq 0) {
    Write-Host "✅ Flight-Mind 학습 완료!" -ForegroundColor Green
    Write-Host ""
    Write-Host "다음 단계:"
    Write-Host "  1. 백테스트 결과 확인: data\integrated_backtest_results.json"
    Write-Host "  2. 모델 체크포인트:    data\models\"
    Write-Host "  3. Paper trading 시작 (별도 가이드 참조)"
} else {
    Write-Host "❌ Pipeline failed (exit code: $exitCode)" -ForegroundColor Red
    Write-Host ""
    Write-Host "재시작:"
    Write-Host "  .\setup_and_train.ps1 -SkipSetup -StartFrom <step_id>"
    Write-Host ""
    Write-Host "단계 ID: diagnose, download_data, load_to_db, build_features,"
    Write-Host "         train_tier2, train_tier4, integrated_backtest"
}

exit $exitCode
