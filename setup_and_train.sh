#!/usr/bin/env bash
# Flight-Mind Setup & Training — Bash (WSL2 / Linux / Mac)
#
# 영길님 PC에서 한 번에 실행 (WSL2 또는 Linux/Mac):
#   $ chmod +x setup_and_train.sh
#   $ ./setup_and_train.sh
#
# Windows 영길님 환경에서는 setup_and_train.ps1 사용 권장.

set -euo pipefail

VENV_PATH=".venv"
PYTHON_EXE="${VENV_PATH}/bin/python"

# Defaults
SKIP_SETUP=false
DIAGNOSE_ONLY=false
START_FROM=""
T2_BATCH=0
T2_EPOCHS=50
T4_BATCH=0
T4_EPOCHS=40
RESET=false


# =============================================================================
# Argument Parsing
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --skip-setup          venv 이미 있음
  --diagnose-only       환경 진단만
  --start-from STEP     특정 단계부터 재시작
  --t2-batch N          Tier 2 batch size (0=auto)
  --t2-epochs N         Tier 2 epochs (default 50)
  --t4-batch N          Tier 4 batch size (0=auto)
  --t4-epochs N         Tier 4 epochs (default 40)
  --reset               체크포인트 초기화
  -h, --help            이 도움말

Step IDs: diagnose, download_data, load_to_db, build_features,
          train_tier2, train_tier4, integrated_backtest
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)     SKIP_SETUP=true; shift ;;
        --diagnose-only)  DIAGNOSE_ONLY=true; shift ;;
        --start-from)     START_FROM="$2"; shift 2 ;;
        --t2-batch)       T2_BATCH="$2"; shift 2 ;;
        --t2-epochs)      T2_EPOCHS="$2"; shift 2 ;;
        --t4-batch)       T4_BATCH="$2"; shift 2 ;;
        --t4-epochs)      T4_EPOCHS="$2"; shift 2 ;;
        --reset)          RESET=true; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "Unknown: $1"; usage; exit 1 ;;
    esac
done


# =============================================================================
# Helpers
# =============================================================================
section() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════"
}


# =============================================================================
# Step 1: Setup
# =============================================================================
if [[ "$SKIP_SETUP" == false ]]; then
    section "Step 1: Python venv + Dependencies"

    PYVER=$(python3 --version 2>&1 || echo "Python not found")
    echo "System Python: $PYVER"

    if [[ ! -d "$VENV_PATH" ]]; then
        echo "Creating venv at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
    else
        echo "venv already exists — reusing"
    fi

    echo ""
    echo "Upgrading pip..."
    "$PYTHON_EXE" -m pip install --upgrade pip --quiet

    echo "Installing core dependencies..."
    "$PYTHON_EXE" -m pip install -e ".[dev]" --quiet

    echo ""
    echo "Installing PyTorch (CUDA 12.1)..."
    if [[ "$(uname)" == "Darwin" ]]; then
        # Mac: MPS backend
        "$PYTHON_EXE" -m pip install torch torchvision --quiet
    else
        "$PYTHON_EXE" -m pip install --upgrade \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121 --quiet \
            || "$PYTHON_EXE" -m pip install torch torchvision --quiet
    fi

    echo ""
    echo "✓ Setup complete"
fi


# =============================================================================
# Step 2: Environment Diagnostic
# =============================================================================
section "Step 2: Environment Diagnostic"

"$PYTHON_EXE" scripts/diagnose_env.py
DIAG_RC=$?

if [[ $DIAG_RC -ne 0 ]]; then
    echo ""
    echo "❌ Environment check failed. Fix blockers and retry."
    exit 1
fi

if [[ "$DIAGNOSE_ONLY" == true ]]; then
    echo ""
    echo "[diagnose-only mode] — exiting"
    exit 0
fi


# =============================================================================
# Step 3: Auto batch size
# =============================================================================
ENV_JSON="${TMPDIR:-/tmp}/flight_mind_env.json"
if [[ -f "$ENV_JSON" ]]; then
    if [[ "$T2_BATCH" -eq 0 ]]; then
        T2_BATCH=$("$PYTHON_EXE" -c "import json; print(json.load(open('$ENV_JSON'))['recommended_t2_batch'])")
    fi
    if [[ "$T4_BATCH" -eq 0 ]]; then
        T4_BATCH=$("$PYTHON_EXE" -c "import json; print(json.load(open('$ENV_JSON'))['recommended_t4_batch'])")
    fi
    echo ""
    echo "Auto-detected batch sizes:"
    echo "  Tier 2: $T2_BATCH"
    echo "  Tier 4: $T4_BATCH"
else
    [[ "$T2_BATCH" -eq 0 ]] && T2_BATCH=32
    [[ "$T4_BATCH" -eq 0 ]] && T4_BATCH=64
fi


# =============================================================================
# Step 4: Master Pipeline
# =============================================================================
section "Step 4: Master Training Pipeline"

# Telegram check
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo "[INFO] TELEGRAM_BOT_TOKEN not set — 알림 비활성화"
else
    echo "[OK] Telegram notifications enabled"
fi

ARGS=(
    "scripts/run_full_training.py"
    "--t2-epochs" "$T2_EPOCHS"
    "--t2-batch"  "$T2_BATCH"
    "--t4-epochs" "$T4_EPOCHS"
    "--t4-batch"  "$T4_BATCH"
)

[[ -n "$START_FROM" ]] && ARGS+=("--start-from" "$START_FROM")
[[ "$RESET" == true ]] && ARGS+=("--reset")

echo ""
echo "Command: $PYTHON_EXE ${ARGS[*]}"
echo ""
echo "예상 시간 (RTX 4090 기준):"
echo "  - 데이터 다운로드 + 적재  : 1~2시간"
echo "  - 피처 빌드               : 30분"
echo "  - Tier 2 학습             : 8시간"
echo "  - Tier 4 학습             : 1~2시간"
echo "  - 통합 백테스트           : 30분"
echo "  ──────────────────────────────────"
echo "  총 예상 시간              : 약 12~15시간"
echo ""

"$PYTHON_EXE" "${ARGS[@]}"
EXIT_CODE=$?

section "Pipeline Result"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Flight-Mind 학습 완료!"
    echo ""
    echo "다음 단계:"
    echo "  1. 결과 확인: data/integrated_backtest_results.json"
    echo "  2. 모델 체크포인트: data/models/"
else
    echo "❌ Pipeline failed (exit code: $EXIT_CODE)"
    echo ""
    echo "재시작:"
    echo "  ./setup_and_train.sh --skip-setup --start-from <step_id>"
fi

exit $EXIT_CODE
