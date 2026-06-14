#!/bin/bash
# PrimeTerminal Daily ML Pipeline
#
# Bu script Docker container içinde günde 1 kez çalışır:
#   1. C# backtest --service modu  (Redis update — mevcut davranış)
#   2. C# --export-training         (incremental CSV append)
#   3. Python train.py              (LightGBM model.json üret)
#   4. Atomic swap                  (engine fsnotify ile yakalar)
#
# Ortam:
#   ML_DATA_PATH   = /ml-shared/data/training_data.csv
#   ML_MODEL_PATH  = /ml-shared/models/model.json
#   ML_TRAIN_ENABLED = 1 (default; 0 ise sadece backtest)

set -euo pipefail

# ── Yollar ──
DATA_DIR="$(dirname "${ML_DATA_PATH:-/ml-shared/data/training_data.csv}")"
MODEL_DIR="$(dirname "${ML_MODEL_PATH:-/ml-shared/models/model.json}")"
ARCHIVE_DIR="${MODEL_DIR}/archive"
MODEL_PATH="${ML_MODEL_PATH:-/ml-shared/models/model.json}"
MODEL_TMP="${MODEL_PATH}.new"

mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${ARCHIVE_DIR}"

log() { echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [run_daily] $*"; }

log "=========================================="
log "Daily ML Pipeline başlıyor"
log "Data:  ${ML_DATA_PATH:-default}"
log "Model: ${MODEL_PATH}"
log "=========================================="

# ── 1. C# Backtest --service modu (Redis update — değişmedi) ──
log "[1/4] C# backtest --service (Redis update)"
cd /app
./alsatrobot-test --service &
SERVICE_PID=$!
log "Service PID: $SERVICE_PID — arka planda Redis güncelliyor"

# ── 2. C# --export-training (incremental CSV) ──
log "[2/4] C# --export-training (incremental)"
./alsatrobot-test --export-training
log "✓ CSV export tamam"

# ── 3. Python LightGBM eğitim ──
if [ "${ML_TRAIN_ENABLED:-1}" = "1" ]; then
    log "[3/4] Python train.py (LightGBM)"
    cd /ml
    python3 train.py \
        --input "${ML_DATA_PATH:-/ml-shared/data/training_data.csv}" \
        --output "${MODEL_TMP}" \
        --threshold 0.55 \
        --test-days 7

    # ── 4. Atomic swap ──
    log "[4/4] Atomic swap (engine fsnotify ile yakalar)"
    if [ -f "${MODEL_PATH}" ]; then
        ARCHIVE_NAME="model_$(date -u +%Y%m%d_%H%M).json"
        cp "${MODEL_PATH}" "${ARCHIVE_DIR}/${ARCHIVE_NAME}"
        log "Eski model arşivlendi: ${ARCHIVE_DIR}/${ARCHIVE_NAME}"
    fi
    mv "${MODEL_TMP}" "${MODEL_PATH}"
    log "✓ Yeni model aktif: ${MODEL_PATH}"

    # Eski arşiv temizle (son 14 günlük tut)
    find "${ARCHIVE_DIR}" -name "model_*.json" -type f -mtime +14 -delete 2>/dev/null || true
else
    log "[3/4] ML_TRAIN_ENABLED=0 — eğitim atlandı"
fi

# Service hâlâ çalışıyorsa bitmesini bekle (Redis update sürebilir)
log "Service'in bitmesini bekle (PID $SERVICE_PID)"
wait $SERVICE_PID || true

log "Daily pipeline tamamlandı."
log "Sonraki çalışma: 24 saat sonra (Docker restart policy / cron)"
