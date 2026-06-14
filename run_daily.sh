#!/bin/bash
# PrimeTerminal Daily ML Pipeline
#
# Bu script her gün 1 kez çalışır:
#   1. C# --export-training        (incremental CSV append)
#   2. Python train.py              (LightGBM model.json üret)
#   3. Atomic swap                  (engine fsnotify ile yakalar)
#   4. 24 saat uyu + exit           (Docker restart=always tekrar başlatır)
#
# NOT: Redis güncelleme (--service modu) AYRI bir container'da çalışır.
# Bu script sadece ML pipeline'a odaklanır.
#
# Ortam:
#   ML_DATA_PATH   = /ml-shared/data/training_data.csv
#   ML_MODEL_PATH  = /ml-shared/models/model.json
#   ML_TRAIN_ENABLED = 1 (default; 0 ise sadece backtest)
#   DAILY_SLEEP_SECONDS = 86400 (default 24 saat)

# Eğitim crash olursa sleep 24h'a geç (restart loop önle)
set -uo pipefail

DATA_DIR="$(dirname "${ML_DATA_PATH:-/ml-shared/data/training_data.csv}")"
MODEL_DIR="$(dirname "${ML_MODEL_PATH:-/ml-shared/models/model.json}")"
ARCHIVE_DIR="${MODEL_DIR}/archive"
MODEL_PATH="${ML_MODEL_PATH:-/ml-shared/models/model.json}"
MODEL_TMP="${MODEL_PATH}.new"
SLEEP_SECS="${DAILY_SLEEP_SECONDS:-86400}"

mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${ARCHIVE_DIR}"

log() { echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [ml-pipeline] $*"; }

log "=========================================="
log "Daily ML Pipeline başlıyor"
log "Data:  ${ML_DATA_PATH:-default}"
log "Model: ${MODEL_PATH}"
log "=========================================="

# ── 1. C# --export-training (incremental CSV) ──
log "[1/3] C# --export-training (incremental)"
cd /app
if dotnet alsatrobot-test.dll --export-training; then
    log "✓ CSV export tamam"
    EXPORT_OK=1
else
    log "⚠ CSV export hatası — eğitim atlanacak, 24h uyu"
    EXPORT_OK=0
fi

# ── 2. Python LightGBM eğitim (sadece export başarılıysa) ──
if [ "${EXPORT_OK}" = "1" ] && [ "${ML_TRAIN_ENABLED:-1}" = "1" ]; then
    log "[2/3] Python train.py (LightGBM)"
    cd /ml
    if python3 train.py \
        --input "${ML_DATA_PATH:-/ml-shared/data/training_data.csv}" \
        --output "${MODEL_TMP}" \
        --threshold 0.50 \
        --test-days 7; then
        TRAIN_OK=1
    else
        TRAIN_OK=0
        log "⚠ Eğitim başarısız — yeni model üretilmedi, eski model korunuyor"
    fi

    # ── 3. Atomic swap (sadece eğitim başarılıysa) ──
    if [ "${TRAIN_OK}" = "1" ] && [ -f "${MODEL_TMP}" ]; then
        log "[3/3] Atomic swap (engine fsnotify ile yakalar)"
        if [ -f "${MODEL_PATH}" ]; then
            ARCHIVE_NAME="model_$(date -u +%Y%m%d_%H%M).json"
            cp "${MODEL_PATH}" "${ARCHIVE_DIR}/${ARCHIVE_NAME}"
            log "Eski model arşivlendi: ${ARCHIVE_DIR}/${ARCHIVE_NAME}"
        fi
        mv "${MODEL_TMP}" "${MODEL_PATH}"
        log "✓ Yeni model aktif: ${MODEL_PATH}"

        find "${ARCHIVE_DIR}" -name "model_*.json" -type f -mtime +14 -delete 2>/dev/null || true
    else
        log "[3/3] Eğitim başarısız ya da model dosyası yok — swap atlandı"
        rm -f "${MODEL_TMP}" 2>/dev/null || true
    fi
else
    log "[2/3] ML_TRAIN_ENABLED=0 — eğitim atlandı"
fi

log "✓ Pipeline tamamlandı. ${SLEEP_SECS} saniye uyuyacak."
log "(Docker restart policy: bir sonraki gün yeniden başlatılır)"

sleep "${SLEEP_SECS}"

# Sleep bitince exit 0 ile çık (restart loop önle — restart: always tetiklenir)
exit 0
