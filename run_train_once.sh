#!/bin/bash
# Tek seferlik ML eğitimi (yerel): 1 yıl veri export + LightGBM train. Uyku YOK, çıkınca durur.
#   ML_DATA_PATH       = /out/data/training_data.csv
#   ML_MODEL_PATH      = /out/models/model.json
#   EXPORT_DAYS        = 365 (ilk export penceresi)
#   ML_SLIDING_DAYS    = 365 (trim penceresi — yoksa 90'a kırpar)
#   ML_EXPORT_WHITELIST= 1   (60 sembol; 0/boş = tüm perpetual'lar)
#   DATACACHE_PATH     = /out/datacache (inen mumlar burada kalıcı)
set -uo pipefail

DATA_PATH="${ML_DATA_PATH:-/out/data/training_data.csv}"
MODEL_PATH="${ML_MODEL_PATH:-/out/models/model.json}"
EXPORT_DAYS="${EXPORT_DAYS:-365}"
mkdir -p "$(dirname "$DATA_PATH")" "$(dirname "$MODEL_PATH")"

log() { echo "[$(date -u +'%Y-%m-%d %H:%M:%S')] [train-once] $*"; }

log "=========================================="
log "TEK SEFERLİK EĞİTİM — ${EXPORT_DAYS} gün"
log "Data:  $DATA_PATH"
log "Model: $MODEL_PATH"
log "Sliding: ${ML_SLIDING_DAYS:-90}g | Whitelist: ${ML_EXPORT_WHITELIST:-0}"
log "=========================================="

# ── 1. Export (1 yıl) ──
log "[1/2] C# --export-training ${EXPORT_DAYS}"
cd /app
if ! dotnet alsatrobot-test.dll --export-training "${EXPORT_DAYS}"; then
    log "✗ Export hatası — çıkılıyor"
    exit 1
fi
ROWS=$(($(wc -l < "$DATA_PATH") - 1))
log "✓ Export tamam — ${ROWS} satır CSV"

# ── 2. Eğitim ──
log "[2/2] Python train.py (LightGBM)"
cd /ml
if ! python3 train.py --input "$DATA_PATH" --output "$MODEL_PATH" --threshold 0.55 --test-days 14; then
    log "✗ Eğitim hatası"
    exit 1
fi

log "✓ TAMAMLANDI — model: $MODEL_PATH"
ls -la "$(dirname "$MODEL_PATH")"
exit 0
