# PrimeTerminal ML

XGBoost-based trade quality classifier for the PrimeTerminal trading bot.

## Architecture

```
C# Backtester ──► training_data.csv ──► train.py (XGBoost) ──► model.onnx
                                                                    │
                                                                    ▼
                                                              Go bot (alsatrobot)
                                                              onnxruntime-go
                                                              Predict P(win)
```

## Workflow

1. **Data Export** (`alsatrobot-test --export-training`)
   - 487 USDT-M perpetual symbols × 30 days
   - Every candle where score ≥ 1.0 (LONG candidate) or ≤ -1.0 (SHORT candidate)
   - Features: 20 indicators + time context
   - Label: forward-simulate trade, did TP or SL hit first?

2. **Training** (`python train.py`)
   - Time-based train/test split (last 7 days = test)
   - XGBoost binary classifier (win=1 / lose=0)
   - Walk-forward cross-validation
   - Export to ONNX for Go runtime

3. **Inference** (Go bot `strategy.go`)
   - Rule-based filters pass → ML predict
   - P(win) ≥ 0.55 → enter trade

4. **Retraining** (weekly cron)
   - Concept drift compensation
   - Hot-reload model.onnx

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

## Train

```bash
# After data export from C# backtest
python train.py --input data/training_data.csv --output models/model_v1.onnx
```

## Repos

- [primeterminal-engine](https://github.com/ksksertac/primeterminal-engine) (Go bot)
- [primeterminal-panel](https://github.com/ksksertac/primeterminal-panel) (UI)
- [alsatrobot-test](https://github.com/ksksertac/alsatrobot-test) (C# backtest)
