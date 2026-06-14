"""
PrimeTerminal ML — LightGBM Trade Quality Classifier

Eğitim:
    python train.py --input data/training_data.csv --output models/model.json

Beklenen CSV (TrainingExporter.cs ile uyumlu):
    timestamp,symbol,
    rsi,atr_ratio,vol_mult,ema_slope,score,
    trend_sig,cvd_sig,oi_sig,fund_sig,htf_trend,
    vol_spike,price_range_30,price_range_5,
    up_run_5,down_run_5,reversal_flag,
    hour_sin,hour_cos,day_of_week,
    side,outcome

Çıktı: models/model.json (LightGBM JSON) + model_metrics.json + feature_importance.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)


# Go bot'taki MLFeatures ile birebir aynı sıra olmalı!
FEATURES = [
    "rsi", "atr_ratio", "vol_mult", "ema_slope", "score",
    "trend_sig", "cvd_sig", "oi_sig", "fund_sig", "htf_trend",
    "vol_spike", "price_range_30", "price_range_5",
    "up_run_5", "down_run_5", "reversal_flag",
    "hour_sin", "hour_cos", "day_of_week",
    "side",
]


def load_and_split(csv_path: Path, test_days: int = 7):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Eksik feature varsa erken hata
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"CSV'de eksik feature'lar: {missing}")
    if "outcome" not in df.columns:
        raise ValueError("CSV'de 'outcome' kolonu yok (label)")

    # Yetersiz veri kontrolü
    if len(df) < 5000:
        raise ValueError(f"Cok az veri ({len(df)} satir) — train.py atlanmali, CSV birikiminde yarin tekrar dene")

    days_span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    print(f"[Data] Toplam: {len(df):,} | Veri ara: {days_span:.1f} gun")
    print(f"[Data] Tarih: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # ── Strateji seçimi ──
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["timestamp"] < cutoff]
    test_df  = df[df["timestamp"] >= cutoff]

    # Time-based split başarısız → random fallback
    if len(train_df) < 1000 or len(test_df) < 500:
        print(f"[Data] Time-based split yetersiz (train={len(train_df)}, test={len(test_df)})")
        print(f"[Data] FALLBACK: random 80/20 split kullanılıyor (gunluk akumulasyon icin)")
        # Seed = sabit (reproducible)
        df_shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split = int(len(df_shuf) * 0.80)
        train_df = df_shuf.iloc[:split]
        test_df  = df_shuf.iloc[split:]

    print(f"[Data] Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"[Data] Train win orani: {train_df['outcome'].mean():.3f}")
    print(f"[Data] Test win orani:  {test_df['outcome'].mean():.3f}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train veya test boş — veri yetersiz")

    return (
        train_df[FEATURES].values, train_df["outcome"].values,
        test_df[FEATURES].values,  test_df["outcome"].values,
    )


def train(X_train, y_train, X_test, y_test):
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    test_set  = lgb.Dataset(X_test,  label=y_test,  feature_name=FEATURES, reference=train_set)

    # 2026-06-14: Class imbalance düzeltmesi
    # İlk eğitimde win rate %11 → model "hep 0" tahmin etti (accuracy %88 ama 0 trade).
    # is_unbalance=True → LightGBM pos/neg weight otomatik dengeler.
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "is_unbalance": True,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 0,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set, test_set],
        valid_names=["train", "test"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=25),
        ],
    )
    return booster


def evaluate(booster, X_test, y_test, threshold: float = 0.50):
    proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred  = (proba >= threshold).astype(int)

    print(f"\n[Eval] Threshold: {threshold}")
    print(f"[Eval] Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"[Eval] AUC:       {roc_auc_score(y_test, proba):.4f}")
    print(f"\n[Probability Distribution]")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{pct:>2}: {np.percentile(proba, pct):.4f}")
    print(f"  Max: {proba.max():.4f}")

    print(f"\n[Threshold Sweep — en iyi trade-off bul]")
    print(f"  {'thr':>5} | {'trades':>7} | {'win%':>6} | {'expectancy':>10}")
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = proba >= thr
        if mask.sum() < 100:
            continue
        wr = y_test[mask].mean()
        # Expectancy: TP $2 - SL $0.78 (10x, $20 margin, after fee)
        # Hedef: >0 olmalı (karlı)
        expectancy = wr * 2.0 - (1 - wr) * 0.78
        print(f"  {thr:>5.2f} | {int(mask.sum()):>7} | {wr*100:>5.1f}% | ${expectancy:>+8.3f}")

    print(f"\n[Confusion Matrix]\n{confusion_matrix(y_test, pred)}")
    print(f"\n[Classification Report]\n{classification_report(y_test, pred, digits=4, zero_division=0)}")

    enter_mask = proba >= threshold
    win_rate = 0.0
    if enter_mask.sum() > 0:
        wins = int(y_test[enter_mask].sum())
        total = int(enter_mask.sum())
        win_rate = wins / total
        print(f"\n[Trade Selection] Model {total:,} trade önerdi, {wins:,} kazandı = %{win_rate*100:.1f} win rate")
        print(f"[Breakeven] %48 üstü → karlı")
    else:
        print(f"\n[Trade Selection] Model HIÇ trade önermedi (threshold çok yüksek?)")

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "trades_proposed": int(enter_mask.sum()),
        "win_rate_when_entered": float(win_rate),
        "best_iteration": booster.best_iteration,
        "num_features": len(FEATURES),
    }


def feature_importance(booster, output_dir: Path):
    importance = booster.feature_importance(importance_type="gain")
    order = np.argsort(importance)[::-1]
    print("\n[Feature Importance — gain]")
    fi = {}
    for i in order:
        print(f"  {FEATURES[i]:<20} {importance[i]:.2f}")
        fi[FEATURES[i]] = float(importance[i])
    (output_dir / "feature_importance.json").write_text(json.dumps(fi, indent=2))


def save_model(booster, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # leaves için: native LightGBM text format (model_str). JSON adıyla kaydedeceğiz
    # ama içerik LightGBM text. leaves.LGEnsembleFromFile bu formatı okur.
    booster.save_model(str(output_path), num_iteration=booster.best_iteration)
    print(f"\n[Model] Kaydedildi: {output_path}")
    print(f"[Model] Boyut: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"[Model] Format: LightGBM native (Go leaves.LGEnsembleFromFile ile uyumlu)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="training_data.csv yolu")
    ap.add_argument("--output", default="models/model.json", help="çıktı model dosyası")
    # 2026-06-14: threshold 0.55 → 0.50. is_unbalance=True ile dağılım daha geniş olur.
    # Engine tarafında ML_THRESHOLD env değişkeni override edebilir.
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--test-days", type=int, default=7)
    args = ap.parse_args()

    csv_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_and_split(csv_path, args.test_days)
    booster = train(X_tr, y_tr, X_te, y_te)
    metrics = evaluate(booster, X_te, y_te, args.threshold)
    feature_importance(booster, out_path.parent)
    save_model(booster, out_path)

    metrics_path = out_path.parent / (out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[Metrics] Kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main()
