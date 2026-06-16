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
    # 2026-06-16: Bellek (OOM) — feature'lari float32, outcome int8 oku.
    # Cift-yonlu veri 5M+ satir; float64 default ~2GB container'da OOM oluyordu.
    _dtypes = {f: "float32" for f in FEATURES}
    _dtypes["outcome"] = "int8"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], dtype=_dtypes)
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

    # 2026-06-16: Bellek — train cok buyukse ornekle (bidirectional + 90 gun -> 15M+ satir OOM).
    # 2M, tek-yonlu calisan boyuta yakin; rastgele ornek iki yonu de korur. Test tam kalir (durust AUC).
    MAX_TRAIN = 2_000_000
    if len(train_df) > MAX_TRAIN:
        print(f"[Data] Train {len(train_df):,} -> {MAX_TRAIN:,} ornekleniyor (bellek/OOM önleme)")
        train_df = train_df.sample(n=MAX_TRAIN, random_state=42)

    return (
        train_df[FEATURES].values, train_df["outcome"].values,
        test_df[FEATURES].values,  test_df["outcome"].values,
    )


def train(X_train, y_train, X_test, y_test):
    # ── 2026-06-15: Zaman feature'larini devre disi birak (overfit kaynagi) ──
    # hour_cos/hour_sin/day_of_week gain'de en ust siralardaydi = donemin seans paternini
    # ezberliyor; 1. agactan sonra test AUC dusuyordu (overfit). Parite icin kolon sayisi
    # 20'de KALIR, egitimde sifirlanir -> model bu kolonlara split atamaz, Go MLFeatures degismez.
    time_idx = [FEATURES.index(f) for f in ("hour_sin", "hour_cos", "day_of_week")]
    X_train = X_train.copy(); X_train[:, time_idx] = 0.0
    X_test  = X_test.copy();  X_test[:, time_idx]  = 0.0

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    test_set  = lgb.Dataset(X_test,  label=y_test,  feature_name=FEATURES, reference=train_set)

    # 2026-06-15: Class imbalance — explicit scale_pos_weight (neg/pos orani).
    # is_unbalance yerine: tahminleri yukari yayar, threshold anlamli hale gelir.
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = int((y_train == 0).sum())
    spw   = n_neg / n_pos
    print(f"[Train] scale_pos_weight = {spw:.2f} (neg={n_neg:,} / pos={n_pos:,})")

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "scale_pos_weight": spw,
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
    for thr in [round(0.30 + i * 0.02, 2) for i in range(21)]:  # 0.30→0.70, ince adim (gercek esigi say ile sec)
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
