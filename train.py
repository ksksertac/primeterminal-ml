"""
PrimeTerminal ML — XGBoost Trade Quality Classifier

Eğitim:
    python train.py --input data/training_data.csv --output models/model_v1.onnx

Beklenen CSV şeması:
    rsi,atr_ratio,vol_mult,ema_slope,score,
    trend_sig,cvd_sig,oi_sig,fund_sig,htf_trend,
    vol_spike,price_range_30,price_range_5,
    up_run_5,down_run_5,reversal_flag,
    hour_sin,hour_cos,day_of_week,
    side,
    timestamp,outcome
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from onnxmltools.convert import convert_xgboost
from onnxconverter_common import FloatTensorType


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

    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["timestamp"] < cutoff]
    test_df  = df[df["timestamp"] >= cutoff]

    print(f"[Data] Toplam: {len(df):,} | Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"[Data] Train tarih: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"[Data] Test tarih:  {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
    print(f"[Data] Train win oranı: {train_df['outcome'].mean():.3f}")
    print(f"[Data] Test win oranı:  {test_df['outcome'].mean():.3f}")

    return (
        train_df[FEATURES].values, train_df["outcome"].values,
        test_df[FEATURES].values,  test_df["outcome"].values,
    )


def train(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20,
    )

    return model


def evaluate(model, X_test, y_test, threshold: float = 0.55):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)

    print(f"\n[Eval] Threshold: {threshold}")
    print(f"[Eval] Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"[Eval] AUC:       {roc_auc_score(y_test, proba):.4f}")
    print(f"\n[Confusion Matrix]\n{confusion_matrix(y_test, pred)}")
    print(f"\n[Classification Report]\n{classification_report(y_test, pred, digits=4)}")

    # Trade subset (sadece model "gir" diyenler)
    enter_mask = proba >= threshold
    if enter_mask.sum() > 0:
        wins = y_test[enter_mask].sum()
        total = enter_mask.sum()
        print(f"\n[Trade Selection] Model {total:,} trade önerdi, {wins:,} kazandı = %{wins/total*100:.1f} win rate")
        print(f"[Breakeven] %48 üstü → karlı")

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "trades_proposed": int(enter_mask.sum()),
        "win_rate_when_entered": float(wins / total) if enter_mask.sum() > 0 else 0.0,
    }


def feature_importance(model, output_dir: Path):
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]
    print("\n[Feature Importance]")
    for i in order:
        print(f"  {FEATURES[i]:<20} {importance[i]:.4f}")

    fi = {FEATURES[i]: float(importance[i]) for i in order}
    (output_dir / "feature_importance.json").write_text(json.dumps(fi, indent=2))


def export_onnx(model, output_path: Path):
    initial_type = [("input", FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())
    print(f"\n[ONNX] Model kaydedildi: {output_path}")
    print(f"[ONNX] Boyut: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="training_data.csv yolu")
    ap.add_argument("--output", default="models/model_v1.onnx")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--test-days", type=int, default=7)
    args = ap.parse_args()

    csv_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_and_split(csv_path, args.test_days)
    model = train(X_tr, y_tr, X_te, y_te)
    metrics = evaluate(model, X_te, y_te, args.threshold)
    feature_importance(model, out_path.parent)
    export_onnx(model, out_path)

    metrics_path = out_path.parent / (out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[Metrics] Kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main()
