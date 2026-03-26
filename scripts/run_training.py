"""
Feature engineering + regression model training (Week 2)

Pipeline:
  1. Load 10s data, build signal frame, build features
  2. Attach regression targets (z_future_30/60/180)
  3. Walk-forward CV on training set
  4. Two-stage final training (early-stop on val, refit on train+val)
  5. Evaluate on test set
  6. Save model artefact to outputs/models/lgbm_regression_h{horizon}.pkl
  7. Save plot_model_predictions to outputs/plots/

Run from repo root:
    python3 scripts/run_training.py [--horizon 60]
"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import compute_future_zscore_targets, split_by_date
from triangulation.model import (
    get_feature_cols,
    train_model,
    retrain_model,
    evaluate_model,
    walk_forward_folds,
)
from triangulation.plots import plot_model_predictions

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR  = ROOT / "outputs" / "plots"

TRAIN_START = "2024-03-01"
TRAIN_END   = "2024-12-31"
VAL_END     = "2025-06-30"


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regression model for z-score forecasting")
    parser.add_argument("--horizon",    type=int,   default=60,
                        help="Target horizon in bars (default: 60 = 10 min at 10s)")
    parser.add_argument("--data-dir",   type=str,   default=str(DATA_DIR),
                        help="Directory containing .gmr tick files")
    parser.add_argument("--output-dir", type=str,   default=str(ROOT / "outputs"),
                        help="Root output directory")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir  = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"z_future_{args.horizon}"

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    divider("Loading data (10s bars)")
    eurusd = load_pair(data_dir, "EURUSD")
    audusd = load_pair(data_dir, "AUDUSD")
    euraud = load_pair(data_dir, "EURAUD")
    print(f"EURUSD: {len(eurusd):,}  AUDUSD: {len(audusd):,}  EURAUD: {len(euraud):,}")

    # -----------------------------------------------------------------------
    # 2. Signal frame + features + targets
    # -----------------------------------------------------------------------
    divider("Building signal frame + features + regression targets")
    sig  = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    del eurusd, audusd, euraud
    feat = build_feature_frame(sig)
    del sig
    feat = compute_future_zscore_targets(feat)
    print(f"Feature frame: {feat.shape}")
    print(f"Target '{target_col}' non-NaN: {feat[target_col].notna().sum():,}")

    feature_cols = get_feature_cols(feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # -----------------------------------------------------------------------
    # 3. Walk-forward CV (training period only)
    # -----------------------------------------------------------------------
    divider("Walk-forward validation (2024-03 → 2024-12)")
    folds = walk_forward_folds(
        feat,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        fold_months=2,
        oos_months=1,
        buffer_bars=60,
    )
    print(f"Folds: {len(folds)}")

    baseline_rmses, full_rmses = [], []
    baseline_daccs, full_daccs = [], []

    for fold in folds:
        tr  = fold["train_df"].dropna(subset=feature_cols + [target_col])
        oos = fold["oos_df"].dropna(subset=feature_cols + [target_col])
        if len(tr) < 500 or len(oos) < 100:
            continue

        X_tr  = tr[feature_cols].values
        y_tr  = tr[target_col].values
        X_oos = oos[feature_cols].values
        y_oos = oos[target_col].values

        split = int(len(X_tr) * 0.9)
        X_tr_, X_es = X_tr[:split], X_tr[split:]
        y_tr_, y_es = y_tr[:split], y_tr[split:]

        base_cols = ["residual", "zscore"]
        base_idx  = [feature_cols.index(c) for c in base_cols]
        m_base, _ = train_model(X_tr_[:, base_idx], y_tr_, X_es[:, base_idx], y_es)
        ev_base   = evaluate_model(m_base, X_oos[:, base_idx], y_oos)

        m_full, _ = train_model(X_tr_, y_tr_, X_es, y_es)
        ev_full   = evaluate_model(m_full, X_oos, y_oos)

        baseline_rmses.append(ev_base["rmse"])
        full_rmses.append(ev_full["rmse"])
        baseline_daccs.append(ev_base["directional_accuracy"])
        full_daccs.append(ev_full["directional_accuracy"])

        print(
            f"  Fold {fold['fold']}  train={len(tr):,}  oos={len(oos):,}"
            f"  ({oos.index[0].date()}→{oos.index[-1].date()})"
            f"  baseline RMSE={ev_base['rmse']:.4f}  full RMSE={ev_full['rmse']:.4f}"
            f"  Δ={ev_full['rmse']-ev_base['rmse']:+.4f}"
            f"  dir_acc={ev_full['directional_accuracy']:.3f}"
        )

    if full_rmses:
        print(f"\n  Mean full RMSE:    {np.mean(full_rmses):.4f}  (baseline: {np.mean(baseline_rmses):.4f})")
        print(f"  Mean dir accuracy: {np.mean(full_daccs):.4f}  (baseline: {np.mean(baseline_daccs):.4f})")

    # -----------------------------------------------------------------------
    # 4. Two-stage final training
    # -----------------------------------------------------------------------
    divider(f"Two-stage training  (h={args.horizon} bars = {args.horizon*10//60} min)")
    train_df, val_df, test_df = split_by_date(feat, TRAIN_END, VAL_END)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    train_clean = train_df.dropna(subset=feature_cols + [target_col])
    val_clean   = val_df.dropna(subset=feature_cols + [target_col])
    test_clean  = test_df.dropna(subset=feature_cols + [target_col])

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_val   = val_clean[feature_cols].values
    y_val   = val_clean[target_col].values
    X_test  = test_clean[feature_cols].values
    y_test  = test_clean[target_col].values

    # Stage 1: early stopping on 90/10 split of training set
    split = int(len(X_train) * 0.9)
    model_s1, _ = train_model(
        X_train[:split], y_train[:split],
        X_train[split:], y_train[split:],
    )
    print(f"Stage 1 best_iteration: {model_s1.best_iteration_}")

    # Stage 2: refit on train + val combined
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    model = retrain_model(model_s1, X_combined, y_combined)
    print(f"Stage 2 refit on {len(X_combined):,} rows (train + val)")

    # -----------------------------------------------------------------------
    # 5. Validation metrics
    # -----------------------------------------------------------------------
    divider("Validation set metrics (2025-01 → 2025-06)")
    val_metrics = evaluate_model(model, X_val, y_val)
    naive_rmse_val = float(np.sqrt(np.mean((y_val - y_train.mean()) ** 2)))
    for k, v in val_metrics.items():
        print(f"  {k:30s}: {v:.4f}")
    print(f"  {'naive_rmse':30s}: {naive_rmse_val:.4f}")
    print(f"  RMSE improvement vs naive:       {naive_rmse_val - val_metrics['rmse']:+.4f}")

    # -----------------------------------------------------------------------
    # 6. Test set metrics
    # -----------------------------------------------------------------------
    divider("Test set metrics (2025-07 → 2026-03)  ← gate check")
    test_metrics = evaluate_model(model, X_test, y_test)
    naive_rmse_test = float(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)))
    rmse_improvement = naive_rmse_test - test_metrics["rmse"]
    for k, v in test_metrics.items():
        print(f"  {k:30s}: {v:.4f}")
    print(f"  {'naive_rmse':30s}: {naive_rmse_test:.4f}")
    print(f"  RMSE improvement vs naive:       {rmse_improvement:+.4f}")

    # Feature importance
    divider("Feature importance (top 15)")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat_name, imp in importance.head(15).items():
        print(f"  {feat_name:30s}: {imp:6.0f}")

    # -----------------------------------------------------------------------
    # 7. Save model artefact
    # -----------------------------------------------------------------------
    model_path = models_dir / f"lgbm_regression_h{args.horizon}.pkl"
    artefact = {"model": model, "feature_cols": feature_cols}
    with open(model_path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\nModel saved → {model_path}")

    # -----------------------------------------------------------------------
    # 8. Plot model predictions
    # -----------------------------------------------------------------------
    divider("Generating plot_model_predictions")
    y_pred_test = model.predict(X_test)
    fig = plot_model_predictions(
        test_clean,
        y_pred_test,
        plots_dir / f"model_predictions_h{args.horizon}.png",
    )
    plt.close(fig)
    print(f"Plot saved → {plots_dir}/model_predictions_h{args.horizon}.png")

    # -----------------------------------------------------------------------
    # 9. Gate summary
    # -----------------------------------------------------------------------
    divider("Gate Summary")
    gate_dir  = test_metrics["directional_accuracy"] > 0.55
    gate_rmse = rmse_improvement > 0
    print(f"  [{'✓' if gate_dir else '✗'}] Directional accuracy > 0.55  (actual: {test_metrics['directional_accuracy']:.4f})")
    print(f"  [{'✓' if gate_rmse else '✗'}] RMSE improvement over naive  (actual: {rmse_improvement:+.4f})")
    all_pass = gate_rmse
    print(f"\n  Overall: {'PASS — proceed to run_backtest.py' if all_pass else 'NEEDS INVESTIGATION'}")

    print(f"\nTraining complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
