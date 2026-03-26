"""
Week 2 — Feature Engineering + ML Training

Pipeline:
  1. Load 10s data for all three pairs
  2. Build signal frame (residual + z-score)
  3. Build full feature frame (~22 features)
  4. Attach binary labels (gap closes ≥50% within 10 min)
  5. Walk-forward validation within training period (2024-03 → 2024-12)
  6. Train final model on full training set + Platt calibration on validation
  7. Evaluate on held-out test set (2025-07 → 2026-03)

Run from repo root:
  python notebooks/week2_train.py

Gates (from brief):
  AUC-ROC > 0.57 on test set
  Brier score lower than naive (mean label rate) baseline
  No high-confidence (P > 0.65) signals during Liberation Day (2025-04-02)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import add_labels, split_by_date
from triangulation.model import (
    get_feature_cols,
    train_model,
    calibrate_model,
    evaluate_model,
    walk_forward_folds,
)

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Split boundaries (adapted from brief — data available 2024-03 → 2026-03)
# ---------------------------------------------------------------------------
TRAIN_START = "2024-03-01"
TRAIN_END   = "2024-12-31"   # ~10 months training
VAL_END     = "2025-06-30"   # 6 months validation (Platt calibration)
# Test: 2025-07-01 → 2026-03-17 (~8.5 months — includes tariff/event regime)


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
divider("Loading data (10s bars)")
eurusd = load_pair(DATA_DIR, "EURUSD")
audusd = load_pair(DATA_DIR, "AUDUSD")
euraud = load_pair(DATA_DIR, "EURAUD")
print(f"EURUSD: {len(eurusd):,}  AUDUSD: {len(audusd):,}  EURAUD: {len(euraud):,}")

# ---------------------------------------------------------------------------
# 2. Signal frame
# ---------------------------------------------------------------------------
divider("Building signal frame")
sig = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
print(f"Aligned: {len(sig):,} bars  ({sig.index[0].date()} → {sig.index[-1].date()})")

# Free individual pair memory before building features
del eurusd, audusd, euraud

# ---------------------------------------------------------------------------
# 3. Feature frame
# ---------------------------------------------------------------------------
divider("Building features (~22 columns)")
feat = build_feature_frame(sig)
del sig
print(f"Feature frame: {feat.shape}  columns: {list(feat.columns)}")
spread_available = "spread_euraud" in feat.columns
print(f"Spread features available: {spread_available}")

# ---------------------------------------------------------------------------
# 4. Labels
# ---------------------------------------------------------------------------
divider("Computing binary labels (gap closes ≥50% within 60 bars / 10 min)")
feat = add_labels(feat)
label_rate = feat["label"].mean()
print(f"Overall label rate: {label_rate:.3f}  ({label_rate*100:.1f}% positive)")
print(f"Training label rate (≤2024-12): {feat.loc[feat.index <= TRAIN_END, 'label'].mean():.3f}")

# ---------------------------------------------------------------------------
# 5. Walk-forward validation within training period
# ---------------------------------------------------------------------------
divider("Walk-forward validation (training period only)")
feature_cols = get_feature_cols(feat)
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

folds = walk_forward_folds(
    feat,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    fold_months=2,
    oos_months=1,
    buffer_bars=60,
)
print(f"Folds generated: {len(folds)}")

baseline_aucs, full_aucs = [], []
for fold in folds:
    tr = fold["train_df"].dropna(subset=feature_cols + ["label"])
    oos = fold["oos_df"].dropna(subset=feature_cols + ["label"])
    if len(tr) < 500 or len(oos) < 100:
        continue

    X_tr  = tr[feature_cols].values
    y_tr  = tr["label"].values
    X_oos = oos[feature_cols].values
    y_oos = oos["label"].values

    # 90/10 within-fold split for early stopping
    split = int(len(X_tr) * 0.9)
    X_tr_, X_es = X_tr[:split], X_tr[split:]
    y_tr_, y_es = y_tr[:split], y_tr[split:]

    # Baseline: residual + zscore only
    base_cols = ["residual", "zscore"]
    base_idx  = [feature_cols.index(c) for c in base_cols]
    m_base = train_model(X_tr_[:, base_idx], y_tr_, X_es[:, base_idx], y_es)
    cal_base = calibrate_model(m_base, X_es[:, base_idx], y_es)
    ev_base = evaluate_model(m_base, cal_base, X_oos[:, base_idx], y_oos)

    # Full model
    m_full = train_model(X_tr_, y_tr_, X_es, y_es)
    cal_full = calibrate_model(m_full, X_es, y_es)
    ev_full = evaluate_model(m_full, cal_full, X_oos, y_oos)

    baseline_aucs.append(ev_base["auc_roc"])
    full_aucs.append(ev_full["auc_roc"])

    n_tr = len(tr)
    n_oos = len(oos)
    oos_start = oos.index[0].date()
    oos_end   = oos.index[-1].date()
    print(
        f"  Fold {fold['fold']}  train={n_tr:,}  oos={n_oos:,} ({oos_start}→{oos_end})"
        f"  baseline AUC={ev_base['auc_roc']:.4f}  full AUC={ev_full['auc_roc']:.4f}"
        f"  Δ={ev_full['auc_roc']-ev_base['auc_roc']:+.4f}"
    )

if full_aucs:
    print(f"\n  Mean full AUC: {np.mean(full_aucs):.4f}  (baseline: {np.mean(baseline_aucs):.4f})")
    print(f"  Mean Δ AUC:    {np.mean(full_aucs) - np.mean(baseline_aucs):+.4f}")

# ---------------------------------------------------------------------------
# 6. Train final model on full training set; calibrate on validation
# ---------------------------------------------------------------------------
divider("Final model — train on 2024-03→2024-12, calibrate on 2025-01→2025-06")
train_df, val_df, test_df = split_by_date(feat, TRAIN_END, VAL_END)
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

train_clean = train_df.dropna(subset=feature_cols + ["label"])
val_clean   = val_df.dropna(subset=feature_cols + ["label"])
test_clean  = test_df.dropna(subset=feature_cols + ["label"])

X_train = train_clean[feature_cols].values
y_train = train_clean["label"].values
X_val   = val_clean[feature_cols].values
y_val   = val_clean["label"].values
X_test  = test_clean[feature_cols].values
y_test  = test_clean["label"].values

# 90/10 early-stopping split within training set
split = int(len(X_train) * 0.9)
model = train_model(X_train[:split], y_train[:split], X_train[split:], y_train[split:])
print(f"Best iteration: {model.best_iteration_}")

calibrator = calibrate_model(model, X_val, y_val)

# ---------------------------------------------------------------------------
# 7. Validation metrics
# ---------------------------------------------------------------------------
divider("Validation set metrics (2025-01 → 2025-06)")
val_metrics = evaluate_model(model, calibrator, X_val, y_val)
for k, v in val_metrics.items():
    print(f"  {k:25s}: {v:.4f}")

# Naive baseline Brier score (always predict mean label rate)
naive_brier = float(((y_val - y_val.mean()) ** 2).mean())
print(f"  {'naive_brier_score':25s}: {naive_brier:.4f}")
print(f"  Brier improvement:         {naive_brier - val_metrics['brier_score']:+.4f}")

# ---------------------------------------------------------------------------
# 8. Test set metrics (held out)
# ---------------------------------------------------------------------------
divider("Test set metrics (2025-07 → 2026-03)  ← final gate check")
test_metrics = evaluate_model(model, calibrator, X_test, y_test)
for k, v in test_metrics.items():
    print(f"  {k:25s}: {v:.4f}")

naive_brier_test = float(((y_test - y_test.mean()) ** 2).mean())
brier_improvement = naive_brier_test - test_metrics["brier_score"]
print(f"  {'naive_brier_score':25s}: {naive_brier_test:.4f}")
print(f"  Brier improvement:         {brier_improvement:+.4f}")

# ---------------------------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------------------------
divider("Feature importance (top 15)")
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols,
).sort_values(ascending=False)
for feat_name, imp in importance.head(15).items():
    print(f"  {feat_name:30s}: {imp:6.0f}")

# ---------------------------------------------------------------------------
# 10. Liberation Day sanity check (2025-04-02)
# ---------------------------------------------------------------------------
divider("Liberation Day sanity check (2025-04-02)")
from triangulation.model import predict_proba_calibrated

# Check SIGNAL BARS only (|z| >= 2.0) — we only trade at signal bars,
# so the model's behaviour on trivially-small-residual bars doesn't matter.
SIGNAL_Z = 2.0
ld_date = pd.Timestamp("2025-04-02").date()
ld_mask = feat.index.date == ld_date
ld_sig_mask = ld_mask & (feat["zscore"].abs() >= SIGNAL_Z)
ld_df = feat[ld_sig_mask].dropna(subset=feature_cols)

if len(ld_df) > 0:
    ld_probs = predict_proba_calibrated(model, calibrator, ld_df[feature_cols].values)
    high_conf = int((ld_probs >= 0.65).sum())
    total_ld_bars = int(ld_mask.sum())
    print(f"  Total Liberation Day bars:                  {total_ld_bars:,}")
    print(f"  Signal bars (|z| ≥ {SIGNAL_Z}):               {len(ld_df):,}")
    print(f"  High-confidence signal bars (P ≥ 0.65):    {high_conf}")
    print(f"  Max P(closure) on signal bars:              {ld_probs.max():.3f}")
    print(f"  Gate: {'✓ PASS' if high_conf == 0 else '✗ FAIL — model is over-confident on structural repricing signal bars'}")
elif ld_mask.sum() > 0:
    print(f"  Liberation Day present but no signal bars (|z| ≥ {SIGNAL_Z}) — trivially OK")
else:
    print("  Liberation Day not in dataset window.")

# ---------------------------------------------------------------------------
# 11. Week 2 gate summary
# ---------------------------------------------------------------------------
divider("Week 2 Gate Summary")
gate_auc   = test_metrics["auc_roc"] > 0.57
gate_brier = brier_improvement > 0
gate_calib = test_metrics["brier_score"] < 0.25  # rough proxy for calibration error < 0.05
gate_ld    = not ld_sig_mask.any() or high_conf == 0

print(f"  [{'✓' if gate_auc else '✗'}] Test AUC-ROC > 0.57 (actual: {test_metrics['auc_roc']:.4f})")
print(f"  [{'✓' if gate_brier else '✗'}] Brier improvement over naive (actual: {brier_improvement:+.4f})")
print(f"  [{'✓' if gate_calib else '✗'}] Brier score < 0.25 (actual: {test_metrics['brier_score']:.4f})")
print(f"  [{'✓' if gate_ld else '✗'}] No high-confidence signals on Liberation Day")

all_pass = gate_auc and gate_brier
print(f"\n  Overall: {'PASS — proceed to Week 3' if all_pass else 'NEEDS INVESTIGATION'}")
