"""
LightGBM binary classifier + Platt calibration for the triangle gap closure model.

Mirrors the architecture of volare/model.py (vol forecasting pipeline) but
adapted for binary classification. Key differences:
  - LGBMClassifier (not Regressor), objective='binary', metric='auc'
  - Platt scaling for probability calibration (LogisticRegression on raw scores)
  - Date-boundary splits (not fraction-based) — see labels.split_by_date()

Python 3.9 compatible: uses Optional[X] from typing, not X | Y union syntax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
)


# ---------------------------------------------------------------------------
# Default hyperparameters (mirrors vol model defaults)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "objective":         "binary",
    "metric":            "auc",
    "n_jobs":            -1,
    "verbose":           -1,
}

# Feature columns: everything except these targets/identifiers
# Exclude raw prices and the target; keep residual and zscore as features
_EXCLUDE_COLS = {"eurusd", "audusd", "euraud", "euraud_bid", "euraud_ask", "label"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model feature column names (all columns except targets/prices)."""
    return [c for c in df.columns if c not in _EXCLUDE_COLS]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[dict] = None,
) -> LGBMClassifier:
    """Train LGBMClassifier with early stopping on validation AUC.

    Args:
        X_train/y_train: Training features and binary labels.
        X_val/y_val:     Validation features and binary labels (for early stopping).
        params:          Override DEFAULT_PARAMS (merged, not replaced).

    Returns:
        Fitted LGBMClassifier.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = LGBMClassifier(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model: LGBMClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> LogisticRegression:
    """Fit a Platt scaling calibrator on the validation set.

    Fits LogisticRegression(raw_scores → calibrated_probs) using the model's
    predicted probabilities on the validation set as inputs.

    Returns:
        Fitted LogisticRegression (single feature: raw P(closure)).
    """
    raw_probs = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(raw_probs, y_val)
    return calibrator


def predict_proba_calibrated(
    model: LGBMClassifier,
    calibrator: LogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    """Return calibrated P(closure) for feature matrix X."""
    raw_probs = model.predict_proba(X)[:, 1].reshape(-1, 1)
    return calibrator.predict_proba(raw_probs)[:, 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LGBMClassifier,
    calibrator: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.65,
) -> dict:
    """Evaluate the calibrated model on a held-out set.

    Args:
        model/calibrator: Trained model and Platt scaler.
        X/y:              Feature matrix and true binary labels.
        threshold:        Probability threshold for precision calculation.

    Returns:
        Dict with keys:
            auc_roc          — area under ROC curve
            brier_score      — calibration quality (lower = better)
            n_above_threshold — number of bars where P(closure) > threshold
            precision_at_thr — precision among bars where P > threshold
            mean_prob        — mean predicted probability
            label_rate       — actual positive rate in y
    """
    probs = predict_proba_calibrated(model, calibrator, X)
    above = probs >= threshold

    result = {
        "auc_roc":           float(roc_auc_score(y, probs)),
        "brier_score":       float(brier_score_loss(y, probs)),
        "n_above_threshold": int(above.sum()),
        "mean_prob":         float(probs.mean()),
        "label_rate":        float(y.mean()),
    }
    if above.sum() > 0:
        result["precision_at_thr"] = float(precision_score(y[above], (probs[above] >= threshold)))
    else:
        result["precision_at_thr"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------

def walk_forward_folds(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    fold_months: int = 2,
    oos_months: int = 1,
    buffer_bars: int = 60,
) -> list[dict]:
    """Generate expanding-window walk-forward folds within the training period.

    Args:
        df:           Full feature+label DataFrame with DatetimeIndex.
        train_start:  Start of the training window (inclusive).
        train_end:    End of the training window — folds are built within this range.
        fold_months:  Size of each incremental training chunk in months.
        oos_months:   Out-of-sample test window per fold in months.
        buffer_bars:  Gap between fold train end and OOS start (label-leakage buffer).

    Returns:
        List of dicts, each with keys: fold, train_df, oos_df.
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(train_start, tz="UTC")
    end   = pd.Timestamp(train_end,   tz="UTC")

    folds = []
    fold_num = 0
    cursor = start + relativedelta(months=fold_months)

    while cursor + relativedelta(months=oos_months) <= end:
        fold_train = df[(df.index >= start) & (df.index < cursor)]
        fold_oos_start = df.index[df.index >= cursor]
        if len(fold_oos_start) == 0:
            break
        oos_start_idx = fold_oos_start[0]
        oos_end = cursor + relativedelta(months=oos_months)
        fold_oos = df[(df.index >= cursor) & (df.index < oos_end)]

        # Apply label-leakage buffer: drop last `buffer_bars` from train end
        if len(fold_train) > buffer_bars and len(fold_oos) > 0:
            folds.append({
                "fold":     fold_num,
                "train_df": fold_train.iloc[:-buffer_bars],
                "oos_df":   fold_oos,
            })

        fold_num += 1
        cursor += relativedelta(months=fold_months)

    return folds
