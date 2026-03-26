"""
LightGBM regression model for the triangle z-score forecasting pipeline.

Mirrors the architecture of volare/model.py (vol forecasting pipeline) closely:
  - LGBMRegressor, objective='regression' (L2), eval_metric='rmse'
  - Two-stage training: Stage 1 finds best_iteration_ on val; Stage 2 retrains
    on combined train+val with fixed n_estimators = best_iteration_
  - No Platt scaling (regression, not classification)
  - Evaluate on RMSE, MAE, directional accuracy, simulated Sharpe

Python 3.9 compatible: uses Optional[X] from typing, not X | Y union syntax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# Default hyperparameters (mirrors vol model defaults)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "feature_fraction":  0.9,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "objective":         "regression",
    "metric":            "rmse",
    "n_jobs":            -1,
    "verbose":           -1,
}

# Feature columns: everything except regression targets and raw prices
_EXCLUDE_COLS = {
    "eurusd", "audusd", "euraud", "euraud_bid", "euraud_ask",
    "z_future_30", "z_future_60", "z_future_180",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model feature column names (all columns except targets/prices)."""
    return [c for c in df.columns if c not in _EXCLUDE_COLS]


# ---------------------------------------------------------------------------
# Training — two-stage (mirrors volare/model.py)
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[dict] = None,
) -> tuple[LGBMRegressor, dict]:
    """Stage 1: train with early stopping on validation RMSE.

    Args:
        X_train/y_train: Training features and continuous z-score targets.
        X_val/y_val:     Validation features and targets (for early stopping).
        params:          Override DEFAULT_PARAMS (merged, not replaced).

    Returns:
        (model, evals_result) — fitted LGBMRegressor and eval history dict.
        model.best_iteration_ gives the optimal number of trees.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = LGBMRegressor(**p)
    evals_result: dict = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
            record_evaluation(evals_result),
        ],
    )
    return model, evals_result


def retrain_model(
    model: LGBMRegressor,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> LGBMRegressor:
    """Stage 2: refit on combined train+val with fixed n_estimators.

    Uses model.best_iteration_ from Stage 1 as the fixed tree count,
    so the final model has the same capacity but more training data.

    Args:
        model:      Stage-1 fitted model (provides best_iteration_ and hyperparams).
        X_combined: Concatenated train+val features.
        y_combined: Concatenated train+val targets.

    Returns:
        New fitted LGBMRegressor trained on combined data.
    """
    params = model.get_params()
    params["n_estimators"] = model.best_iteration_
    model_refit = LGBMRegressor(**params)
    model_refit.fit(X_combined, y_combined)
    return model_refit


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LGBMRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Evaluate model on a held-out set.

    Args:
        model: Trained LGBMRegressor (Stage 2 refit or Stage 1).
        X/y:   Feature matrix and true future z-score targets.

    Returns:
        Dict with keys:
            rmse               — root mean squared error
            mae                — mean absolute error
            directional_accuracy — fraction where sign(pred) == sign(actual)
                                   (trading-relevant: did we get direction right?)
            mean_pred          — mean predicted z-score
            mean_actual        — mean actual z-score (sanity check vs zero)
    """
    y_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae  = float(mean_absolute_error(y, y_pred))

    # Directional accuracy: exclude near-zero actuals to avoid noise dominating
    nonzero_mask = np.abs(y) > 1e-6
    if nonzero_mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(y_pred[nonzero_mask]) == np.sign(y[nonzero_mask])))
    else:
        dir_acc = float("nan")

    return {
        "rmse":                rmse,
        "mae":                 mae,
        "directional_accuracy": dir_acc,
        "mean_pred":           float(y_pred.mean()),
        "mean_actual":         float(y.mean()),
    }


def simulated_sharpe(
    feat_df: pd.DataFrame,
    model: LGBMRegressor,
    feature_cols: list[str],
    move_threshold: float = 1.0,
    horizon_bars: int = 60,
    bars_per_year: int = 2_628_000,  # 365.25 * 24 * 3600 / 10
) -> dict:
    """Compute simulated Sharpe on a feature+target DataFrame.

    For each bar where |predicted_move| > move_threshold, enter trade at bar t,
    exit at bar t + horizon_bars. P&L = direction × (zscore[t] - zscore[t + horizon_bars]).

    Args:
        feat_df:         DataFrame with feature columns and 'zscore' column.
        model:           Trained LGBMRegressor.
        feature_cols:    Ordered list of feature column names.
        move_threshold:  Minimum |predicted_move| to trigger entry (z-score units).
        horizon_bars:    Exit after this many bars.
        bars_per_year:   Bars per year for annualisation (10s bars: 365.25d × 8640).

    Returns:
        Dict with keys:
            n_trades        — number of signal bars
            sharpe          — annualised Sharpe ratio
            mean_pnl        — mean per-trade P&L (z-score units)
            total_pnl       — sum of all trade P&Ls
            hit_rate        — fraction of trades with positive P&L
    """
    X = feat_df[feature_cols].values
    z_current = feat_df["zscore"].values

    # Forward z-score (actual exit value)
    z_exit = feat_df["zscore"].shift(-horizon_bars).values

    y_pred = model.predict(X)
    predicted_move = z_current - y_pred   # expected closure direction/magnitude

    signal_mask = (np.abs(predicted_move) > move_threshold) & ~np.isnan(z_exit)
    n_trades = int(signal_mask.sum())

    if n_trades == 0:
        return {
            "n_trades": 0,
            "sharpe": float("nan"),
            "mean_pnl": float("nan"),
            "total_pnl": float("nan"),
            "hit_rate": float("nan"),
        }

    direction = np.sign(predicted_move[signal_mask])
    pnl = direction * (z_current[signal_mask] - z_exit[signal_mask])

    ann_factor = np.sqrt(bars_per_year / horizon_bars)
    sharpe = float(pnl.mean() / (pnl.std() + 1e-10) * ann_factor)

    return {
        "n_trades":  n_trades,
        "sharpe":    sharpe,
        "mean_pnl":  float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "hit_rate":  float((pnl > 0).mean()),
    }


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
        df:           Full feature+target DataFrame with DatetimeIndex.
        train_start:  Start of the training window (inclusive).
        train_end:    End of the training window — folds are built within this range.
        fold_months:  Size of each incremental training chunk in months.
        oos_months:   Out-of-sample test window per fold in months.
        buffer_bars:  Gap between fold train end and OOS start (target-leakage buffer).

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
        oos_end = cursor + relativedelta(months=oos_months)
        fold_oos = df[(df.index >= cursor) & (df.index < oos_end)]

        # Apply target-leakage buffer: drop last `buffer_bars` from train end
        if len(fold_train) > buffer_bars and len(fold_oos) > 0:
            folds.append({
                "fold":     fold_num,
                "train_df": fold_train.iloc[:-buffer_bars],
                "oos_df":   fold_oos,
            })

        fold_num += 1
        cursor += relativedelta(months=fold_months)

    return folds
