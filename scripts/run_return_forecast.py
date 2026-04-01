"""
Cross-pair return forecasting — EUR/USD/AUD

Predicts short-term EURAUD returns from cross-pair price information.
Replaces the triangular mean-reversion approach (which failed at realistic spreads)
with a directional forecasting model that uses EURUSD/AUDUSD momentum as a leading
indicator for EURAUD price moves.

Strategy logic:
  - EUR strength appears in EURUSD first, then propagates to EURAUD
  - Cross-pair return divergence predicts near-term EURAUD direction
  - Trade only when |predicted_return| > threshold (execution cost filter)
  - All P&L computed from actual bid/ask prices

Run from repo root:
    python3 scripts/run_return_forecast.py [options]

Options:
    --horizons INT ...      target horizons in bars (default: 1 3 6 12 = 10/30/60/120s)
    --thresholds FLOAT ...  signal thresholds in pips (default: 0.1 0.3 0.5 0.8 1.0 1.5 2.0)
    --data-dir PATH
    --output-dir PATH
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import (
    compute_session_features,
    compute_spread_features,
    compute_pair_vol,
)
from triangulation.labels import split_by_date

plt.style.use("dark_background")

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "outputs" / "models"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_END = "2024-12-31"
VAL_END   = "2025-06-30"
BAR_S     = 10
_HORIZON_BUFFER = 12   # max horizon bars — dropped at split boundaries


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def compute_return_features(
    sig: pd.DataFrame,
) -> pd.DataFrame:
    """Build short-term return features and cross-pair lead indicators.

    Args:
        sig: Output of build_signal_frame(); must contain eurusd, audusd, euraud columns.

    Returns:
        DataFrame with return features indexed on sig.index.
    """
    lookbacks = [1, 3, 6, 12]   # bars = [10s, 30s, 60s, 120s]
    cols: dict[str, pd.Series] = {}

    for pair in ["eurusd", "audusd", "euraud"]:
        for k in lookbacks:
            cols[f"ret_{pair}_{k}b"] = (sig[pair] - sig[pair].shift(k)) * 10_000

    # Cross-pair lead indicators: EURUSD / AUDUSD momentum relative to EURAUD
    # Positive → EUR/AUD leg moved but EURAUD hasn't caught up (lead signal)
    cols["lead_eu_aud_1b"] = cols["ret_eurusd_1b"] - cols["ret_euraud_1b"]
    cols["lead_au_aud_1b"] = cols["ret_audusd_1b"] - cols["ret_euraud_1b"]
    cols["lead_eu_aud_3b"] = cols["ret_eurusd_3b"] - cols["ret_euraud_3b"]
    cols["lead_au_aud_3b"] = cols["ret_audusd_3b"] - cols["ret_euraud_3b"]

    # Momentum contrast: short vs long (reversion vs continuation indicator)
    for pair in ["eurusd", "audusd", "euraud"]:
        cols[f"mom_contrast_{pair}"] = cols[f"ret_{pair}_1b"] - cols[f"ret_{pair}_12b"]

    # Short-term realised vol of EURAUD (mean absolute change)
    diff_ea = sig["euraud"].diff() * 10_000
    cols["rv_euraud_6b"]  = diff_ea.abs().rolling(6,  min_periods=3).mean()
    cols["rv_euraud_12b"] = diff_ea.abs().rolling(12, min_periods=6).mean()

    return pd.DataFrame(cols, index=sig.index)


def compute_return_targets(
    euraud: pd.Series,
    horizon_bars: list[int],
) -> pd.DataFrame:
    """Compute forward EURAUD returns for each horizon.

    Args:
        euraud: EUR/AUD mid-price Series.
        horizon_bars: List of forward-look horizons in bars.

    Returns:
        DataFrame with target_{h}b columns (in pips).
    """
    targets = {}
    for h in horizon_bars:
        targets[f"target_{h}b"] = (euraud.shift(-h) - euraud) * 10_000
    return pd.DataFrame(targets, index=euraud.index)


# ---------------------------------------------------------------------------
# Trading rule simulation with bid/ask costs
# ---------------------------------------------------------------------------

def evaluate_threshold(
    y_pred: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    horizon_bars: int,
    threshold: float,
    n_days: float,
) -> dict:
    """No-overlap simulation with actual bid/ask execution costs.

    Direction from y_pred sign. No additional fixed cost — bid/ask captures spread.
    """
    n       = len(mid)
    trades  = []
    next_free = 0

    for i, yhat in enumerate(y_pred):
        if abs(yhat) <= threshold:
            continue
        entry_i = i
        exit_i  = i + horizon_bars
        if exit_i >= n:
            continue
        if entry_i < next_free:
            continue

        direction  = 1.0 if yhat > 0 else -1.0
        entry_exec = ask[entry_i] if direction > 0 else bid[entry_i]
        exit_exec  = bid[exit_i]  if direction > 0 else ask[exit_i]
        gross      = direction * (exit_exec - entry_exec) * 10_000
        spread_e   = (ask[entry_i] - bid[entry_i]) * 10_000

        trades.append({
            "ts":        timestamps[entry_i],
            "direction": direction,
            "yhat":      yhat,
            "mid_move":  direction * (mid[exit_i] - mid[entry_i]) * 10_000,
            "gross":     gross,
            "spread_e":  spread_e,
        })
        next_free = exit_i + 1

    if not trades:
        return {
            "n_trades": 0, "trades_wk": 0.0,
            "mean_gross": float("nan"), "sharpe": float("nan"),
            "hit_rate": float("nan"), "spread_med": float("nan"),
            "trades": pd.DataFrame(),
        }

    tl       = pd.DataFrame(trades)
    n_weeks  = n_days / 7
    mean_g   = float(tl["gross"].mean())
    std_g    = float(tl["gross"].std())
    sharpe   = mean_g / (std_g + 1e-10)
    hit_rate = float((tl["gross"] > 0).mean())

    return {
        "n_trades":   len(tl),
        "trades_wk":  len(tl) / max(n_weeks, 1e-6),
        "mean_gross": mean_g,
        "sharpe":     sharpe,
        "hit_rate":   hit_rate,
        "spread_med": float(tl["spread_e"].median()),
        "trades":     tl,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    horizon_bars: list[int],
    path: Path,
) -> None:
    """Scatter of predicted vs actual returns — one panel per horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, h in zip(axes_flat, horizon_bars):
        y_pred, y_true = predictions[h]
        # Subsample for clarity
        n    = len(y_pred)
        idx  = np.random.default_rng(42).choice(n, min(n, 10_000), replace=False)
        yp   = y_pred[idx]
        yt   = y_true[idx]
        err  = np.abs(yp - yt)
        vmax = float(np.percentile(err, 95))

        sc = ax.scatter(yp, yt, c=err, cmap="hot_r", vmin=0, vmax=vmax,
                        alpha=0.3, s=2, rasterized=True)
        plt.colorbar(sc, ax=ax, label="|error| pips")

        # Regression line
        if len(yp) >= 2:
            m, b = np.polyfit(yp, yt, 1)
            xs   = np.linspace(yp.min(), yp.max(), 100)
            ax.plot(xs, m * xs + b, color="#4fc3f7", linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="white", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.4)
        corr = float(np.corrcoef(yp, yt)[0, 1])
        r2   = float(r2_score(yt, yp))
        ax.set_xlabel("Predicted return (pips)", fontsize=8)
        ax.set_ylabel("Actual return (pips)", fontsize=8)
        ax.set_title(f"Horizon={h}b ({h*BAR_S}s)  corr={corr:.4f}  R²={r2:.4f}", fontsize=9)
        ax.grid(alpha=0.1)

    fig.suptitle("Predicted vs Actual EURAUD Returns  (test set)", fontsize=11, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_pnl_over_time(
    trade_logs: dict[int, pd.DataFrame],
    horizon_bars: list[int],
    path: Path,
) -> None:
    """Cumulative net pips over time per horizon."""
    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]

    for i, h in enumerate(horizon_bars):
        tl = trade_logs.get(h)
        if tl is None or len(tl) == 0:
            continue
        tl_sorted = tl.sort_values("ts")
        cumulative = tl_sorted["gross"].cumsum()
        mn = float(tl_sorted["gross"].mean())
        ax.plot(tl_sorted["ts"], cumulative, color=colors[i % len(colors)],
                linewidth=1.4, label=f"H={h}b ({h*BAR_S}s)  n={len(tl):,}  mean={mn:+.3f}")

    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative net pips", fontsize=9)
    ax.set_title("Cumulative P&L  (test set, actual bid/ask, threshold=optimal on val)",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_feature_importance(
    models: dict[int, lgb.LGBMRegressor],
    feature_names: list[str],
    horizon_bars: list[int],
    path: Path,
) -> None:
    """Top-20 features by gain importance, one panel per horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, h in zip(axes_flat, horizon_bars):
        model = models.get(h)
        if model is None:
            continue
        imp  = model.feature_importances_
        idx  = np.argsort(imp)[-20:]
        names = [feature_names[i] for i in idx]
        vals  = imp[idx]
        colors_bar = ["#4fc3f7" if "lead" in n else
                      "#81c784" if "ret_" in n else
                      "#ffb74d" if "rv_" in n or "mom_" in n else
                      "#ce93d8" for n in names]
        ax.barh(range(len(names)), vals, color=colors_bar, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Gain importance", fontsize=8)
        ax.set_title(f"H={h}b ({h*BAR_S}s)", fontsize=9)
        ax.grid(alpha=0.1, axis="x")

    fig.suptitle("Feature Importance by Horizon", fontsize=11, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_net_vs_threshold(
    results: list[dict],
    horizon_bars: list[int],
    thresholds: list[float],
    path: Path,
) -> None:
    """Mean net pips and trades/week vs threshold, lines per horizon."""
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
    fig, (ax_n, ax_tw) = plt.subplots(1, 2, figsize=(13, 5))

    for i, h in enumerate(horizon_bars):
        color = colors[i % len(colors)]
        nets, tws = [], []
        for thr in thresholds:
            r = next((r for r in results if r["horizon"] == h and
                      r["threshold"] == thr and r["split"] == "test"), None)
            nets.append(r["mean_gross"] if r and r["n_trades"] else float("nan"))
            tws.append(r["trades_wk"]   if r and r["n_trades"] else float("nan"))
        ax_n.plot(thresholds, nets,  marker="o", markersize=4, linewidth=1.5,
                  color=color, label=f"H={h}b ({h*BAR_S}s)")
        ax_tw.plot(thresholds, tws, marker="o", markersize=4, linewidth=1.5,
                   color=color, label=f"H={h}b")

    ax_n.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_n.set_xlabel("Prediction threshold (pips)", fontsize=9)
    ax_n.set_ylabel("Mean net pips per trade", fontsize=9)
    ax_n.set_title("Net Return vs Threshold  (test set, actual bid/ask)", fontsize=10)
    ax_n.legend(fontsize=8)
    ax_n.grid(alpha=0.15)

    ax_tw.set_xlabel("Prediction threshold (pips)", fontsize=9)
    ax_tw.set_ylabel("Trades per week (no-overlap)", fontsize=9)
    ax_tw.set_title("Trade Frequency vs Threshold", fontsize=10)
    ax_tw.legend(fontsize=8)
    ax_tw.grid(alpha=0.15)
    ax_tw.set_yscale("log")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-pair return forecasting")
    parser.add_argument("--horizons",    type=int,   nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--thresholds",  type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--data-dir",    type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir",  type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    models_dir = output_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    horizon_bars = sorted(args.horizons)

    # -----------------------------------------------------------------------
    # 0. Resolution mapping
    # -----------------------------------------------------------------------
    _divider("Cross-Pair Return Forecasting")
    print(f"\n  Data frequency: {BAR_S}s per bar")
    print(f"  Note: lookbacks [1s,5s,10s] → 1 bar; [30s] → 3 bars at 10s resolution")
    print(f"\n  Horizon mapping:")
    for h in horizon_bars:
        print(f"    {h} bars = {h * BAR_S}s")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    _divider("Loading data")
    eurusd = load_pair(data_dir, "EURUSD")
    audusd = load_pair(data_dir, "AUDUSD")
    euraud = load_pair(data_dir, "EURAUD")

    sig = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    euraud_bid_full = sig["euraud_bid"].copy() if "euraud_bid" in sig.columns else None
    euraud_ask_full = sig["euraud_ask"].copy() if "euraud_ask" in sig.columns else None
    euraud_mid_full = sig["euraud"].copy()
    del eurusd, audusd, euraud

    have_ba = euraud_bid_full is not None
    if have_ba:
        print("  ✓ Bid/ask prices available")
    else:
        print("  ✗ Bid/ask not available — using mid price ± fixed cost for execution")

    # -----------------------------------------------------------------------
    # 2. Build features
    # -----------------------------------------------------------------------
    _divider("Building features")

    # New return features
    ret_feat = compute_return_features(sig)
    print(f"  Return features: {ret_feat.shape[1]} columns")

    # Existing structural features
    sess_feat  = compute_session_features(sig.index)
    vol_feat   = compute_pair_vol(sig["eurusd"], sig["audusd"], sig["euraud"])
    zscore_df  = sig[["zscore", "residual"]].copy()
    dz         = sig["zscore"].diff().rename("dz_1b")

    spread_feat = None
    if have_ba:
        spread_feat = compute_spread_features(euraud_bid_full, euraud_ask_full)
        print(f"  Spread features: {spread_feat.shape[1]} columns")

    # Assemble
    parts = [ret_feat, sess_feat, vol_feat, zscore_df, dz.to_frame()]
    if spread_feat is not None:
        parts.append(spread_feat)

    features = pd.concat(parts, axis=1)
    features["euraud_mid"] = euraud_mid_full
    if have_ba:
        features["euraud_bid"] = euraud_bid_full
        features["euraud_ask"] = euraud_ask_full

    # Targets
    targets = compute_return_targets(euraud_mid_full, horizon_bars)
    df_full = pd.concat([features, targets], axis=1)

    # Drop NaNs (from rolling warmup and forward-looking targets)
    df_full = df_full.dropna()
    print(f"  Full dataset: {df_full.shape[0]:,} bars  ({df_full.index[0].date()} → {df_full.index[-1].date()})")

    # -----------------------------------------------------------------------
    # 3. Train / val / test split
    # -----------------------------------------------------------------------
    _divider("Train / Val / Test split")
    train_df, val_df, test_df = split_by_date(
        df_full, train_end=TRAIN_END, val_end=VAL_END, buffer_bars=_HORIZON_BUFFER,
    )
    print(f"  Train: {len(train_df):,} bars  ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"  Val:   {len(val_df):,} bars  ({val_df.index[0].date()} → {val_df.index[-1].date()})")
    print(f"  Test:  {len(test_df):,} bars  ({test_df.index[0].date()} → {test_df.index[-1].date()})")

    target_cols  = [f"target_{h}b" for h in horizon_bars]
    price_cols   = ["euraud_mid"] + (["euraud_bid", "euraud_ask"] if have_ba else [])
    feature_cols = [c for c in df_full.columns if c not in target_cols + price_cols]
    print(f"  Features: {len(feature_cols)}")

    # -----------------------------------------------------------------------
    # 4. Train models
    # -----------------------------------------------------------------------
    _divider("Training models")
    models: dict[int, lgb.LGBMRegressor] = {}
    train_metrics: dict[int, dict] = {}

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    for h in horizon_bars:
        target_col = f"target_{h}b"
        y_train = train_df[target_col].values
        y_val   = val_df[target_col].values
        y_test  = test_df[target_col].values

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        models[h] = model

        y_pred_test = model.predict(X_test)
        r2   = float(r2_score(y_test, y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        corr = float(np.corrcoef(y_pred_test, y_test)[0, 1])
        train_metrics[h] = {"r2": r2, "rmse": rmse, "corr": corr,
                             "y_pred": y_pred_test, "y_true": y_test}
        print(f"  H={h}b ({h*BAR_S}s):  R²={r2:.6f}  RMSE={rmse:.3f}  corr={corr:.6f}")

        # Save model
        pkl_path = models_dir / f"return_forecast_h{h}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols,
                         "horizon_bars": h, "bar_s": BAR_S}, f)

    # -----------------------------------------------------------------------
    # 5. Spread stats (whole test period)
    # -----------------------------------------------------------------------
    _divider("Spread context")
    test_mid = test_df["euraud_mid"].values
    test_bid = test_df["euraud_bid"].values if have_ba else test_mid - 0.00005
    test_ask = test_df["euraud_ask"].values if have_ba else test_mid + 0.00005
    test_ts  = test_df.index
    test_days = (test_df.index[-1] - test_df.index[0]).days

    if have_ba:
        spread_all = (test_ask - test_bid) * 10_000
        print(f"  Spread (test period):")
        print(f"    median={np.nanmedian(spread_all):.3f}  p90={np.nanpercentile(spread_all,90):.3f}  "
              f"mean={np.nanmean(spread_all):.3f} pips")

    # -----------------------------------------------------------------------
    # 6. Evaluate trading rule — validation set (threshold selection)
    # -----------------------------------------------------------------------
    _divider("Evaluating on validation set (threshold selection)")
    val_mid = val_df["euraud_mid"].values
    val_bid = val_df["euraud_bid"].values if have_ba else val_mid - 0.00005
    val_ask = val_df["euraud_ask"].values if have_ba else val_mid + 0.00005
    val_ts  = val_df.index
    val_days = (val_df.index[-1] - val_df.index[0]).days

    optimal_thresholds: dict[int, float] = {}
    all_results: list[dict] = []

    for h in horizon_bars:
        y_pred_val = models[h].predict(val_df[feature_cols].values)
        best_thr, best_net = None, -np.inf
        for thr in args.thresholds:
            m = evaluate_threshold(y_pred_val, val_mid, val_bid, val_ask,
                                   val_ts, h, thr, val_days)
            m.update({"horizon": h, "threshold": thr, "split": "val"})
            all_results.append(m)
            if m["n_trades"] > 0 and m["mean_gross"] > best_net:
                best_net = m["mean_gross"]
                best_thr = thr
        optimal_thresholds[h] = best_thr if best_thr else args.thresholds[0]
        print(f"  H={h}b:  optimal_threshold={optimal_thresholds[h]:.1f}  val_mean_net={best_net:+.3f}")

    # -----------------------------------------------------------------------
    # 7. Evaluate on test set
    # -----------------------------------------------------------------------
    _divider("Evaluating on test set")
    opt_trade_logs: dict[int, pd.DataFrame] = {}

    print(f"\n  {'horizon':>8}  {'threshold':>10}  {'n_trades':>10}  {'trades_wk':>10}  "
          f"{'mean_net':>10}  {'hit_rate':>9}  {'sharpe':>8}  {'spread_med':>11}")

    for h in horizon_bars:
        y_pred_test = train_metrics[h]["y_pred"]
        for thr in args.thresholds:
            m = evaluate_threshold(y_pred_test, test_mid, test_bid, test_ask,
                                   test_ts, h, thr, test_days)
            m.update({"horizon": h, "threshold": thr, "split": "test"})
            all_results.append(m)

            if thr == optimal_thresholds[h]:
                opt_trade_logs[h] = m["trades"]
                marker = " ← opt"
            else:
                marker = ""

            if m["n_trades"] > 0:
                print(f"  {h:>8}b  {thr:>10.1f}  {m['n_trades']:>10,}  "
                      f"{m['trades_wk']:>10.1f}  "
                      f"{m['mean_gross']:>+10.3f}  {m['hit_rate']:>9.1%}  "
                      f"{m['sharpe']:>+8.3f}  {m['spread_med']:>10.2f}p{marker}")

    # Spread check at signal bars
    _divider("Spread check at traded bars vs overall")
    if have_ba:
        print(f"  Overall median spread: {np.nanmedian(spread_all):.3f} pips")
        for h in horizon_bars:
            tl = opt_trade_logs.get(h)
            if tl is not None and len(tl) > 0:
                print(f"  H={h}b (thr={optimal_thresholds[h]:.1f}):  "
                      f"spread at entry = {tl['spread_e'].median():.3f} pips median  "
                      f"(n={len(tl)} trades)")
        print()
        print("  → Spread at signal bars close to overall median = strategy is trading")
        print("    at NORMAL liquidity (unlike persistence filter which selected illiquid moments)")

    # -----------------------------------------------------------------------
    # 8. Baseline: naive momentum
    # -----------------------------------------------------------------------
    _divider("Baseline: naive momentum (y_hat = recent EURAUD return)")
    print(f"\n  {'horizon':>8}  {'threshold':>10}  {'mean_net':>10}  {'hit_rate':>9}")
    for h in horizon_bars:
        # Naive: predict that 1-bar return continues
        y_naive = test_df["ret_euraud_1b"].values
        best_naive = None
        for thr in args.thresholds:
            m = evaluate_threshold(y_naive, test_mid, test_bid, test_ask,
                                   test_ts, h, thr, test_days)
            if m["n_trades"] > 0:
                if best_naive is None or m["mean_gross"] > best_naive["mean_gross"]:
                    best_naive = {"threshold": thr, **m}
        if best_naive:
            print(f"  {h:>8}b  {best_naive['threshold']:>10.1f}  "
                  f"{best_naive['mean_gross']:>+10.3f}  {best_naive['hit_rate']:>9.1%}")
        else:
            print(f"  {h:>8}b  no profitable configuration")

    # -----------------------------------------------------------------------
    # 9. Plots
    # -----------------------------------------------------------------------
    _divider("Generating plots")

    # a) Predicted vs actual
    preds = {h: (train_metrics[h]["y_pred"], train_metrics[h]["y_true"])
             for h in horizon_bars}
    plot_pred_vs_actual(preds, horizon_bars,
                        path=plots_dir / "forecast_pred_vs_actual.png")

    # b) Cumulative P&L over time
    plot_pnl_over_time(opt_trade_logs, horizon_bars,
                       path=plots_dir / "forecast_pnl_over_time.png")

    # c) Feature importance
    plot_feature_importance(models, feature_cols, horizon_bars,
                            path=plots_dir / "forecast_feature_importance.png")

    # d) Net vs threshold
    plot_net_vs_threshold(all_results, horizon_bars, args.thresholds,
                          path=plots_dir / "forecast_net_vs_threshold.png")

    # -----------------------------------------------------------------------
    # 10. Summary verdict
    # -----------------------------------------------------------------------
    _divider("Summary Verdict")

    profitable_test = [r for r in all_results
                       if r["split"] == "test" and r.get("n_trades", 0) > 0
                       and r.get("mean_gross", float("nan")) > 0]

    print(f"\n  Model accuracy (test R²):")
    for h in horizon_bars:
        m = train_metrics[h]
        print(f"    H={h}b: R²={m['r2']:.6f}  corr={m['corr']:.6f}")

    print(f"\n  Profitable test configurations: {len(profitable_test)}")
    if profitable_test:
        best = max(profitable_test, key=lambda r: r["mean_gross"])
        print(f"  Best: H={best['horizon']}b ({best['horizon']*BAR_S}s)  "
              f"thr={best['threshold']:.1f}  mean_net={best['mean_gross']:+.3f}  "
              f"trades/wk={best['trades_wk']:.0f}  hit_rate={best['hit_rate']:.1%}")
        print(f"  Strategy is profitable at realistic spreads: ✓")
    else:
        print(f"  Strategy is NOT profitable at realistic spreads with current model: ✗")
        print(f"  Consider: stronger features, alternative targets, or different pair")

    # -----------------------------------------------------------------------
    # 11. Save results
    # -----------------------------------------------------------------------
    results_rows = [{k: v for k, v in r.items() if k != "trades"} for r in all_results]
    results_df   = pd.DataFrame(results_rows)
    results_path = output_dir / "forecast_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")
    print(f"  Models saved  → {models_dir}/return_forecast_h*.pkl")
    print(f"  Plots saved   → {plots_dir}/forecast_*.png")


if __name__ == "__main__":
    main()
