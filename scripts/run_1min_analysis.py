"""
1-Minute Resolution Analysis — EUR/USD/AUD Triangle Strategy

Tests whether moving to the OU timescale (~1.5 min) recovers a tradable signal in
either mean-reversion or predictive mode, with a spread filter to ensure liquid entry.

Sections
--------
A : Mean-reversion simulation (z-score threshold triggers)
B : Predictive model (LGBMRegressor, cross-pair return features)
C : Comparison table vs sub-minute results

Usage
-----
# Quick sanity check
python3 scripts/run_1min_analysis.py --mode mr --thresholds 2.5 --horizons 2

# Full run
python3 scripts/run_1min_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"

sys.path.insert(0, str(ROOT / "src"))
from triangulation.data     import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import compute_session_features
from triangulation.labels   import split_by_date

TRAIN_END   = "2024-12-31"
VAL_END     = "2025-06-30"
BAR_S       = 60          # 1-minute bars
SPREAD_MAX  = 1.5         # pips — entry spread filter


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10s OHLCV+bid/ask bars to 1-minute bars.

    Uses the last tick of each minute for close/bid/ask so that execution
    prices reflect the price at signal-observation time.
    """
    return (
        df.resample("1min")
        .agg(
            open  = ("open",  "first"),
            high  = ("high",  "max"),
            low   = ("low",   "min"),
            close = ("close", "last"),
            bid   = ("bid",   "last"),
            ask   = ("ask",   "last"),
        )
        .dropna()
    )


# ---------------------------------------------------------------------------
# Feature construction (1-min scale, computed inline)
# ---------------------------------------------------------------------------

def build_features_1min(sig: pd.DataFrame) -> pd.DataFrame:
    """Build feature frame at 1-minute resolution.

    Args:
        sig: Output of build_signal_frame (columns: eurusd, audusd, euraud,
             euraud_bid, euraud_ask, residual, zscore).

    Returns:
        DataFrame of ~25 features, NaN rows dropped.
    """
    feats: dict[str, pd.Series] = {}

    # Core signal
    feats["zscore"]   = sig["zscore"]
    feats["residual"] = sig["residual"]
    feats["dz_1m"]    = sig["zscore"].diff()

    # Cross-pair returns at 1m, 5m, 10m lookbacks
    for pair in ["eurusd", "audusd", "euraud"]:
        px = sig[pair]
        for k in [1, 5, 10]:
            feats[f"ret_{pair}_{k}m"] = (px - px.shift(k)) * 10_000

    # Cross-pair lead indicators: EURUSD/AUDUSD momentum relative to EURAUD
    feats["lead_eu_aud_1m"] = feats["ret_eurusd_1m"] - feats["ret_euraud_1m"]
    feats["lead_au_aud_1m"] = feats["ret_audusd_1m"] - feats["ret_euraud_1m"]
    feats["lead_eu_aud_5m"] = feats["ret_eurusd_5m"] - feats["ret_euraud_5m"]

    # Momentum contrast: short-term vs longer-term
    for pair in ["eurusd", "audusd", "euraud"]:
        feats[f"mom_{pair}"] = feats[f"ret_{pair}_1m"] - feats[f"ret_{pair}_10m"]

    # Realised vol of EURAUD
    diff_ea = sig["euraud"].diff() * 10_000
    feats["rv_euraud_5m"]  = diff_ea.abs().rolling(5,  min_periods=3).mean()
    feats["rv_euraud_60m"] = diff_ea.abs().rolling(60, min_periods=30).mean()

    # Spread in pips — used both as feature and entry filter
    feats["spread_pips"] = (sig["euraud_ask"] - sig["euraud_bid"]) * 10_000

    # Session
    sess = compute_session_features(sig.index)
    for c in ["hour_sin", "hour_cos", "is_london_ny"]:
        feats[c] = sess[c]

    return pd.DataFrame(feats, index=sig.index).dropna()


# ---------------------------------------------------------------------------
# Section A: Mean-reversion simulation
# ---------------------------------------------------------------------------

def simulate_mr(
    z: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    spread_pips: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    horizon_bars: int,
    spread_max: float = SPREAD_MAX,
) -> pd.DataFrame:
    """No-overlap mean-reversion simulation with spread filter.

    Entry: next bar after z-threshold crossing (direction = -sign(z)).
    Skip if spread_pips at entry bar > spread_max.
    Exit conditions (first wins):
      1. Reversal stop: |z| > 1.5 × |z_entry| AND same sign (gap widened)
      2. Time: horizon_bars elapsed

    Returns:
        trade_log DataFrame with columns ts, direction, entry_z, gross, spread_e.
    """
    n         = len(z)
    trades    = []
    next_free = 0

    for i in range(n):
        if i < next_free:
            continue
        if abs(z[i]) < threshold:
            continue

        entry_i = i + 1         # execute on the bar after signal
        if entry_i >= n - 1:
            continue
        if entry_i < next_free:
            continue
        if spread_pips[entry_i] > spread_max:
            continue

        direction  = -float(np.sign(z[i]))   # mean-revert: short when z>0
        entry_z    = float(z[entry_i])
        entry_exec = ask[entry_i] if direction > 0 else bid[entry_i]

        # Scan horizon for exit
        end_i    = min(entry_i + horizon_bars, n - 1)
        exit_off = horizon_bars   # default: time exit

        for j in range(entry_i + 1, end_i + 1):
            zj = z[j]
            # Reversal stop: gap widened to 1.5× entry z (same sign means worse)
            if abs(zj) > 1.5 * abs(entry_z) and np.sign(zj) == np.sign(entry_z):
                exit_off = j - entry_i
                break

        exit_i    = entry_i + exit_off
        exit_exec = bid[exit_i] if direction > 0 else ask[exit_i]
        gross     = direction * (exit_exec - entry_exec) * 10_000
        spread_e  = spread_pips[entry_i]

        trades.append({
            "ts":        timestamps[entry_i],
            "direction": direction,
            "entry_z":   entry_z,
            "gross":     gross,
            "spread_e":  spread_e,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


def run_mr_sweep(
    z: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    spread_pips: np.ndarray,
    timestamps: pd.DatetimeIndex,
    thresholds: list[float],
    horizons: list[int],
    n_days: float,
    spread_max: float,
) -> list[dict]:
    """Sweep all (threshold × horizon) combinations. Returns list of result dicts."""
    results = []
    for thr in thresholds:
        for h in horizons:
            tl = simulate_mr(z, mid, bid, ask, spread_pips, timestamps, thr, h, spread_max)
            if len(tl) == 0:
                results.append({
                    "threshold": thr, "horizon_m": h,
                    "n_trades": 0, "trades_wk": 0.0,
                    "mean_net": float("nan"), "hit_rate": float("nan"),
                    "sharpe": float("nan"), "spread_med": float("nan"),
                    "trades": tl,
                })
                continue
            n_weeks  = n_days / 7
            mean_g   = float(tl["gross"].mean())
            std_g    = float(tl["gross"].std() + 1e-10)
            results.append({
                "threshold": thr,
                "horizon_m": h,
                "n_trades":  len(tl),
                "trades_wk": len(tl) / max(n_weeks, 1e-6),
                "mean_net":  mean_g,
                "hit_rate":  float((tl["gross"] > 0).mean()),
                "sharpe":    mean_g / std_g,
                "spread_med": float(tl["spread_e"].median()),
                "trades":    tl,
            })
    return results


# ---------------------------------------------------------------------------
# Section B: Predictive model
# ---------------------------------------------------------------------------

def build_targets(euraud: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Forward EURAUD returns in pips for each horizon (in bars)."""
    tgt = {}
    for h in horizons:
        tgt[f"target_{h}m"] = (euraud.shift(-h) - euraud) * 10_000
    return pd.DataFrame(tgt, index=euraud.index)


def evaluate_pred_threshold(
    y_pred: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    spread_pips: np.ndarray,
    timestamps: pd.DatetimeIndex,
    horizon_bars: int,
    threshold: float,
    n_days: float,
    spread_max: float,
) -> dict:
    """No-overlap predictive simulation with bid/ask execution and spread filter."""
    n         = len(mid)
    trades    = []
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
        if spread_pips[entry_i] > spread_max:
            continue

        direction  = 1.0 if yhat > 0 else -1.0
        entry_exec = ask[entry_i] if direction > 0 else bid[entry_i]
        exit_exec  = bid[exit_i]  if direction > 0 else ask[exit_i]
        gross      = direction * (exit_exec - entry_exec) * 10_000
        spread_e   = spread_pips[entry_i]

        trades.append({
            "ts":       timestamps[entry_i],
            "yhat":     yhat,
            "gross":    gross,
            "spread_e": spread_e,
        })
        next_free = exit_i + 1

    if not trades:
        return {
            "n_trades": 0, "trades_wk": 0.0,
            "mean_gross": float("nan"), "sharpe": float("nan"),
            "hit_rate": float("nan"), "spread_med": float("nan"),
            "trades": pd.DataFrame(),
        }
    tl      = pd.DataFrame(trades)
    n_weeks = n_days / 7
    mean_g  = float(tl["gross"].mean())
    std_g   = float(tl["gross"].std() + 1e-10)
    return {
        "n_trades":   len(tl),
        "trades_wk":  len(tl) / max(n_weeks, 1e-6),
        "mean_gross": mean_g,
        "sharpe":     mean_g / std_g,
        "hit_rate":   float((tl["gross"] > 0).mean()),
        "spread_med": float(tl["spread_e"].median()),
        "trades":     tl,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mean_net_vs_horizon(
    mr_results: list[dict],
    thresholds: list[float],
    horizons: list[int],
    path: Path,
) -> None:
    """mean_net vs horizon, one line per z-threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#4fc3f7", "#81c784", "#ffb74d"]

    for i, thr in enumerate(thresholds):
        rows = [r for r in mr_results if r["threshold"] == thr]
        xs   = [r["horizon_m"] for r in rows]
        ys   = [r["mean_net"]  for r in rows]
        ax.plot(xs, ys, marker="o", color=colors[i % len(colors)],
                linewidth=1.8, label=f"z≥{thr}")

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Holding horizon (minutes)", fontsize=9)
    ax.set_ylabel("Mean net pips (bid/ask, spread<1.5p)", fontsize=9)
    ax.set_title("Mean-Reversion: mean_net vs horizon  (test set, 1-min bars)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_signal_vs_spread(
    z_all: np.ndarray,
    spread_all: np.ndarray,
    is_london: np.ndarray,
    spread_max: float,
    path: Path,
) -> None:
    """Scatter of |z| vs spread and histogram of spread at signal bars vs all bars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter |z| vs spread, colour by session
    ax = axes[0]
    mask_liq = spread_all < 5.0   # cap spread for readability
    colours  = np.where(is_london, "#81c784", "#4fc3f7")
    ax.scatter(
        np.abs(z_all[mask_liq]),
        spread_all[mask_liq],
        c=colours[mask_liq],
        alpha=0.15, s=1, rasterized=True,
    )
    ax.axhline(spread_max, color="#ffb74d", linewidth=1.5,
               linestyle="--", label=f"spread_max={spread_max}p")
    ax.set_xlabel("|z-score|", fontsize=9)
    ax.set_ylabel("Spread (pips)", fontsize=9)
    ax.set_title("Signal strength vs spread (green=London/NY, blue=other)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.1)
    # Proxy legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#81c784", label="London/NY overlap"),
        Patch(color="#4fc3f7", label="Other session"),
        plt.Line2D([0], [0], color="#ffb74d", linestyle="--",
                   label=f"spread_max={spread_max}p"),
    ], fontsize=8)

    # Right: histogram of spread at signal bars (z>=2.0) vs all bars
    ax2    = axes[1]
    sig20  = spread_all[np.abs(z_all) >= 2.0]
    bins   = np.linspace(0, 5, 51)
    ax2.hist(spread_all, bins=bins, alpha=0.5, color="#4fc3f7",
             density=True, label="All bars")
    ax2.hist(sig20, bins=bins, alpha=0.7, color="#ffb74d",
             density=True, label="Signal bars (|z|≥2.0)")
    ax2.axvline(spread_max, color="#f48fb1", linewidth=1.5,
                linestyle="--", label=f"Filter cutoff ({spread_max}p)")
    ax2.set_xlabel("Spread (pips)", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Spread distribution: all bars vs signal bars", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.1)

    fig.suptitle("Signal Quality at 1-Minute Resolution", fontsize=11, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_pnl_over_time(
    lines: list[dict],   # each: {"label": str, "trades": pd.DataFrame, "color": str}
    path: Path,
) -> None:
    """Cumulative net pips vs time, one line per strategy/config."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for cfg in lines:
        tl = cfg["trades"]
        if len(tl) == 0:
            continue
        tl_s  = tl.sort_values("ts")
        cumul = tl_s["gross"].cumsum()
        ax.plot(tl_s["ts"], cumul, color=cfg["color"],
                linewidth=1.5, label=cfg["label"])

    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative net pips", fontsize=9)
    ax.set_title("P&L over time — test set (actual bid/ask, spread filter)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="both",
                        choices=["mr", "pred", "both"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[2.0, 2.5, 3.0])
    parser.add_argument("--horizons",   nargs="+", type=int,
                        default=[1, 2, 5, 10],
                        help="Holding horizons in minutes (MR mode)")
    parser.add_argument("--pred-horizons", nargs="+", type=int,
                        default=[2, 5, 10],
                        help="Target horizons in minutes (predictive mode)")
    parser.add_argument("--spread-max", type=float, default=SPREAD_MAX)
    args = parser.parse_args()

    SPREAD_FILTER = args.spread_max

    # ------------------------------------------------------------------
    # 1. Load and resample
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading 10s data...")
    eurusd_10s = load_pair(DATA_DIR, "EURUSD")
    audusd_10s = load_pair(DATA_DIR, "AUDUSD")
    euraud_10s = load_pair(DATA_DIR, "EURAUD")

    n10s = len(eurusd_10s)
    print(f"  10s bars loaded: {n10s:,} per pair")

    print("Resampling to 1-minute...")
    eurusd = resample_to_1min(eurusd_10s)
    audusd = resample_to_1min(audusd_10s)
    euraud = resample_to_1min(euraud_10s)

    n1m = len(eurusd)
    print(f"  1-min bars:      {n1m:,}  (ratio {n1m/n10s:.3f}, expected ~{1/6:.3f})")

    # ------------------------------------------------------------------
    # 2. Signal frame and features
    # ------------------------------------------------------------------
    print("Building signal frame (ewma_halflife=60 bars = 1 hour)...")
    sig = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=60)

    print("Building 1-min features...")
    features = build_features_1min(sig)

    spread_all = features["spread_pips"].values
    z_all      = features["zscore"].values
    is_london  = features["is_london_ny"].values.astype(bool)

    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Feature bars:    {len(features):,}")
    print(f"  Date range:      {features.index[0].date()} → {features.index[-1].date()}")
    print(f"  Spread (pips):   median={np.median(spread_all):.3f}  "
          f"p90={np.percentile(spread_all, 90):.3f}  "
          f"mean={spread_all.mean():.3f}")

    total_weeks = (features.index[-1] - features.index[0]).days / 7
    for thr in sorted(set([2.0] + args.thresholds)):
        sig_mask    = np.abs(z_all) >= thr
        n_sig       = sig_mask.sum()
        n_liq       = (sig_mask & (spread_all < SPREAD_FILTER)).sum()
        pct         = 100 * n_liq / n_sig if n_sig > 0 else 0
        print(f"  z≥{thr:.1f}: {n_sig:,} signal bars ({n_sig/total_weeks:.1f}/wk)  "
              f"→ {n_liq:,} liquid ({pct:.1f}% pass spread<{SPREAD_FILTER}p)")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    train, val, test = split_by_date(
        features, TRAIN_END, VAL_END, buffer_bars=max(args.pred_horizons)
    )
    n_train_days = (train.index[-1] - train.index[0]).days
    n_val_days   = (val.index[-1]   - val.index[0]).days
    n_test_days  = (test.index[-1]  - test.index[0]).days
    print(f"\n  Train: {len(train):,} bars ({n_train_days} days)")
    print(f"  Val:   {len(val):,} bars ({n_val_days} days)")
    print(f"  Test:  {len(test):,} bars ({n_test_days} days)")

    # Convenience arrays for test set
    test_z   = test["zscore"].values
    test_mid = sig.loc[test.index, "euraud"].values
    test_bid = sig.loc[test.index, "euraud_bid"].values
    test_ask = sig.loc[test.index, "euraud_ask"].values
    test_sp  = test["spread_pips"].values
    test_ts  = test.index

    mr_results_test: list[dict] = []
    mr_results_val:  list[dict] = []

    # ------------------------------------------------------------------
    # Section A: Mean-reversion
    # ------------------------------------------------------------------
    if args.mode in ("mr", "both"):
        print()
        print("=" * 60)
        print("SECTION A — MEAN-REVERSION (1-min bars, spread filter)")
        print("=" * 60)

        # Val set arrays
        val_z   = val["zscore"].values
        val_mid = sig.loc[val.index, "euraud"].values
        val_bid = sig.loc[val.index, "euraud_bid"].values
        val_ask = sig.loc[val.index, "euraud_ask"].values
        val_sp  = val["spread_pips"].values
        val_ts  = val.index
        n_val_days_f = float(n_val_days)

        mr_results_val = run_mr_sweep(
            val_z, val_mid, val_bid, val_ask, val_sp, val_ts,
            args.thresholds, args.horizons, n_val_days_f, SPREAD_FILTER,
        )

        mr_results_test = run_mr_sweep(
            test_z, test_mid, test_bid, test_ask, test_sp, test_ts,
            args.thresholds, args.horizons, float(n_test_days), SPREAD_FILTER,
        )

        # Print table
        hdr = (f"{'threshold':>10} {'horizon_m':>10} {'n_trades':>9} "
               f"{'trades_wk':>10} {'mean_net':>10} {'hit_rate':>9} "
               f"{'sharpe':>8} {'spread_med':>11}")
        print(f"\n  Test set results:")
        print(f"  {hdr}")
        print(f"  {'-'*len(hdr)}")
        for r in mr_results_test:
            print(
                f"  {r['threshold']:>10.1f} {r['horizon_m']:>10d} "
                f"{r['n_trades']:>9,} {r['trades_wk']:>10.1f} "
                f"{r['mean_net']:>10.3f} {r['hit_rate']:>9.1%} "
                f"{r['sharpe']:>8.3f} {r['spread_med']:>10.3f}p"
            )

    # ------------------------------------------------------------------
    # Section B: Predictive model
    # ------------------------------------------------------------------
    pred_results:  list[dict] = []
    best_pred_trades: dict[int, pd.DataFrame] = {}

    if args.mode in ("pred", "both"):
        print()
        print("=" * 60)
        print("SECTION B — PREDICTIVE MODEL (LGBMRegressor, 1-min bars)")
        print("=" * 60)

        # Attach targets to features and re-split (with per-target buffer)
        pred_thresholds = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        feature_cols    = [c for c in features.columns if c != "spread_pips"]

        for h in args.pred_horizons:
            col = f"target_{h}m"
            feats_with_target = features.copy()
            feats_with_target[col] = (
                sig.loc[features.index, "euraud"].shift(-h) - sig.loc[features.index, "euraud"]
            ) * 10_000

            tr, va, te = split_by_date(
                feats_with_target, TRAIN_END, VAL_END, buffer_bars=h
            )
            tr = tr.dropna(subset=[col])
            va = va.dropna(subset=[col])
            te = te.dropna(subset=[col])

            X_tr = tr[feature_cols].values
            y_tr = tr[col].values
            X_va = va[feature_cols].values
            y_va = va[col].values
            X_te = te[feature_cols].values
            y_te = te[col].values

            model = lgb.LGBMRegressor(
                objective       = "regression",
                n_estimators    = 300,
                num_leaves      = 31,
                learning_rate   = 0.05,
                subsample       = 0.8,
                colsample_bytree= 0.8,
                n_jobs          = -1,
                random_state    = 42,
                verbose         = -1,
            )
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])

            y_pred_te = model.predict(X_te)
            r2        = float(r2_score(y_te, y_pred_te))
            print(f"\n  H={h}m  R²={r2:.4f}  corr={float(np.corrcoef(y_pred_te, y_te)[0,1]):.4f}")

            # Threshold selection on val set
            y_pred_va = model.predict(X_va)
            va_bid    = sig.loc[va.index, "euraud_bid"].values
            va_ask    = sig.loc[va.index, "euraud_ask"].values
            va_sp     = va["spread_pips"].values
            n_va_days = (va.index[-1] - va.index[0]).days

            best_thr  = None
            best_val  = -np.inf
            for thr in pred_thresholds:
                res = evaluate_pred_threshold(
                    y_pred_va,
                    sig.loc[va.index, "euraud"].values,
                    va_bid, va_ask, va_sp, va.index,
                    h, thr, float(n_va_days), SPREAD_FILTER,
                )
                if res["n_trades"] > 0 and res["mean_gross"] > best_val:
                    best_val = res["mean_gross"]
                    best_thr = thr

            if best_thr is None:
                best_thr = pred_thresholds[0]

            print(f"  Optimal threshold (val): {best_thr}p  (mean_net={best_val:.3f}p)")

            # Evaluate on test set across all thresholds
            te_bid  = sig.loc[te.index, "euraud_bid"].values
            te_ask  = sig.loc[te.index, "euraud_ask"].values
            te_sp   = te["spread_pips"].values
            te_mid  = sig.loc[te.index, "euraud"].values
            n_te_d  = float((te.index[-1] - te.index[0]).days)

            print(f"\n  {'threshold':>10} {'n_trades':>9} {'trades_wk':>10} "
                  f"{'mean_net':>10} {'hit_rate':>9} {'sharpe':>8} {'spread_med':>11}")
            for thr in pred_thresholds:
                res = evaluate_pred_threshold(
                    y_pred_te, te_mid, te_bid, te_ask, te_sp, te.index,
                    h, thr, n_te_d, SPREAD_FILTER,
                )
                opt_flag = " ← opt" if thr == best_thr else ""
                print(
                    f"  {thr:>10.1f} {res['n_trades']:>9,} {res['trades_wk']:>10.1f} "
                    f"{res['mean_gross']:>10.3f} {res['hit_rate']:>9.1%} "
                    f"{res['sharpe']:>8.3f} {res['spread_med']:>10.3f}p{opt_flag}"
                )
                pred_results.append({
                    "horizon_m": h, "threshold": thr,
                    "r2": r2, **{k: v for k, v in res.items() if k != "trades"},
                })
                if thr == best_thr:
                    best_pred_trades[h] = res["trades"]

    # ------------------------------------------------------------------
    # Section C: Comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SECTION C — COMPARISON: SUB-MINUTE vs 1-MINUTE")
    print("=" * 60)
    print(f"""
  ┌──────────────┬───────────────────────────────┬───────────┬──────────┬────────────┬────────────┐
  │ Resolution   │ Strategy                      │ mean_net  │ hit_rate │ spread_med │ trades/wk  │
  ├──────────────┼───────────────────────────────┼───────────┼──────────┼────────────┼────────────┤
  │ 10s          │ Mean-rev (mid-price)          │  +7.1p    │  64.2%   │  20.7p     │    7       │
  │ 10s          │ Mean-rev (bid/ask)            │ -19.5p    │    —     │  20.7p     │    7       │
  │ 10s          │ Return forecast (best)        │  -5.0p    │  31.0%   │   1.3p     │  118       │""")

    # Find best 1-min MR result
    if mr_results_test:
        best_mr = max(
            (r for r in mr_results_test if r["n_trades"] > 0),
            key=lambda r: r["mean_net"],
            default=None,
        )
        if best_mr:
            print(
                f"  │ 1-min (ours) │ Mean-rev (spread<{SPREAD_FILTER}p)"
                f" thr={best_mr['threshold']},h={best_mr['horizon_m']}m │"
                f" {best_mr['mean_net']:>+6.2f}p  │"
                f" {best_mr['hit_rate']:>6.1%}   │"
                f"  {best_mr['spread_med']:>5.2f}p    │"
                f"   {best_mr['trades_wk']:>5.1f}      │"
            )

    if pred_results:
        best_pred = max(
            (r for r in pred_results if r["n_trades"] > 0),
            key=lambda r: r["mean_gross"],
            default=None,
        )
        if best_pred:
            print(
                f"  │ 1-min (ours) │ Return forecast h={best_pred['horizon_m']}m,"
                f" thr={best_pred['threshold']}p  │"
                f" {best_pred['mean_gross']:>+6.2f}p  │"
                f" {best_pred['hit_rate']:>6.1%}   │"
                f"  {best_pred['spread_med']:>5.2f}p    │"
                f"   {best_pred['trades_wk']:>5.1f}      │"
            )
    print("  └──────────────┴───────────────────────────────┴───────────┴──────────┴────────────┴────────────┘")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    profitable_mr   = [r for r in mr_results_test   if r.get("mean_net", -np.inf) > 0 and r["n_trades"] > 0]
    profitable_pred = [r for r in pred_results if r.get("mean_gross", -np.inf) > 0 and r["n_trades"] > 0]

    if profitable_mr:
        best = max(profitable_mr, key=lambda r: r["mean_net"])
        print(f"  ✓ Mean-reversion PROFITABLE: best config thr={best['threshold']}, "
              f"h={best['horizon_m']}m → {best['mean_net']:+.3f}p/trade, "
              f"{best['trades_wk']:.1f}/wk")
    else:
        print("  ✗ Mean-reversion NOT profitable at realistic spreads")

    if profitable_pred:
        best = max(profitable_pred, key=lambda r: r["mean_gross"])
        print(f"  ✓ Predictive model PROFITABLE: best config h={best['horizon_m']}m, "
              f"thr={best['threshold']}p → {best['mean_gross']:+.3f}p/trade, "
              f"{best['trades_wk']:.1f}/wk")
    else:
        print("  ✗ Predictive model NOT profitable at realistic spreads")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    if mr_results_test:
        plot_mean_net_vs_horizon(
            mr_results_test, args.thresholds, args.horizons,
            PLOT_DIR / "1min_mean_net_vs_horizon.png",
        )

    plot_signal_vs_spread(
        features["zscore"].values,
        features["spread_pips"].values,
        features["is_london_ny"].values.astype(bool),
        SPREAD_FILTER,
        PLOT_DIR / "1min_signal_vs_spread.png",
    )

    # Collect P&L lines for best configs
    pnl_lines = []
    if mr_results_test:
        mr_sorted = sorted(
            [r for r in mr_results_test if r["n_trades"] > 0],
            key=lambda r: r["mean_net"], reverse=True,
        )
        for i, r in enumerate(mr_sorted[:2]):
            lbl = f"MR z≥{r['threshold']}, h={r['horizon_m']}m ({r['mean_net']:+.2f}p)"
            pnl_lines.append({
                "label":  lbl,
                "trades": r["trades"],
                "color":  ["#4fc3f7", "#81c784"][i],
            })
    for i, (h, tl) in enumerate(best_pred_trades.items()):
        if len(tl) > 0:
            pnl_lines.append({
                "label":  f"Pred h={h}m",
                "trades": tl,
                "color":  ["#ffb74d", "#f48fb1"][i % 2],
            })

    if pnl_lines:
        plot_pnl_over_time(pnl_lines, PLOT_DIR / "1min_pnl_over_time.png")

    # ------------------------------------------------------------------
    # Save results CSV
    # ------------------------------------------------------------------
    rows = []
    for r in mr_results_test:
        rows.append({
            "mode": "mr", "horizon_m": r["horizon_m"],
            "threshold": r["threshold"], "n_trades": r["n_trades"],
            "trades_wk": r["trades_wk"], "mean_net": r["mean_net"],
            "hit_rate": r["hit_rate"], "sharpe": r["sharpe"],
            "spread_med": r["spread_med"],
        })
    for r in pred_results:
        rows.append({
            "mode": "pred", "horizon_m": r["horizon_m"],
            "threshold": r["threshold"], "n_trades": r["n_trades"],
            "trades_wk": r["trades_wk"], "mean_net": r.get("mean_gross"),
            "hit_rate": r["hit_rate"], "sharpe": r["sharpe"],
            "spread_med": r["spread_med"],
        })
    if rows:
        out_csv = OUT_DIR / "1min_results.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  Results saved → {out_csv}")


if __name__ == "__main__":
    main()
