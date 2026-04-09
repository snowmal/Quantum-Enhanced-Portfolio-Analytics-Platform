"""
live_optimizer.py
=================
Background process: polls yfinance every POLL_INTERVAL seconds, recomputes
a rolling covariance from the most recent ROLLING_WINDOW trading days
(including today's intraday data), re-runs Markowitz optimization, and
writes live_state.json to the artifacts directory.

The dashboard reads live_state.json on every browser refresh — no WebSocket
needed. Decoupling compute from display keeps the dashboard lightweight.

Usage
-----
    # Run in a separate terminal alongside the dashboard:
    python live_optimizer.py

    # Custom poll interval and risk-aversion:
    python live_optimizer.py --interval 300 --lambda 0.5

    # HPC: run multiple lambda values in parallel:
    python live_optimizer.py --lambda-scan   # scans 5 lambda values via joblib

Architecture
------------
    live_optimizer.py  →  artifacts/live_state.json
    dashboard_server.py reads live_state.json every refresh

HPC Notes
---------
    ROLLING_WINDOW covariance is computed once per poll (fast).
    Monte Carlo paths are handled by projection_engine.py (separate process).
    For parallel lambda scanning on HPC: joblib Parallel with n_jobs=-1
    routes to all available cores automatically.

Dependencies
------------
    pip install yfinance pandas numpy scipy scikit-learn joblib
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("[LIVE] WARNING: yfinance not installed — pip install yfinance")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
ARTIFACTS     = HERE / "portfolio_training" / "artifacts"
LIVE_STATE    = ARTIFACTS / "live_state.json"
LOG_FILE      = ARTIFACTS / "live_optimizer.log"

# ── Defaults ──────────────────────────────────────────────────────────────────
POLL_INTERVAL  = 300          # seconds between refreshes (5 min)
ROLLING_WINDOW = 252          # trading days for covariance estimate
RISK_FREE_RATE = 0.04 / 252   # daily (~4% annual)
DEFAULT_LAMBDA = 0.5          # balanced risk/return
LAMBDA_SCAN    = [0.2, 0.4, 0.5, 0.6, 0.8]   # for HPC parallel scan
MAX_RETRIES    = 3            # yfinance fetch retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LIVE] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ]
)
log = logging.getLogger("live_optimizer")

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_running = True

def _sighandler(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping after current cycle.")
    _running = False

signal.signal(signal.SIGINT,  _sighandler)
signal.signal(signal.SIGTERM, _sighandler)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD STATIC ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_static_artifacts() -> dict:
    """
    Load the tickers and historical returns from the classical pipeline.
    These are the anchor: we extend them with live intraday data.
    """
    returns_path = ARTIFACTS / "returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError(
            f"returns.csv not found at {returns_path}.\n"
            "Run main_classical.py first to generate the historical dataset."
        )

    returns  = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    tickers  = returns.columns.tolist()
    n        = len(tickers)

    # Load static covariance and mu as fallback if live fetch fails
    sigma_path = ARTIFACTS / "sigma_full.npy"
    mu_path    = ARTIFACTS / "mu_annual.npy"

    sigma_static = np.load(sigma_path) if sigma_path.exists() else np.eye(n) * 0.01
    mu_static    = np.load(mu_path)    if mu_path.exists()    else np.zeros(n)

    log.info(f"Static artifacts loaded — {n} assets: {tickers}")
    return {
        "returns":      returns,
        "tickers":      tickers,
        "sigma_static": sigma_static,
        "mu_static":    mu_static / 252,   # daily
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LIVE PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_returns(tickers: list,
                       historical_returns: pd.DataFrame,
                       window: int = ROLLING_WINDOW) -> tuple[pd.DataFrame, dict]:
    """
    Pull the last `window` trading days of data from yfinance (including
    today's session if market is open) and return:
      - extended_returns : DataFrame of daily log returns (window × N)
      - live_prices      : dict of latest price per ticker

    Falls back to historical data alone if yfinance fails.
    """
    if not HAS_YF:
        log.warning("yfinance unavailable — using historical data only")
        recent = historical_returns.iloc[-window:]
        prices = {t: None for t in tickers}
        return recent, prices

    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(
                tickers,
                period=f"{window + 10}d",   # extra buffer for non-trading days
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                raise ValueError("Empty response from yfinance")
            break
        except Exception as e:
            log.warning(f"yfinance attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                log.error("All yfinance retries exhausted — using cached data")
                recent = historical_returns.iloc[-window:]
                prices = {t: None for t in tickers}
                return recent, prices
            time.sleep(2 ** attempt)

    # Extract adjusted close
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) \
                 else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw

    # Align to our ticker list
    closes = closes[[t for t in tickers if t in closes.columns]]
    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        log.warning(f"Missing tickers from yfinance: {missing} — using historical fill")
        for t in missing:
            if t in historical_returns.columns:
                closes[t] = np.exp(historical_returns[t].iloc[-len(closes):].values.cumsum())

    closes = closes[tickers].dropna(how="all").ffill()

    # Log returns
    live_ret = np.log(closes / closes.shift(1)).dropna()
    live_ret.index = pd.to_datetime(live_ret.index)

    # Merge with historical, drop duplicates, keep latest window
    combined = pd.concat([historical_returns, live_ret])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index().iloc[-window:]

    # Latest prices for display
    live_prices = {t: float(closes[t].iloc[-1]) if t in closes.columns else None
                   for t in tickers}

    # Intraday return (today vs yesterday close)
    intraday = {}
    if len(closes) >= 2:
        today_r = (closes.iloc[-1] / closes.iloc[-2] - 1)
        intraday = {t: round(float(today_r[t]) * 100, 3)
                    if t in today_r.index else 0.0 for t in tickers}

    log.info(f"Live data: {len(combined)} days, latest={combined.index[-1].date()}")
    return combined, live_prices, intraday


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MARKOWITZ OPTIMIZATION (mirrors classical_optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _markowitz_obj(w, sigma, mu, lam):
    return float(w @ sigma @ w) - lam * float(w @ mu)

def _markowitz_grad(w, sigma, mu, lam):
    return 2 * sigma @ w - lam * mu

def _optimize_single(sigma: np.ndarray, mu: np.ndarray,
                     lam: float, w0: np.ndarray) -> dict:
    """Single-lambda Markowitz solve. Used directly or via joblib."""
    n      = len(mu)
    bounds = [(0.0, 1.0)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(_markowitz_obj, w0, args=(sigma, mu, lam),
                   jac=_markowitz_grad, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 500})
    w = res.x if res.success else w0
    var    = float(w @ sigma @ w)
    sig_p  = float(np.sqrt(max(var, 0)))
    mu_p   = float(w @ mu)
    sharpe = (mu_p - RISK_FREE_RATE) / (sig_p + 1e-12)
    return {
        "lambda":  lam,
        "weights": w.tolist(),
        "variance": var,
        "sigma_p":  sig_p,
        "mu_p":     mu_p,
        "sharpe":   sharpe,
        "success":  bool(res.success),
        "ann_return": mu_p * 252,
        "ann_vol":    sig_p * np.sqrt(252),
        "ann_sharpe": sharpe * np.sqrt(252),
    }


def optimize_live(sigma: np.ndarray, mu: np.ndarray,
                  lam: float = DEFAULT_LAMBDA,
                  lambda_scan: bool = False,
                  n_jobs: int = -1) -> dict:
    """
    Run Markowitz optimization on live covariance + returns.

    If lambda_scan=True, sweeps LAMBDA_SCAN values in parallel via joblib
    (HPC: set n_jobs=-1 to use all cores, or n_jobs=N for N cores).
    Returns the best Sharpe portfolio plus all frontier points.
    """
    n  = len(mu)
    w0 = np.ones(n) / n

    if not lambda_scan:
        return {"best": _optimize_single(sigma, mu, lam, w0), "frontier": []}

    # HPC parallel lambda scan
    if HAS_JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_optimize_single)(sigma, mu, l, w0) for l in LAMBDA_SCAN
        )
    else:
        results = [_optimize_single(sigma, mu, l, w0) for l in LAMBDA_SCAN]

    best = max(results, key=lambda r: r["sharpe"])
    return {"best": best, "frontier": results}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIVE STATE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_live_state(static: dict,
                       lam: float = DEFAULT_LAMBDA,
                       lambda_scan: bool = False) -> dict:
    """
    Full live cycle:
      1. Fetch latest prices + returns from yfinance
      2. Recompute rolling covariance and expected returns
      3. Re-optimize Markowitz weights
      4. Compute live risk metrics on new weights
      5. Return structured state dict
    """
    tickers = static["tickers"]
    n       = len(tickers)

    # ── Fetch live data ───────────────────────────────────────────────────────
    fetch_result = fetch_live_returns(tickers, static["returns"])
    if len(fetch_result) == 3:
        live_returns, live_prices, intraday = fetch_result
    else:
        live_returns, live_prices = fetch_result
        intraday = {t: 0.0 for t in tickers}

    # ── Covariance + expected returns ─────────────────────────────────────────
    sigma = live_returns.cov().values.astype(np.float64)
    mu    = live_returns.mean().values.astype(np.float64)   # daily

    # Regularise if near-singular (Ledoit-Wolf shrinkage lite)
    cond = np.linalg.cond(sigma)
    if cond > 1e10:
        log.warning(f"Ill-conditioned Σ (cond={cond:.1e}) — applying diagonal shrinkage")
        alpha = 0.05
        sigma = (1 - alpha) * sigma + alpha * np.diag(np.diag(sigma))

    # ── Optimize ──────────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    opt    = optimize_live(sigma, mu, lam, lambda_scan)
    opt_ms = (time.perf_counter() - t0) * 1000
    best   = opt["best"]
    w      = np.array(best["weights"])

    # ── Live risk metrics ─────────────────────────────────────────────────────
    r_p       = live_returns.values @ w
    var_95    = float(np.percentile(r_p, 5))          # historical VaR (negative)
    tail      = r_p[r_p <= var_95]
    es_95     = float(tail.mean()) if len(tail) > 0 else var_95
    max_dd    = _max_drawdown(r_p)
    port_ret_today = float(sum(intraday.get(t, 0) * w[i] / 100
                               for i, t in enumerate(tickers)))

    # MRC / CRC
    sigma_w = sigma @ w
    sig_p   = best["sigma_p"]
    mrc     = (sigma_w / (sig_p + 1e-12)).tolist()
    crc     = (w * np.array(mrc)).tolist()
    crc_pct = [(c / (sig_p + 1e-12)) * 100 for c in crc]

    # Momentum signals (5-day vs 20-day return)
    momentum = {}
    for i, t in enumerate(tickers):
        r_col = live_returns.iloc[:, i]
        m5    = float(r_col.iloc[-5:].sum())   if len(r_col) >= 5  else 0.0
        m20   = float(r_col.iloc[-20:].sum())  if len(r_col) >= 20 else 0.0
        momentum[t] = {"5d": round(m5*100, 3), "20d": round(m20*100, 3),
                        "signal": "bullish" if m5 > m20 else "bearish" if m5 < m20 else "neutral"}

    now = datetime.now()
    state = {
        "timestamp":          now.isoformat(),
        "market_date":        str(live_returns.index[-1].date()),
        "market_open":        _market_is_open(now),
        "tickers":            tickers,
        "live_prices":        live_prices,
        "intraday_return_pct": intraday,
        "portfolio_intraday_pct": round(port_ret_today * 100, 4),
        "optimization": {
            "lambda":       lam,
            "lambda_scan":  lambda_scan,
            "runtime_ms":   round(opt_ms, 2),
            "weights":      {t: round(w[i], 6) for i, t in enumerate(tickers)},
            "ann_return_pct": round(best["ann_return"] * 100, 3),
            "ann_vol_pct":    round(best["ann_vol"]    * 100, 3),
            "ann_sharpe":     round(best["ann_sharpe"],        4),
            "frontier":       opt["frontier"],
        },
        "risk": {
            "variance":       round(best["variance"], 8),
            "sigma_p_daily":  round(best["sigma_p"],  6),
            "sigma_p_annual_pct": round(best["ann_vol"] * 100, 3),
            "var_95_pct":     round(abs(var_95) * 100, 4),
            "es_95_pct":      round(abs(es_95)  * 100, 4),
            "max_drawdown_pct": round(max_dd * 100, 3),
            "cond_number":    round(float(cond), 2),
        },
        "attribution": {
            t: {"weight_pct": round(w[i]*100, 3),
                "MRC":        round(mrc[i], 6),
                "CRC":        round(crc[i], 6),
                "CRC_pct":    round(crc_pct[i], 3)}
            for i, t in enumerate(tickers)
        },
        "momentum": momentum,
        "data_quality": {
            "n_days":       len(live_returns),
            "rolling_window": ROLLING_WINDOW,
            "shrinkage_applied": cond > 1e10,
        },
    }

    return state


def _max_drawdown(r: np.ndarray) -> float:
    cum   = np.cumprod(1 + r)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / (peak + 1e-12)
    return float(dd.min())


def _market_is_open(now: datetime) -> bool:
    """Rough check: NYSE is open Mon-Fri 9:30-16:00 ET."""
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        loc = now.astimezone(et)
    except Exception:
        loc = now   # fallback if pytz not available
    if loc.weekday() >= 5:
        return False
    open_t  = loc.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = loc.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= loc <= close_t


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def write_state(state: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(LIVE_STATE)   # atomic rename — dashboard never reads partial file


def main():
    parser = argparse.ArgumentParser(description="Live portfolio optimizer")
    parser.add_argument("--interval",    type=int,   default=POLL_INTERVAL,
                        help=f"Poll interval in seconds (default: {POLL_INTERVAL})")
    parser.add_argument("--lambda",      type=float, default=DEFAULT_LAMBDA,
                        dest="lam",
                        help=f"Risk-aversion λ (default: {DEFAULT_LAMBDA})")
    parser.add_argument("--lambda-scan", action="store_true",
                        help="Scan multiple λ values in parallel (HPC mode)")
    parser.add_argument("--n-jobs",      type=int,   default=-1,
                        help="Joblib parallel workers (-1 = all cores)")
    parser.add_argument("--once",        action="store_true",
                        help="Run one cycle then exit (useful for testing)")
    args = parser.parse_args()

    log.info("=" * 56)
    log.info("  LIVE PORTFOLIO OPTIMIZER")
    log.info(f"  Poll interval : {args.interval}s")
    log.info(f"  Lambda        : {args.lam}")
    log.info(f"  Lambda scan   : {args.lambda_scan}")
    log.info(f"  HPC n_jobs    : {args.n_jobs}")
    log.info("=" * 56)

    static = load_static_artifacts()
    cycle  = 0

    while _running:
        cycle += 1
        log.info(f"── Cycle {cycle} ──────────────────────────────────────")
        try:
            state = compute_live_state(static, args.lam, args.lambda_scan)
            write_state(state)
            log.info(
                f"  Sharpe={state['optimization']['ann_sharpe']:.3f}  "
                f"Vol={state['optimization']['ann_vol_pct']:.1f}%  "
                f"VaR={state['risk']['var_95_pct']:.2f}%  "
                f"Optimized in {state['optimization']['runtime_ms']:.0f}ms"
            )
        except Exception as e:
            log.error(f"Cycle {cycle} failed: {e}", exc_info=True)
            # Write error state so dashboard shows it
            error_state = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "last_cycle": cycle,
            }
            write_state(error_state)

        if args.once:
            log.info("--once flag set — exiting after first cycle.")
            break

        log.info(f"  Next refresh in {args.interval}s …")
        # Sleep in small chunks so SIGINT is caught promptly
        for _ in range(args.interval):
            if not _running:
                break
            time.sleep(1)

    log.info("Live optimizer stopped cleanly.")


if __name__ == "__main__":
    main()