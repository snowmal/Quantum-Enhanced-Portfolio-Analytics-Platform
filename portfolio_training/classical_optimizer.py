"""
classical_optimizer.py
======================
Markowitz mean-variance optimizer with SLSQP.
Traces the efficient frontier by scanning risk-aversion λ from 0 → 1.
Optional Black-Litterman (Woodbury form) if covariance instability is detected.

Inputs  : artifacts/returns.csv, artifacts/sigma_full.npy, artifacts/mu_annual.npy
Outputs : artifacts/w_classical.csv       — optimal weights per λ
          artifacts/efficient_frontier.csv — Sharpe, return, vol per frontier point
          artifacts/optimizer_report.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv, cond

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RISK_FREE_RATE = 0.02 / 252   # daily risk-free rate ~2% annual
TRADING_DAYS   = 252
LAMBDA_GRID    = np.linspace(0, 1, 51)   # 51 points on the frontier
CONDITION_THRESH = 1e12   # trigger BL if cond(Σ) exceeds this


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── Markowitz Objective ────────────────────────────────────────────────────────

def markowitz_objective(w, sigma, mu, lam):
    """Minimize: w'Σw - λ * w'μ"""
    return float(w @ sigma @ w) - lam * float(w @ mu)


def markowitz_gradient(w, sigma, mu, lam):
    """∇_w [w'Σw - λw'μ] = 2Σw - λμ"""
    return 2 * sigma @ w - lam * mu


def optimize_markowitz(sigma: np.ndarray, mu: np.ndarray, lam: float,
                       long_only: bool = True, w0: np.ndarray = None) -> dict:
    """
    Solve min_w  w'Σw - λw'μ
    subject to   Σw_i = 1  (fully invested)
                 w_i ≥ 0   (long-only, if enabled)
    """
    N    = len(mu)
    w0   = w0 if w0 is not None else np.ones(N) / N

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds      = [(0.0, 1.0)] * N if long_only else [(None, None)] * N

    result = minimize(
        markowitz_objective,
        w0,
        args=(sigma, mu, lam),
        jac=markowitz_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result


def sharpe_ratio(w, sigma, mu, rf=RISK_FREE_RATE):
    """Daily Sharpe: (w'μ - rf) / σ_p"""
    ret  = float(w @ mu) - rf
    vol  = float(np.sqrt(max(w @ sigma @ w, 0)))
    return ret / (vol + 1e-12)


# ── Black-Litterman (Woodbury form) ───────────────────────────────────────────

def black_litterman_returns(sigma: np.ndarray, w_market: np.ndarray,
                             tau: float = 0.05,
                             P: np.ndarray = None,
                             q: np.ndarray = None,
                             omega: np.ndarray = None) -> np.ndarray:
    """
    Black-Litterman posterior expected returns using Woodbury identity.
    Inverts only the K×K view matrix — stable for large N.

    μ_BL = π + τΣP'(PτΣP' + Ω)^{-1}(q - Pπ)

    If no views (P=None), returns equilibrium returns π = δΣw_mkt.
    """
    delta = 2.5   # market risk aversion (typical value)
    pi    = delta * sigma @ w_market   # implied equilibrium returns

    if P is None or q is None:
        return pi

    tau_sigma = tau * sigma
    M         = P @ tau_sigma @ P.T + omega               # K×K — cheap to invert
    try:
        M_inv = inv(M)
    except Exception:
        print("  [WARN] BL: M inversion failed, returning equilibrium returns.")
        return pi

    mu_bl = pi + tau_sigma @ P.T @ M_inv @ (q - P @ pi)
    return mu_bl


# ── Efficient Frontier ─────────────────────────────────────────────────────────

def trace_efficient_frontier(sigma: np.ndarray, mu: np.ndarray,
                              tickers: list,
                              lambda_grid=LAMBDA_GRID,
                              long_only: bool = True) -> pd.DataFrame:
    """
    Sweep λ from 0 (min variance) to 1 (max return) and collect:
    weight vector, Sharpe, annualised return, annualised vol.
    """
    print(f"[OPT] Tracing efficient frontier ({len(lambda_grid)} points) …")
    N   = len(mu)
    w0  = np.ones(N) / N
    records = []

    for lam in lambda_grid:
        res = optimize_markowitz(sigma, mu, lam, long_only, w0)
        if not res.success:
            continue
        w   = res.x
        w0  = w.copy()   # warm-start next iteration

        ann_ret = float(w @ mu) * TRADING_DAYS
        ann_vol = float(np.sqrt(max(w @ sigma @ w, 0))) * np.sqrt(TRADING_DAYS)
        sr      = (ann_ret - RISK_FREE_RATE * TRADING_DAYS) / (ann_vol + 1e-10)

        rec = {
            "lambda":        lam,
            "ann_return":    ann_ret,
            "ann_vol":       ann_vol,
            "sharpe":        sr,
            **{t: float(w[i]) for i, t in enumerate(tickers)},
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"  [OK] Frontier: {len(df)} feasible points, "
          f"max Sharpe: {df['sharpe'].max():.3f}")
    return df


def find_tangency_portfolio(frontier_df: pd.DataFrame,
                             sigma: np.ndarray, mu: np.ndarray,
                             tickers: list) -> np.ndarray:
    """Return the weights of the max-Sharpe (tangency) portfolio."""
    idx  = frontier_df["sharpe"].idxmax()
    row  = frontier_df.loc[idx]
    return np.array([row[t] for t in tickers])


def find_min_variance_portfolio(sigma: np.ndarray, mu: np.ndarray,
                                 tickers: list,
                                 long_only: bool = True) -> np.ndarray:
    """Solve for the global minimum variance portfolio (λ=0)."""
    res = optimize_markowitz(sigma, mu, lam=0.0, long_only=long_only)
    return res.x if res.success else np.ones(len(mu)) / len(mu)


# ── Per-asset weight stability across rolling windows ─────────────────────────

def rolling_optimal_weights(returns: pd.DataFrame, lam: float = 0.5,
                              window: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Re-optimise Markowitz at every window step and track weight evolution.
    Returns a DataFrame (dates x tickers) of optimal weights over time.
    """
    print(f"[OPT] Rolling optimization (λ={lam:.2f}, window={window}) …")
    records = []
    tickers = returns.columns.tolist()
    N       = len(tickers)

    for end_i in range(window, len(returns) + 1):
        r_win     = returns.iloc[end_i - window: end_i]
        sigma_win = r_win.cov().values
        mu_win    = r_win.mean().values

        # Check covariance conditioning; fall back to BL if ill-conditioned
        c = cond(sigma_win)
        if c > CONDITION_THRESH:
            w_mkt  = np.ones(N) / N
            mu_use = black_litterman_returns(sigma_win, w_mkt)
        else:
            mu_use = mu_win

        res = optimize_markowitz(sigma_win, mu_use, lam, long_only=True)
        w   = res.x if res.success else np.ones(N) / N

        rec = {"date": returns.index[end_i - 1]}
        rec.update({t: float(w[i]) for i, t in enumerate(tickers)})
        records.append(rec)

    df = pd.DataFrame(records).set_index("date")
    print(f"  [OK] Rolling weights: {df.shape}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def run_optimizer():
    print("\n" + "="*60)
    print("  MARKOWITZ OPTIMIZER")
    print("="*60 + "\n")

    returns = pd.read_csv(os.path.join(ARTIFACTS_DIR, "returns.csv"),
                          index_col=0, parse_dates=True)
    sigma   = np.load(os.path.join(ARTIFACTS_DIR, "sigma_full.npy"))
    mu_ann  = np.load(os.path.join(ARTIFACTS_DIR, "mu_annual.npy"))
    tickers = returns.columns.tolist()
    N       = len(tickers)

    mu_daily = mu_ann / TRADING_DAYS

    # Check covariance conditioning
    c = cond(sigma)
    print(f"[OPT] Covariance condition number: {c:.2e}")
    if c > CONDITION_THRESH:
        print("  [WARN] Ill-conditioned Σ — applying Black-Litterman shrinkage.")
        mu_use = black_litterman_returns(sigma, np.ones(N) / N)
    else:
        mu_use = mu_daily

    # Efficient frontier
    frontier = trace_efficient_frontier(sigma, mu_use, tickers)
    frontier.to_csv(os.path.join(ARTIFACTS_DIR, "efficient_frontier.csv"), index=False)

    # Key portfolios
    w_tangency  = find_tangency_portfolio(frontier, sigma, mu_use, tickers)
    w_minvar    = find_min_variance_portfolio(sigma, mu_use, tickers)
    w_equal     = np.ones(N) / N

    portfolios = {
        "tangency":    w_tangency.tolist(),
        "min_variance": w_minvar.tolist(),
        "equal_weight": w_equal.tolist(),
    }

    # Save all optimal weight vectors
    w_df = pd.DataFrame(portfolios, index=tickers)
    w_df.to_csv(os.path.join(ARTIFACTS_DIR, "w_classical.csv"))
    print(f"  Saved w_classical.csv")

    # Rolling weights for λ=0.5 (balanced)
    rolling_w = rolling_optimal_weights(returns, lam=0.5)
    rolling_w.to_csv(os.path.join(ARTIFACTS_DIR, "rolling_weights.csv"))

    # Weight stability: std of each asset weight over rolling windows
    weight_stability = rolling_w.std().to_dict()

    def _portfolio_stats(w_vec, label):
        ann_ret = float(w_vec @ mu_use) * TRADING_DAYS
        ann_vol = float(np.sqrt(max(w_vec @ sigma @ w_vec, 0))) * np.sqrt(TRADING_DAYS)
        sr      = (ann_ret - RISK_FREE_RATE * TRADING_DAYS) / (ann_vol + 1e-10)
        return {"label": label, "ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4), "sharpe": round(sr, 4)}

    stats = [
        _portfolio_stats(w_tangency,  "tangency"),
        _portfolio_stats(w_minvar,    "min_variance"),
        _portfolio_stats(w_equal,     "equal_weight"),
    ]

    print("\n[OPT] Key portfolio statistics:")
    for s in stats:
        print(f"  {s['label']:15s}  ret={s['ann_return']*100:.1f}%  "
              f"vol={s['ann_vol']*100:.1f}%  Sharpe={s['sharpe']:.3f}")

    report = {
        "key_portfolios":       stats,
        "frontier_points":      len(frontier),
        "max_sharpe":           float(frontier["sharpe"].max()),
        "condition_number":     float(c),
        "bl_applied":           c > CONDITION_THRESH,
        "weight_stability_std": {k: round(v, 4) for k, v in weight_stability.items()},
    }
    _save_json(report, os.path.join(ARTIFACTS_DIR, "optimizer_report.json"))

    print("\n[OPT] Optimization complete.")
    return frontier, w_tangency, w_minvar, rolling_w


if __name__ == "__main__":
    run_optimizer()