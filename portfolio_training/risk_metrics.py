"""
risk_metrics.py
===============
Parametric (Gaussian / FHE-compatible) and Historical VaR & ES.
Also implements the two sqrt approximation strategies needed for FHE later:
  (a) Taylor expansion around expected variance
  (b) Chebyshev polynomial fit on [0, max_var]

Inputs  : artifacts/returns.csv, artifacts/sigma_full.npy, artifacts/mu_annual.npy
Outputs : artifacts/risk_metrics_report.json
          artifacts/sqrt_approx_coeffs.json   — Chebyshev coeffs for FHE
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import chebyt
import numpy.polynomial.chebyshev as C

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
ALPHA         = 0.05    # 95% confidence level  (VaR_0.05 covers 95% of days)
TRADING_DAYS  = 252


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── Portfolio return helpers ───────────────────────────────────────────────────

def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """R_p(t) = Σ w_i * r_i(t)"""
    return returns @ w


def portfolio_mean(returns: pd.DataFrame, w: np.ndarray) -> float:
    return float(returns.mean().values @ w)


def portfolio_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(float(w @ sigma @ w), 0)))


# ── Parametric VaR & ES (Gaussian) ────────────────────────────────────────────

def parametric_var(w: np.ndarray, sigma: np.ndarray, mu: float,
                   alpha: float = ALPHA, horizon: int = 1) -> float:
    """
    VaR_α = -(μ_p*h + z_α * σ_p * √h)
    Positive value = loss threshold (convention: VaR is reported as positive loss).
    """
    sig_p  = portfolio_vol(w, sigma) * np.sqrt(horizon)
    mu_p   = mu * horizon
    z_alpha = norm.ppf(alpha)
    return float(-(mu_p + z_alpha * sig_p))


def parametric_es(w: np.ndarray, sigma: np.ndarray, mu: float,
                  alpha: float = ALPHA, horizon: int = 1) -> float:
    """
    ES_α = -(μ_p*h - σ_p*√h * φ(z_α)/(α))
    where φ is the standard normal PDF.
    """
    sig_p   = portfolio_vol(w, sigma) * np.sqrt(horizon)
    mu_p    = mu * horizon
    z_alpha = norm.ppf(alpha)
    phi     = norm.pdf(z_alpha)
    return float(-(mu_p - sig_p * phi / alpha))


# ── Historical (Non-parametric) VaR & ES ──────────────────────────────────────

def historical_var(r_p: pd.Series, alpha: float = ALPHA) -> float:
    """VaR_α = -Quantile_α(R_p)"""
    return float(-np.quantile(r_p.dropna(), alpha))


def historical_es(r_p: pd.Series, alpha: float = ALPHA) -> float:
    """ES_α = -E[R_p | R_p ≤ -VaR_α]"""
    var = historical_var(r_p, alpha)
    tail = r_p[r_p <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ── Sqrt Approximation (for FHE later) ────────────────────────────────────────

def taylor_sqrt_approx(x: float, mu_var: float) -> float:
    """
    1st-order Taylor of √x around μ_var:
        √x ≈ √μ + (x - μ) / (2√μ)
    """
    sq = np.sqrt(max(mu_var, 1e-10))
    return sq + (x - mu_var) / (2 * sq)


def fit_chebyshev_sqrt(max_var: float, degree: int = 5, n_points: int = 500):
    """
    Fit a Chebyshev polynomial of given degree to √x on [0, max_var].
    Returns coefficients in the monomial basis for easy FHE evaluation.
    """
    # Sample points in [ε, max_var] to avoid √0 edge
    x_sample = np.linspace(1e-6, max_var, n_points)
    y_sample  = np.sqrt(x_sample)

    # Map x from [0, max_var] → [-1, 1] for Chebyshev domain
    # t = 2x/max_var - 1
    t_sample  = 2 * x_sample / max_var - 1
    cheb_coeffs = C.chebfit(t_sample, y_sample, deg=degree)

    # Evaluate approximation quality
    y_hat = C.chebval(t_sample, cheb_coeffs)
    rmse  = float(np.sqrt(np.mean((y_hat - y_sample) ** 2)))
    max_err = float(np.max(np.abs(y_hat - y_sample)))

    return {
        "chebyshev_coeffs": cheb_coeffs.tolist(),
        "degree":    degree,
        "max_var":   max_var,
        "rmse":      rmse,
        "max_error": max_err,
    }


def eval_chebyshev_sqrt(x: float, coeffs_dict: dict) -> float:
    """Evaluate the fitted Chebyshev approximation at x."""
    max_var = coeffs_dict["max_var"]
    t = 2 * x / max_var - 1
    return float(C.chebval(t, coeffs_dict["chebyshev_coeffs"]))


# ── Rolling VaR Coverage Test (Kupiec) ────────────────────────────────────────

def var_coverage_test(r_p: pd.Series, var_series: pd.Series,
                      alpha: float = ALPHA) -> dict:
    """
    Check what fraction of days the actual loss exceeds VaR prediction.
    Target: fraction ≈ alpha (e.g. 5% for 95% VaR).
    """
    common  = r_p.index.intersection(var_series.index)
    r_align = r_p.loc[common]
    v_align = var_series.loc[common]

    breaches      = ((-r_align) > v_align).sum()
    breach_rate   = float(breaches / len(common))
    target        = alpha
    kupiec_ratio  = round(breach_rate / target, 3) if target > 0 else None

    return {
        "n_observations":  len(common),
        "n_breaches":      int(breaches),
        "breach_rate":     round(breach_rate, 4),
        "target_rate":     alpha,
        "kupiec_ratio":    kupiec_ratio,   # ~1.0 = well-calibrated; >1.5 = under-estimating risk
        "assessment":      "well-calibrated" if abs(breach_rate - alpha) < 0.02
                           else ("conservative" if breach_rate < alpha else "under-estimating"),
    }


# ── Risk Metrics over rolling windows ─────────────────────────────────────────

def compute_rolling_risk(returns: pd.DataFrame, w: np.ndarray,
                         window: int = TRADING_DAYS,
                         alpha: float = ALPHA) -> pd.DataFrame:
    """
    Compute rolling VaR, ES, volatility, and Sharpe for a fixed weight vector.
    Returns a DataFrame indexed by date.
    """
    r_p = portfolio_returns(returns, w)
    records = []

    for end_i in range(window, len(returns) + 1):
        r_win   = r_p.iloc[end_i - window: end_i]
        sigma_w = returns.iloc[end_i - window: end_i].cov().values
        mu_w    = float(returns.iloc[end_i - window: end_i].mean().values @ w)
        sig_p   = portfolio_vol(w, sigma_w)

        records.append({
            "date":          returns.index[end_i - 1],
            "rolling_vol":   sig_p * np.sqrt(TRADING_DAYS),
            "hist_var_95":   historical_var(r_win, alpha),
            "hist_es_95":    historical_es(r_win, alpha),
            "param_var_95":  parametric_var(w, sigma_w, mu_w, alpha),
            "param_es_95":   parametric_es(w, sigma_w, mu_w, alpha),
            "rolling_sharpe": (r_win.mean() * TRADING_DAYS) / (sig_p * np.sqrt(TRADING_DAYS) + 1e-10),
        })

    return pd.DataFrame(records).set_index("date")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_risk_metrics():
    print("\n" + "="*60)
    print("  RISK METRICS")
    print("="*60 + "\n")

    returns_path = os.path.join(ARTIFACTS_DIR, "returns.csv")
    sigma_path   = os.path.join(ARTIFACTS_DIR, "sigma_full.npy")
    mu_path      = os.path.join(ARTIFACTS_DIR, "mu_annual.npy")

    returns  = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    sigma    = np.load(sigma_path)
    mu_ann   = np.load(mu_path)        # annualised expected returns
    mu_daily = mu_ann / TRADING_DAYS
    tickers  = returns.columns.tolist()
    N        = len(tickers)

    w_equal  = np.ones(N) / N
    r_p      = portfolio_returns(returns, w_equal)
    mu_p_d   = float(mu_daily @ w_equal)

    print(f"[RISK] Computing risk metrics for equal-weight portfolio …")

    # Point-in-time metrics (full sample)
    param_var = parametric_var(w_equal, sigma, mu_p_d)
    param_es  = parametric_es(w_equal, sigma, mu_p_d)
    hist_var  = historical_var(r_p)
    hist_es   = historical_es(r_p)

    print(f"  Parametric VaR 95%:  {param_var*100:.3f}%")
    print(f"  Historical VaR 95%:  {hist_var*100:.3f}%")
    print(f"  Parametric ES  95%:  {param_es*100:.3f}%")
    print(f"  Historical ES  95%:  {hist_es*100:.3f}%")

    # Rolling risk metrics
    print("[RISK] Computing rolling risk metrics …")
    rolling_risk = compute_rolling_risk(returns, w_equal)
    rolling_risk.to_csv(os.path.join(ARTIFACTS_DIR, "rolling_risk.csv"))

    # Sqrt approximation for FHE
    max_var = float((w_equal @ sigma @ w_equal) * 5)   # headroom
    cheb    = fit_chebyshev_sqrt(max_var, degree=5)
    taylor_test = {
        "at_current_var": float(portfolio_vol(w_equal, sigma) ** 2),
        "true_sqrt":      float(portfolio_vol(w_equal, sigma)),
        "taylor_approx":  taylor_sqrt_approx(float(portfolio_vol(w_equal, sigma)**2),
                                              float(portfolio_vol(w_equal, sigma)**2)),
        "cheb_approx":    eval_chebyshev_sqrt(float(portfolio_vol(w_equal, sigma)**2), cheb),
    }
    print(f"  Chebyshev sqrt approx RMSE: {cheb['rmse']:.6f}, max_err: {cheb['max_error']:.6f}")
    _save_json(cheb, os.path.join(ARTIFACTS_DIR, "sqrt_approx_coeffs.json"))

    # VaR coverage test using rolling parametric VaR
    var_series = rolling_risk["param_var_95"]
    coverage   = var_coverage_test(r_p, var_series)
    print(f"  VaR coverage: {coverage['breach_rate']*100:.2f}% breaches ({coverage['assessment']})")

    # Crisis period analysis
    crisis_periods = {
        "GFC_2008":    ("2008-09-01", "2009-03-31"),
        "COVID_2020":  ("2020-02-15", "2020-05-15"),
        "Bear_2022":   ("2022-01-01", "2022-12-31"),
    }
    crisis_stats = {}
    for name, (cs, ce) in crisis_periods.items():
        mask  = (r_p.index >= cs) & (r_p.index <= ce)
        r_win = r_p[mask]
        if len(r_win) > 5:
            crisis_stats[name] = {
                "hist_var_95": round(historical_var(r_win) * 100, 3),
                "hist_es_95":  round(historical_es(r_win)  * 100, 3),
                "ann_vol_pct": round(r_win.std() * np.sqrt(TRADING_DAYS) * 100, 2),
                "total_return_pct": round(r_win.sum() * 100, 2),
            }

    report = {
        "equal_weight": {
            "parametric_var_95_pct": round(param_var * 100, 4),
            "parametric_es_95_pct":  round(param_es  * 100, 4),
            "historical_var_95_pct": round(hist_var   * 100, 4),
            "historical_es_95_pct":  round(hist_es    * 100, 4),
        },
        "var_coverage_test": coverage,
        "sqrt_approximation": {
            "chebyshev_rmse":    cheb["rmse"],
            "chebyshev_max_err": cheb["max_error"],
            "taylor_test":       taylor_test,
        },
        "crisis_analysis": crisis_stats,
    }
    _save_json(report, os.path.join(ARTIFACTS_DIR, "risk_metrics_report.json"))

    print("\n[RISK] Risk metrics complete. Saved rolling_risk.csv, risk_metrics_report.json")
    return rolling_risk, report


if __name__ == "__main__":
    run_risk_metrics()