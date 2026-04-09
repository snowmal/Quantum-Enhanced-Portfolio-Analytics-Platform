"""
classical_model.py
==================
Linear Factor Model: OLS regression per asset, portfolio variance,
Marginal Risk Contribution (MRC), and Component Risk Contribution (CRC).

Inputs  : artifacts/returns.csv, artifacts/factors.csv
Outputs : artifacts/B_matrix.csv      — factor loading matrix (N x K)
          artifacts/residual_cov.npy  — residual covariance Σ_ε
          artifacts/factor_model_report.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
ROLLING_WINDOW = 252   # 1 trading year

FACTOR_COLS = ["Mkt-RF", "SMB", "HML"]   # FF3; extend to FF5 if available


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── OLS Factor Model ──────────────────────────────────────────────────────────

def fit_ols_factor_model(returns: pd.DataFrame, factors: pd.DataFrame):
    """
    Fit OLS regression  r_i = B_i * f + ε_i  for each asset i.

    Returns
    -------
    B      : DataFrame (N x K)  factor loadings
    alphas : Series   (N,)      intercepts (α)
    resid  : DataFrame (T x N) residuals
    R2     : Series   (N,)      in-sample R²
    """
    available_factors = [c for c in FACTOR_COLS if c in factors.columns]
    if not available_factors:
        print("  [WARN] No matching factor columns found. Returning empty factor model.")
        return pd.DataFrame(), pd.Series(), returns.copy(), pd.Series()

    # Align index
    common = returns.index.intersection(factors.index)
    r = returns.loc[common]
    f = factors.loc[common, available_factors]

    # Add intercept column
    X = np.column_stack([np.ones(len(f)), f.values])   # (T x K+1)
    Y = r.values                                         # (T x N)

    # OLS: β = (X'X)^{-1} X'Y
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except LinAlgError as e:
        print(f"  [ERROR] OLS failed: {e}")
        return pd.DataFrame(), pd.Series(), returns.copy(), pd.Series()

    alphas   = pd.Series(beta[0], index=r.columns, name="alpha")
    B        = pd.DataFrame(beta[1:], index=available_factors, columns=r.columns).T  # N x K
    Y_hat    = X @ beta
    resid    = pd.DataFrame(Y - Y_hat, index=common, columns=r.columns)

    ss_res   = (resid ** 2).sum()
    ss_tot   = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    R2       = pd.Series(1 - ss_res.values / (ss_tot + 1e-12), index=r.columns, name="R2")

    return B, alphas, resid, R2


def compute_residual_covariance(residuals: pd.DataFrame) -> np.ndarray:
    """Diagonal residual covariance matrix (assume idiosyncratic risks uncorrelated)."""
    return np.diag(residuals.var().values)


# ── Portfolio Risk ─────────────────────────────────────────────────────────────

def portfolio_variance(w: np.ndarray, sigma: np.ndarray) -> float:
    """σ²_p = w' Σ w"""
    return float(w @ sigma @ w)


def portfolio_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    """σ_p = √(w'Σw)"""
    return float(np.sqrt(max(portfolio_variance(w, sigma), 0)))


def marginal_risk_contribution(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """MRC_i = (Σw)_i / σ_p"""
    sigma_w = sigma @ w
    sig_p   = portfolio_vol(w, sigma)
    return sigma_w / (sig_p + 1e-12)


def component_risk_contribution(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRC_i = w_i * (Σw)_i / σ_p   (sums to σ_p)"""
    return w * marginal_risk_contribution(w, sigma)


def risk_decomposition(w: np.ndarray, sigma: np.ndarray, tickers: list) -> pd.DataFrame:
    """Return a tidy DataFrame of per-asset risk contributions."""
    mrc = marginal_risk_contribution(w, sigma)
    crc = component_risk_contribution(w, sigma)
    sig_p = portfolio_vol(w, sigma)
    pct_crc = crc / (sig_p + 1e-12) * 100   # percentage contribution

    return pd.DataFrame({
        "ticker":      tickers,
        "weight":      w,
        "MRC":         mrc,
        "CRC":         crc,
        "CRC_pct":     pct_crc,
    }).set_index("ticker")


# ── Rolling Factor Model ───────────────────────────────────────────────────────

def fit_rolling_factor_model(returns: pd.DataFrame, factors: pd.DataFrame,
                              window: int = ROLLING_WINDOW):
    """
    Fit OLS factor model on a rolling basis (window = 252 days).
    Returns the most-recent B matrix and rolling R² history.
    """
    print(f"[MODEL] Fitting rolling OLS factor model (window={window}) …")
    available_factors = [c for c in FACTOR_COLS if c in factors.columns]
    if not available_factors:
        print("  [WARN] Factors not available — fitting full-sample model only.")
        B, alphas, resid, R2 = fit_ols_factor_model(returns, factors)
        return B, alphas, resid, R2, pd.DataFrame()

    common = returns.index.intersection(factors.index)
    r = returns.loc[common]
    f = factors.loc[common, available_factors]

    rolling_R2 = []
    for end_i in range(window, len(r) + 1):
        start_i = end_i - window
        r_win   = r.iloc[start_i:end_i]
        f_win   = f.iloc[start_i:end_i]
        _, _, _, R2_win = fit_ols_factor_model(r_win, f_win)
        rolling_R2.append({"date": r.index[end_i - 1], "mean_R2": R2_win.mean()})

    rolling_R2_df = pd.DataFrame(rolling_R2).set_index("date")

    # Final window fit
    B, alphas, resid, R2 = fit_ols_factor_model(r.iloc[-window:], f.iloc[-window:])

    print(f"  [OK] Final B: {B.shape}, mean R²: {R2.mean():.3f}")
    return B, alphas, resid, R2, rolling_R2_df


# ── Main ───────────────────────────────────────────────────────────────────────

def run_factor_model():
    print("\n" + "="*60)
    print("  CLASSICAL FACTOR MODEL")
    print("="*60 + "\n")

    # Load data
    returns_path = os.path.join(ARTIFACTS_DIR, "returns.csv")
    factors_path = os.path.join(ARTIFACTS_DIR, "factors.csv")

    if not os.path.exists(returns_path):
        raise FileNotFoundError("returns.csv not found — run data_pipeline.py first.")

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    factors = pd.read_csv(factors_path, index_col=0, parse_dates=True) \
              if os.path.exists(factors_path) else pd.DataFrame()

    print(f"[MODEL] Returns: {returns.shape}, Factors: {factors.shape}")

    # Full-sample covariance (used throughout classical pipeline)
    sigma_full = returns.cov().values
    tickers    = returns.columns.tolist()
    N          = len(tickers)

    # Fit rolling factor model
    B, alphas, resid, R2, rolling_R2_df = fit_rolling_factor_model(returns, factors)

    # Residual covariance
    sigma_eps = compute_residual_covariance(resid) if not resid.empty else np.zeros((N, N))

    # Equal-weight portfolio for demonstration
    w_equal = np.ones(N) / N
    rd      = risk_decomposition(w_equal, sigma_full, tickers)

    # ── Annualised summary ──
    mu_annual   = (returns.mean() * 252).to_dict()
    vol_annual  = (returns.std() * np.sqrt(252)).to_dict()
    corr_matrix = returns.corr()

    # ── Save artefacts ──
    if not B.empty:
        B.to_csv(os.path.join(ARTIFACTS_DIR, "B_matrix.csv"))
        print(f"  Saved B_matrix.csv  {B.shape}")

    np.save(os.path.join(ARTIFACTS_DIR, "residual_cov.npy"), sigma_eps)
    np.save(os.path.join(ARTIFACTS_DIR, "sigma_full.npy"),   sigma_full)
    np.save(os.path.join(ARTIFACTS_DIR, "mu_annual.npy"),    np.array(list(mu_annual.values())))

    rd.to_csv(os.path.join(ARTIFACTS_DIR, "risk_decomposition.csv"))
    corr_matrix.to_csv(os.path.join(ARTIFACTS_DIR, "correlation_matrix.csv"))

    if not rolling_R2_df.empty:
        rolling_R2_df.to_csv(os.path.join(ARTIFACTS_DIR, "rolling_R2.csv"))

    report = {
        "tickers":           tickers,
        "factor_columns":    B.index.tolist() if not B.empty else [],
        "mean_R2":           float(R2.mean()) if not R2.empty else None,
        "per_asset_R2":      R2.to_dict()     if not R2.empty else {},
        "annualised_return": mu_annual,
        "annualised_vol":    vol_annual,
        "equal_weight_vol_annual": round(portfolio_vol(w_equal, sigma_full) * np.sqrt(252), 4),
    }
    _save_json(report, os.path.join(ARTIFACTS_DIR, "factor_model_report.json"))

    print("\n[MODEL] Factor model complete.")
    print(f"  Mean R²:              {report['mean_R2']:.3f}" if report['mean_R2'] else "  Mean R²: N/A (no factors)")
    print(f"  Equal-weight ann.vol: {report['equal_weight_vol_annual']*100:.1f}%")
    print()

    return B, alphas, resid, R2, sigma_full, tickers


if __name__ == "__main__":
    run_factor_model()