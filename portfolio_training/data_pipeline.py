"""
data_pipeline.py
================
Fetches and preprocesses all data for the Quantum-Enhanced Portfolio Risk System.

Sources:
  - yfinance      : 10–15 S&P 500 adjusted close prices, 2005–2023
  - fredapi       : VIX (VIXCLS), Fed Funds Rate (DFF), yield curve spread (DGS10 - DGS2)
  - pandas_datareader (or manual CSV): Fama-French 3-factor data

Outputs (saved to artifacts/):
  - returns.csv          : daily log returns (T x N)
  - factors.csv          : Fama-French factors aligned to trading days
  - macro.csv            : VIX, FFR, yield spread aligned to trading days
  - covariance.npz       : rolling 252-day covariance slices
  - scaler_params.json   : mean/std used for standardisation (needed for FHE later)
  - clean_data.npz       : standardised feature matrix Z ready for modelling
  - pipeline_report.json : data quality report for the dashboard
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

START_DATE = "2005-01-01"
END_DATE   = "2023-01-01"

# 15 S&P 500 tickers across 5 sectors — all continuously listed 2005-2023
TICKERS = {
    "Tech":       ["AAPL", "MSFT", "INTC"],
    "Finance":    ["JPM",  "BAC",  "GS"],
    "Healthcare": ["JNJ",  "PFE",  "ABT"],
    "Energy":     ["XOM",  "CVX"],
    "Consumer":   ["PG",   "KO",   "WMT", "MCD"],
}
ALL_TICKERS = [t for sector in TICKERS.values() for t in sector]

ROLLING_WINDOW = 252   # 1 trading year for covariance
CLIP_STD       = 3.0   # clip standardised features at ±3σ

# Fama-French 3-factor URL (Kenneth French data library — no auth needed)
FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


# ── Helper functions ───────────────────────────────────────────────────────────

def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _pct(series):
    """Return percentage of non-null values."""
    return round(100 * series.notna().mean(), 2)


# ── Step 1 : Asset Returns ─────────────────────────────────────────────────────

def fetch_asset_returns(tickers=ALL_TICKERS, start=START_DATE, end=END_DATE):
    """
    Download adjusted close prices from Yahoo Finance and compute daily log returns.
    Returns a DataFrame (T x N) of log returns.
    """
    print(f"[DATA] Fetching adjusted close prices for {len(tickers)} tickers …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives Adj Close as 'Close'; handles splits/dividends
        progress=False,
    )["Close"]

    if isinstance(raw, pd.Series):   # single ticker edge-case
        raw = raw.to_frame()

    # Drop any tickers with >5% missing data
    missing_frac = raw.isna().mean()
    bad = missing_frac[missing_frac > 0.05].index.tolist()
    if bad:
        print(f"  [WARN] Dropping tickers with >5% missing data: {bad}")
        raw = raw.drop(columns=bad)

    # Forward-fill remaining gaps (holidays, short halts), then drop leading NaNs
    prices = raw.ffill().dropna()

    # Daily log returns: ln(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Sanity check: flag suspiciously large single-day moves
    extreme = (log_returns.abs() > 0.5).any(axis=1)
    if extreme.any():
        print(f"  [WARN] {extreme.sum()} days with |return| > 50% detected — check data quality.")

    print(f"  [OK] Returns matrix: {log_returns.shape}  ({log_returns.index[0].date()} → {log_returns.index[-1].date()})")
    return log_returns


# ── Step 2 : Fama-French Factors ───────────────────────────────────────────────

def fetch_fama_french(start=START_DATE, end=END_DATE):
    """
    Download the Fama-French 3-factor daily CSV directly from Ken French's library.
    Returns a DataFrame with columns [Mkt-RF, SMB, HML, RF] as decimals.
    """
    print("[DATA] Fetching Fama-French 3-factor data …")
    try:
        ff = pd.read_csv(FF3_URL, skiprows=3, index_col=0)
        # The CSV has a footer section starting with a blank/non-numeric index
        ff = ff[pd.to_numeric(ff.index, errors="coerce").notna()]
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
        ff = ff.astype(float) / 100.0   # convert from percent to decimal
        ff.index.name = "Date"
        ff.columns = [c.strip() for c in ff.columns]
        mask = (ff.index >= start) & (ff.index < end)
        ff = ff.loc[mask]
        print(f"  [OK] FF3 factors: {ff.shape}  ({ff.index[0].date()} → {ff.index[-1].date()})")
        return ff
    except Exception as e:
        print(f"  [WARN] Could not fetch FF3 from web ({e}). Returning empty factors.")
        return pd.DataFrame()


# ── Step 3 : FRED Macro Variables ─────────────────────────────────────────────

def fetch_macro_variables(start=START_DATE, end=END_DATE):
    """
    Fetch VIX, Federal Funds Rate, and yield curve spread (10Y - 2Y) from FRED.
    Requires a FRED API key set as environment variable FRED_API_KEY,
    or passed explicitly.  Falls back to yfinance proxies if unavailable.
    """
    print("[DATA] Fetching macro variables …")
    api_key = os.environ.get("FRED_API_KEY", "")

    macro = pd.DataFrame()

    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            vix    = fred.get_series("VIXCLS", start, end).rename("VIX")
            ffr    = fred.get_series("DFF",    start, end).rename("FFR")
            t10    = fred.get_series("DGS10",  start, end)
            t2     = fred.get_series("DGS2",   start, end)
            spread = (t10 - t2).rename("yield_spread")
            macro  = pd.concat([vix, ffr, spread], axis=1)
            macro.index.name = "Date"
            print(f"  [OK] FRED macro: {macro.shape}")
        except Exception as e:
            print(f"  [WARN] FRED fetch failed ({e}). Falling back to yfinance proxies.")
        
    print(api_key)

    if macro.empty:
        # yfinance fallback: ^VIX is freely available; yield data via Treasury ETFs
        print("  [INFO] Using yfinance fallback for macro data (^VIX).")
        vix_raw = yf.download("^VIX", start=start, end=end,
                              auto_adjust=True, progress=False)["Close"]
        vix_raw = vix_raw.rename("VIX")

        # Yield spread proxy: IEF (7-10Y Treasury ETF) - SHY (1-3Y) log-price diff as proxy
        # For a true yield spread use FRED; this is only a directional proxy
        try:
            ief = yf.download("IEF", start=start, end=end, auto_adjust=True, progress=False)["Close"]
            shy = yf.download("SHY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
            spread_proxy = (np.log(ief) - np.log(shy)).rename("yield_spread")
        except Exception:
            spread_proxy = pd.Series(dtype=float, name="yield_spread")

        macro = pd.concat([vix_raw, spread_proxy], axis=1)
        macro["FFR"] = np.nan   # placeholder — not available without FRED key
        macro.index.name = "Date"
        print(f"  [OK] yfinance macro fallback: {macro.shape}")

    return macro


# ── Step 4 : Alignment ─────────────────────────────────────────────────────────

def align_data(returns, factors, macro):
    """
    Align all three DataFrames to the equity trading calendar (returns.index).
    Forward-fill macro and factor values on non-release days.
    """
    print("[DATA] Aligning time series to equity trading calendar …")
    trading_days = returns.index

    if not factors.empty:
        factors = factors.reindex(trading_days).ffill().fillna(0)
    else:
        factors = pd.DataFrame(index=trading_days)

    macro = macro.reindex(trading_days).ffill().fillna(method="bfill")

    # Drop any remaining all-NaN columns
    macro = macro.dropna(axis=1, how="all")

    print(f"  [OK] Aligned: returns{returns.shape}, factors{factors.shape}, macro{macro.shape}")
    return returns, factors, macro


# ── Step 5 : Rolling Covariance ────────────────────────────────────────────────

def compute_rolling_covariance(returns, window=ROLLING_WINDOW):
    """
    Compute rolling 252-day covariance matrices.
    Returns a dict {date_str: cov_matrix_array} and the final (most recent) Σ.
    """
    print(f"[DATA] Computing rolling {window}-day covariance matrices …")
    dates   = returns.index[window:]
    cov_slices = {}
    for i, date in enumerate(dates):
        window_ret = returns.iloc[i: i + window]
        cov_slices[str(date.date())] = window_ret.cov().values

    latest_cov = returns.iloc[-window:].cov()
    print(f"  [OK] Computed {len(cov_slices)} covariance slices. Latest Σ shape: {latest_cov.shape}")
    return cov_slices, latest_cov


# ── Step 6 : Standardisation ───────────────────────────────────────────────────

def standardise_features(returns, factors, macro, clip=CLIP_STD):
    """
    Build the combined feature matrix Z, standardise, and clip at ±clip σ.
    Saves scaler parameters for later FHE encoding.
    Returns Z (numpy array) and column names.
    """
    print("[DATA] Building and standardising feature matrix …")

    parts = [returns]
    if not factors.empty:
        # Only keep factor columns (drop RF — it's the risk-free rate, used separately)
        factor_cols = [c for c in factors.columns if c != "RF"]
        parts.append(factors[factor_cols])
    if not macro.empty:
        parts.append(macro)

    combined = pd.concat(parts, axis=1).dropna()
    col_names = combined.columns.tolist()

    scaler = StandardScaler()
    Z = scaler.fit_transform(combined.values)
    Z = np.clip(Z, -clip, clip)

    # Save scaler for FHE encoding later
    scaler_params = {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "columns": col_names,
        "clip":  clip,
    }
    _save_json(scaler_params, os.path.join(ARTIFACTS_DIR, "scaler_params.json"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    print(f"  [OK] Feature matrix Z: {Z.shape}, clipped at ±{clip}σ")
    return Z, col_names, combined.index


# ── Step 7 : Data Quality Report ──────────────────────────────────────────────

def build_quality_report(returns, factors, macro, tickers):
    report = {
        "generated_at":   datetime.now().isoformat(),
        "date_range":     {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())},
        "n_trading_days": len(returns),
        "tickers": tickers,
        "n_assets":       len(returns.columns),
        "returns_completeness_pct": {col: _pct(returns[col]) for col in returns.columns},
        "factor_columns": factors.columns.tolist() if not factors.empty else [],
        "macro_columns":  macro.columns.tolist()   if not macro.empty  else [],
        "macro_completeness_pct": {col: _pct(macro[col]) for col in macro.columns} if not macro.empty else {},
        "annualised_return_pct":  {col: round(returns[col].mean() * 252 * 100, 2) for col in returns.columns},
        "annualised_vol_pct":     {col: round(returns[col].std() * np.sqrt(252) * 100, 2) for col in returns.columns},
    }
    _save_json(report, os.path.join(ARTIFACTS_DIR, "pipeline_report.json"))
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "="*60)
    print("  PORTFOLIO DATA PIPELINE")
    print("="*60 + "\n")

    returns = fetch_asset_returns()
    factors = fetch_fama_french()
    macro   = fetch_macro_variables()

    returns, factors, macro = align_data(returns, factors, macro)

    cov_slices, latest_cov = compute_rolling_covariance(returns)

    Z, col_names, z_index = standardise_features(returns, factors, macro)

    report = build_quality_report(returns, factors, macro, returns.columns.tolist())

    # ── Save artefacts ──
    returns.to_csv(os.path.join(ARTIFACTS_DIR, "returns.csv"))
    if not factors.empty:
        factors.to_csv(os.path.join(ARTIFACTS_DIR, "factors.csv"))
    if not macro.empty:
        macro.to_csv(os.path.join(ARTIFACTS_DIR, "macro.csv"))

    np.savez(
        os.path.join(ARTIFACTS_DIR, "covariance.npz"),
        latest_cov=latest_cov.values,
        tickers=np.array(returns.columns.tolist()),
        **{k: v for k, v in list(cov_slices.items())[-50:]},  # save last 50 slices (memory)
    )

    np.savez(
        os.path.join(ARTIFACTS_DIR, "clean_data.npz"),
        Z=Z,
        columns=np.array(col_names),
        dates=np.array(z_index.astype(str)),
    )

    print("\n[DATA] Pipeline complete. Artifacts saved to:", ARTIFACTS_DIR)
    print(f"  returns.csv            {returns.shape}")
    print(f"  factors.csv            {factors.shape}")
    print(f"  macro.csv              {macro.shape}")
    print(f"  covariance.npz         {latest_cov.shape} latest Σ")
    print(f"  clean_data.npz         Z={Z.shape}")
    print(f"  scaler_params.json     {len(col_names)} features")
    print(f"  pipeline_report.json   quality report")
    print()
    return returns, factors, macro, latest_cov, Z, col_names


if __name__ == "__main__":
    run_pipeline()