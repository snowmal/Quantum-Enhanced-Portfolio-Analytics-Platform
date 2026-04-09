"""
evaluate_classical.py
=====================
Rolling-window backtest for the classical Markowitz model.
Train 3 years → Test 1 year → Step quarterly.

Metrics computed for each window AND aggregated:
  - Out-of-sample Sharpe ratio
  - Maximum drawdown
  - Realised VaR coverage (Kupiec test)
  - Realised ES
  - Portfolio turnover
  - Weight stability (std across windows)

Crisis-period isolation: GFC 2008–09, COVID 2020, Bear 2022.

Inputs  : artifacts/returns.csv
Outputs : artifacts/backtest_results.csv
          artifacts/classical_metrics.json    ← primary output for dashboard
          artifacts/crisis_metrics.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ARTIFACTS_DIR  = os.path.join(os.path.dirname(__file__), "artifacts")
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.02   # annual
ALPHA          = 0.05   # 95% VaR

TRAIN_YEARS  = 3
TEST_MONTHS  = 12
STEP_MONTHS  = 3


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── Re-used helpers (mirror classical_optimizer to avoid circular import) ──────

def _optimize(sigma, mu, lam=0.5, long_only=True):
    N  = len(mu)
    w0 = np.ones(N) / N
    res = minimize(
        lambda w: float(w @ sigma @ w) - lam * float(w @ mu),
        w0,
        jac=lambda w: 2 * sigma @ w - lam * mu,
        method="SLSQP",
        bounds=[(0, 1)] * N if long_only else None,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return res.x if res.success else w0


def _sharpe(r_series, rf=RISK_FREE_RATE / TRADING_DAYS):
    excess = r_series - rf
    return float(excess.mean() / (excess.std() + 1e-12)) * np.sqrt(TRADING_DAYS)


def _max_drawdown(r_series):
    cum   = (1 + r_series).cumprod()
    roll_max = cum.cummax()
    dd    = (cum - roll_max) / roll_max
    return float(dd.min())


def _hist_var(r_series, alpha=ALPHA):
    return float(-np.quantile(r_series.dropna(), alpha))


def _hist_es(r_series, alpha=ALPHA):
    var   = _hist_var(r_series, alpha)
    tail  = r_series[r_series <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def _turnover(w_prev, w_curr):
    return float(np.sum(np.abs(w_curr - w_prev)))


# ── Rolling Backtest ───────────────────────────────────────────────────────────

def rolling_backtest(returns: pd.DataFrame,
                     train_years: int  = TRAIN_YEARS,
                     test_months: int  = TEST_MONTHS,
                     step_months: int  = STEP_MONTHS,
                     lam: float        = 0.5) -> pd.DataFrame:
    """
    Walk-forward backtest:
      Train on `train_years` of data, optimise Markowitz,
      apply to next `test_months` out-of-sample.
      Step forward by `step_months` and repeat.
    """
    print(f"[EVAL] Rolling backtest: train={train_years}yr  test={test_months}mo  step={step_months}mo")
    tickers = returns.columns.tolist()
    N       = len(tickers)

    train_days = train_years * TRADING_DAYS
    test_days  = int(test_months * TRADING_DAYS / 12)
    step_days  = int(step_months * TRADING_DAYS / 12)

    records   = []
    w_history = []
    w_prev    = np.ones(N) / N

    start_i = 0
    window_id = 0

    while start_i + train_days + test_days <= len(returns):
        train_ret = returns.iloc[start_i: start_i + train_days]
        test_ret  = returns.iloc[start_i + train_days: start_i + train_days + test_days]

        # Fit on train
        sigma_tr = train_ret.cov().values
        mu_tr    = train_ret.mean().values
        w_opt    = _optimize(sigma_tr, mu_tr, lam)

        # Evaluate on test
        r_p_test = test_ret @ w_opt

        sr    = _sharpe(r_p_test)
        mdd   = _max_drawdown(r_p_test)
        hvar  = _hist_var(r_p_test)
        hes   = _hist_es(r_p_test)
        turn  = _turnover(w_prev, w_opt)
        ann_ret = float(r_p_test.mean() * TRADING_DAYS)
        ann_vol = float(r_p_test.std() * np.sqrt(TRADING_DAYS))

        records.append({
            "window_id":    window_id,
            "train_start":  train_ret.index[0],
            "train_end":    train_ret.index[-1],
            "test_start":   test_ret.index[0],
            "test_end":     test_ret.index[-1],
            "sharpe":       sr,
            "ann_return":   ann_ret,
            "ann_vol":      ann_vol,
            "max_drawdown": mdd,
            "hist_var_95":  hvar,
            "hist_es_95":   hes,
            "turnover":     turn,
            **{f"w_{t}": float(w_opt[i]) for i, t in enumerate(tickers)},
        })

        w_history.append({"date": test_ret.index[-1],
                          **{t: float(w_opt[i]) for i, t in enumerate(tickers)}})
        w_prev  = w_opt.copy()
        start_i += step_days
        window_id += 1

    print(f"  [OK] {window_id} backtest windows completed.")
    return pd.DataFrame(records)


# ── Crisis-Period Analysis ─────────────────────────────────────────────────────

CRISIS_PERIODS = {
    "GFC_2008-09":   ("2008-09-01", "2009-03-31"),
    "COVID_2020":    ("2020-02-15", "2020-05-15"),
    "Bear_2022":     ("2022-01-01", "2022-12-31"),
}


def crisis_analysis(returns: pd.DataFrame, w: np.ndarray) -> dict:
    """Evaluate a fixed weight vector over each crisis window."""
    r_p = returns @ w
    out = {}
    for name, (cs, ce) in CRISIS_PERIODS.items():
        mask  = (r_p.index >= cs) & (r_p.index <= ce)
        r_win = r_p[mask]
        if len(r_win) < 5:
            continue
        out[name] = {
            "n_days":       len(r_win),
            "sharpe":       round(_sharpe(r_win), 3),
            "total_return": round(r_win.sum() * 100, 2),
            "ann_vol_pct":  round(r_win.std() * np.sqrt(TRADING_DAYS) * 100, 2),
            "hist_var_95":  round(_hist_var(r_win) * 100, 3),
            "hist_es_95":   round(_hist_es(r_win)  * 100, 3),
            "max_drawdown": round(_max_drawdown(r_win) * 100, 2),
        }
    return out


# ── VaR Coverage (rolling) ────────────────────────────────────────────────────

def rolling_var_coverage(returns: pd.DataFrame, backtest_df: pd.DataFrame,
                          tickers: list) -> dict:
    """
    For each backtest window, check what fraction of test-period days
    the realised loss exceeded the train-period VaR forecast.
    """
    breaches = []
    for _, row in backtest_df.iterrows():
        w      = np.array([row[f"w_{t}"] for t in tickers])
        test_s = pd.Timestamp(row["test_start"])
        test_e = pd.Timestamp(row["test_end"])
        train_e = pd.Timestamp(row["train_end"])

        # Train-period VaR
        r_train = (returns.loc[:train_e] @ w).iloc[-TRADING_DAYS:]
        var_fc  = _hist_var(r_train)

        # Test-period breaches
        r_test   = returns.loc[test_s:test_e] @ w
        n_breach = ((-r_test) > var_fc).sum()
        breaches.append(n_breach / len(r_test) if len(r_test) > 0 else np.nan)

    breach_rate = float(np.nanmean(breaches))
    return {
        "mean_breach_rate":  round(breach_rate, 4),
        "target_rate":       ALPHA,
        "kupiec_ratio":      round(breach_rate / ALPHA, 3),
        "assessment":        "well-calibrated" if abs(breach_rate - ALPHA) < 0.02
                             else ("conservative" if breach_rate < ALPHA else "under-estimating"),
    }


# ── Aggregate Metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(backtest_df: pd.DataFrame) -> dict:
    cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown",
            "hist_var_95", "hist_es_95", "turnover"]
    agg = {}
    for c in cols:
        if c in backtest_df.columns:
            agg[c] = {
                "mean":   round(float(backtest_df[c].mean()), 4),
                "std":    round(float(backtest_df[c].std()),  4),
                "min":    round(float(backtest_df[c].min()),  4),
                "max":    round(float(backtest_df[c].max()),  4),
                "median": round(float(backtest_df[c].median()), 4),
            }
    return agg


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)

    print("\n" + "="*60)
    print("  CLASSICAL EVALUATION & BACKTEST")
    print("="*60 + "\n")

    returns = pd.read_csv(os.path.join(ARTIFACTS_DIR, "returns.csv"),
                          index_col=0, parse_dates=True)
    tickers = returns.columns.tolist()
    N       = len(tickers)

    # Full-sample tangency weights (for crisis analysis)
    sigma_full = returns.cov().values
    mu_full    = returns.mean().values
    try:
        from classical_optimizer import find_tangency_portfolio, trace_efficient_frontier
        frontier = trace_efficient_frontier(sigma_full, mu_full / TRADING_DAYS, tickers,
                                            lambda_grid=np.linspace(0, 1, 21))
        w_tangency = find_tangency_portfolio(frontier, sigma_full, mu_full / TRADING_DAYS, tickers)
    except Exception:
        w_tangency = np.ones(N) / N

    # Rolling backtest
    backtest_df = rolling_backtest(returns)
    backtest_df.to_csv(os.path.join(ARTIFACTS_DIR, "backtest_results.csv"), index=False)

    # Aggregate stats
    agg = aggregate_metrics(backtest_df)

    # VaR coverage
    coverage = rolling_var_coverage(returns, backtest_df, tickers)

    # Crisis analysis
    crisis = crisis_analysis(returns, w_tangency)

    # Weight stability across rolling windows
    w_cols   = [c for c in backtest_df.columns if c.startswith("w_")]
    w_stab   = {c.replace("w_", ""): round(float(backtest_df[c].std()), 4) for c in w_cols}
    avg_turn = round(float(backtest_df["turnover"].mean()), 4)

    # ── Time series of portfolio cumulative returns ──
    r_p_series = []
    for _, row in backtest_df.iterrows():
        w_opt   = np.array([row[f"w_{t}"] for t in tickers])
        test_s  = pd.Timestamp(row["test_start"])
        test_e  = pd.Timestamp(row["test_end"])
        r_test  = returns.loc[test_s:test_e] @ w_opt
        r_p_series.append(r_test)

    if r_p_series:
        r_p_all = pd.concat(r_p_series).sort_index()
        r_p_all = r_p_all[~r_p_all.index.duplicated(keep="last")]
        r_p_all.name = "portfolio_return"
        r_p_all.to_csv(os.path.join(ARTIFACTS_DIR, "backtest_portfolio_returns.csv"))
        overall_sharpe  = _sharpe(r_p_all)
        overall_mdd     = _max_drawdown(r_p_all)
        overall_hvar    = _hist_var(r_p_all)
        overall_hes     = _hist_es(r_p_all)
    else:
        overall_sharpe = overall_mdd = overall_hvar = overall_hes = None

    print(f"\n[EVAL] BACKTEST SUMMARY:")
    print(f"  Windows:           {len(backtest_df)}")
    print(f"  Overall Sharpe:    {overall_sharpe:.3f}")
    print(f"  Overall MaxDD:     {overall_mdd*100:.1f}%")
    print(f"  Overall VaR 95%:   {overall_hvar*100:.3f}%")
    print(f"  Avg Turnover:      {avg_turn:.3f}")
    print(f"  VaR Coverage:      {coverage['assessment']} ({coverage['mean_breach_rate']*100:.2f}% breaches)")

    print("\n[EVAL] CRISIS PERFORMANCE:")
    for name, stats in crisis.items():
        print(f"  {name}: Sharpe={stats['sharpe']:.2f}  "
              f"Return={stats['total_return']:.1f}%  MaxDD={stats['max_drawdown']:.1f}%")

    # ── Build classical_metrics.json (primary dashboard input) ──
    classical_metrics = {
        "generated_at":     pd.Timestamp.now().isoformat(),
        "model":            "Classical Markowitz (SLSQP, long-only)",
        "data_range":       {"start": str(returns.index[0].date()),
                             "end":   str(returns.index[-1].date())},
        "n_assets":         N,
        "tickers":          tickers,
        "backtest_config":  {"train_years": TRAIN_YEARS, "test_months": TEST_MONTHS,
                             "step_months": STEP_MONTHS, "lambda": 0.5},
        "overall": {
            "sharpe":        round(overall_sharpe,    3) if overall_sharpe    else None,
            "max_drawdown":  round(overall_mdd,       3) if overall_mdd       else None,
            "hist_var_95":   round(overall_hvar,      4) if overall_hvar      else None,
            "hist_es_95":    round(overall_hes,       4) if overall_hes       else None,
            "avg_turnover":  avg_turn,
        },
        "rolling_windows":      agg,
        "var_coverage":         coverage,
        "crisis_performance":   crisis,
        "weight_stability_std": w_stab,
        "n_backtest_windows":   len(backtest_df),
    }

    _save_json(classical_metrics,
               os.path.join(ARTIFACTS_DIR, "classical_metrics.json"))
    _save_json(crisis,
               os.path.join(ARTIFACTS_DIR, "crisis_metrics.json"))

    print("\n[EVAL] Evaluation complete.")
    print("  classical_metrics.json  — primary dashboard input")
    print("  backtest_results.csv    — per-window metrics")
    print("  backtest_portfolio_returns.csv  — daily returns series")
    return classical_metrics


if __name__ == "__main__":
    run_evaluation()