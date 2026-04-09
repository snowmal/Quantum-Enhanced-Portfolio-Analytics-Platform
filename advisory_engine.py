"""
advisory_engine.py
==================
Rule-based advisory engine. Reads live_state.json and projections.json,
applies a hierarchy of financial rules, and writes structured advice to
artifacts/advice.json.

The advice covers four areas:
  1. ALLOCATION ADVICE    — are current weights appropriate?
  2. RISK ALERTS         — VaR, drawdown, volatility warnings
  3. REBALANCING SIGNALS — drift from target, momentum-based tilts
  4. SCENARIO GUIDANCE   — what to do given projection scenarios

Rules are tiered by severity:
  ACTION   — immediate change recommended (red)
  WATCH    — monitor closely (amber)
  INFO     — positive signal or context (green)

Design
------
Each rule is a function that takes the state/projections dicts and returns
either None (rule not triggered) or an Advice namedtuple. Rules are
evaluated in priority order; the top-N most severe are surfaced.

Adding a new rule: write a function with signature
    def rule_my_rule(state, proj) -> Optional[Advice]
and add it to RULE_REGISTRY. No other changes needed.

Usage
-----
    # Run once:
    python3 advisory_engine.py

    # Called automatically by live_optimizer (set --with-advice flag):
    python3 live_optimizer.py --with-advice

Dependencies
------------
    pip install numpy pandas
"""

import os
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, NamedTuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = HERE / "portfolio_training" / "artifacts"
LIVE_STATE = ARTIFACTS / "live_state.json"
PROJ_FILE  = ARTIFACTS / "projections.json"
ADVICE_OUT = ARTIFACTS / "advice.json"

# ── Thresholds (tune these to your client's risk profile) ─────────────────────
THR = {
    # Risk alerts
    "VAR_WARN_PCT":          2.0,    # daily VaR % triggering a watch
    "VAR_ACTION_PCT":        3.5,    # daily VaR % triggering action
    "VOL_WARN_ANN_PCT":     18.0,    # annualised vol % watch
    "VOL_ACTION_ANN_PCT":   25.0,    # annualised vol % action
    "MAX_DD_WARN_PCT":      10.0,    # max drawdown watch
    "MAX_DD_ACTION_PCT":    20.0,    # max drawdown action

    # Concentration
    "CONC_WARN_PCT":        35.0,    # single asset weight % — watch
    "CONC_ACTION_PCT":      50.0,    # single asset weight % — action
    "SECTOR_CONC_PCT":      60.0,    # sector concentration watch

    # Momentum
    "MOMENTUM_STRONG_PCT":   5.0,    # 20d return — bullish signal
    "MOMENTUM_WEAK_PCT":    -5.0,    # 20d return — bearish signal

    # Return/Sharpe
    "SHARPE_STRONG":         0.7,    # annualised Sharpe — positive signal
    "SHARPE_WEAK":           0.2,    # annualised Sharpe — watch
    "SHARPE_NEGATIVE":       0.0,    # negative Sharpe — action

    # Projection
    "PROB_LOSS_10PCT_WARN": 0.30,    # >30% chance of 10% loss in 3M — watch
    "PROB_LOSS_10PCT_ACT":  0.50,    # >50% — action

    # Turnover (how much to rebalance)
    "REBAL_THRESHOLD_PCT":   5.0,    # position drift > 5% triggers rebalance signal

    # Correlation
    "HIGH_CORR_WARN":        0.85,   # avg pairwise correlation — diversification warning
}

# Sector map (mirrors data_pipeline.py TICKERS)
SECTOR_MAP = {
    "AAPL": "Tech",     "MSFT": "Tech",     "INTC": "Tech",
    "JPM":  "Finance",  "BAC":  "Finance",  "GS":   "Finance",
    "JNJ":  "Healthcare","PFE": "Healthcare","ABT":  "Healthcare",
    "XOM":  "Energy",   "CVX":  "Energy",
    "PG":   "Consumer", "KO":   "Consumer", "WMT":  "Consumer", "MCD": "Consumer",
}

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# ADVICE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class Advice(NamedTuple):
    category:    str     # ALLOCATION | RISK | REBALANCING | PROJECTION
    severity:    str     # ACTION | WATCH | INFO
    title:       str     # short headline
    body:        str     # 1–3 sentences of explanation
    metric:      str     # the specific number driving the advice
    action:      str     # concrete next step
    priority:    int     # lower = more important (used for sorting)


def _advice(category, severity, title, body, metric, action, priority=50):
    return Advice(category, severity, title, body, metric, action, priority)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RISK ALERT RULES
# ─────────────────────────────────────────────────────────────────────────────

def rule_var_level(state, proj) -> Optional[Advice]:
    risk = state.get("risk", {})
    var  = risk.get("var_95_pct", 0)
    if var >= THR["VAR_ACTION_PCT"]:
        return _advice(
            "RISK", "ACTION",
            "Daily VaR exceeds action threshold",
            f"The portfolio's 95% daily Value-at-Risk is {var:.2f}%, meaning on the worst 5% "
            f"of days you can expect to lose more than {var:.2f}% of portfolio value. "
            f"This is above the {THR['VAR_ACTION_PCT']}% action threshold.",
            f"VaR 95% = {var:.2f}%",
            "Consider reducing allocation to the highest Component Risk Contribution (CRC) "
            "assets or adding defensive positions (e.g. Treasuries, cash) to bring VaR below "
            f"{THR['VAR_ACTION_PCT']}%.",
            priority=5,
        )
    elif var >= THR["VAR_WARN_PCT"]:
        return _advice(
            "RISK", "WATCH",
            "Daily VaR approaching warning level",
            f"Daily VaR of {var:.2f}% is elevated. While not requiring immediate action, "
            f"this warrants monitoring — especially given current macro conditions.",
            f"VaR 95% = {var:.2f}%",
            f"Monitor CRC attribution. If VaR reaches {THR['VAR_ACTION_PCT']}%, "
            "rebalance toward lower-volatility assets.",
            priority=20,
        )
    return None


def rule_volatility_regime(state, proj) -> Optional[Advice]:
    risk   = state.get("risk", {})
    vol    = risk.get("sigma_p_annual_pct", 0)
    if vol >= THR["VOL_ACTION_ANN_PCT"]:
        return _advice(
            "RISK", "ACTION",
            "Portfolio volatility in high-risk regime",
            f"Annualised portfolio volatility of {vol:.1f}% is above the {THR['VOL_ACTION_ANN_PCT']}% "
            "action threshold, indicating the portfolio is operating in a high-risk regime. "
            "Historical data shows elevated drawdown probability in this environment.",
            f"Ann. vol = {vol:.1f}%",
            "Reduce equity concentration. Increase allocation to low-beta assets or sectors "
            "with negative correlation to current risk drivers. Target vol below "
            f"{THR['VOL_WARN_ANN_PCT']}%.",
            priority=6,
        )
    elif vol >= THR["VOL_WARN_ANN_PCT"]:
        return _advice(
            "RISK", "WATCH",
            "Volatility above normal range",
            f"Annualised volatility of {vol:.1f}% exceeds the {THR['VOL_WARN_ANN_PCT']}% "
            "warning level. The market environment is more uncertain than the historical average.",
            f"Ann. vol = {vol:.1f}%",
            "Review momentum signals for trend confirmation. Consider tightening stop-loss "
            "levels or adding variance-reduction overlays.",
            priority=22,
        )
    elif vol < 10.0:
        return _advice(
            "RISK", "INFO",
            "Volatility at favourable low levels",
            f"Annualised portfolio volatility of {vol:.1f}% is low, suggesting a benign "
            "risk environment. Risk-adjusted returns are likely to be strong.",
            f"Ann. vol = {vol:.1f}%",
            "Maintain current positioning. Low-vol regimes often persist — no action needed "
            "unless momentum signals diverge.",
            priority=60,
        )
    return None


def rule_max_drawdown(state, proj) -> Optional[Advice]:
    risk = state.get("risk", {})
    mdd  = risk.get("max_drawdown_pct", 0)
    if mdd >= THR["MAX_DD_ACTION_PCT"]:
        return _advice(
            "RISK", "ACTION",
            "Maximum drawdown breached action threshold",
            f"The portfolio has experienced a peak-to-trough drawdown of {mdd:.1f}% "
            f"over the observed window. This exceeds the {THR['MAX_DD_ACTION_PCT']}% "
            "action threshold and signals sustained loss accumulation.",
            f"Max drawdown = {mdd:.1f}%",
            "Review allocation immediately. Identify assets driving the drawdown via CRC "
            "attribution and consider systematic de-risking. Avoid panic selling — "
            "assess whether the drawdown reflects regime change or temporary dislocation.",
            priority=3,
        )
    elif mdd >= THR["MAX_DD_WARN_PCT"]:
        return _advice(
            "RISK", "WATCH",
            "Drawdown approaching warning threshold",
            f"Current drawdown of {mdd:.1f}% is approaching the {THR['MAX_DD_ACTION_PCT']}% "
            "action level. Monitor closely for further deterioration.",
            f"Max drawdown = {mdd:.1f}%",
            "Check crisis-period performance in the Risk Analysis tab. If the drawdown "
            "is concentrated in one sector, consider targeted rebalancing.",
            priority=18,
        )
    return None


def rule_sharpe_quality(state, proj) -> Optional[Advice]:
    opt    = state.get("optimization", {})
    sharpe = opt.get("ann_sharpe", None)
    if sharpe is None:
        return None
    if sharpe < THR["SHARPE_NEGATIVE"]:
        return _advice(
            "RISK", "ACTION",
            "Negative Sharpe ratio — portfolio destroying risk-adjusted value",
            f"The annualised Sharpe ratio is {sharpe:.3f}, meaning the portfolio is earning "
            "less than the risk-free rate per unit of risk taken. The portfolio is currently "
            "destroying risk-adjusted value.",
            f"Sharpe = {sharpe:.3f}",
            "Re-optimize with a higher risk-aversion λ to shift toward the minimum-variance "
            "portfolio. Review whether current factor exposures are appropriate for the regime.",
            priority=4,
        )
    elif sharpe < THR["SHARPE_WEAK"]:
        return _advice(
            "RISK", "WATCH",
            "Sharpe ratio below minimum acceptable level",
            f"Annualised Sharpe of {sharpe:.3f} is below the {THR['SHARPE_WEAK']} minimum "
            "threshold. The portfolio's return is not adequately compensating for its risk.",
            f"Sharpe = {sharpe:.3f}",
            "Consider tilting toward assets with higher recent momentum scores while "
            "maintaining diversification constraints.",
            priority=25,
        )
    elif sharpe >= THR["SHARPE_STRONG"]:
        return _advice(
            "RISK", "INFO",
            "Strong risk-adjusted return — Sharpe above target",
            f"Annualised Sharpe of {sharpe:.3f} exceeds the {THR['SHARPE_STRONG']} target. "
            "The portfolio is generating strong risk-adjusted returns in the current regime.",
            f"Sharpe = {sharpe:.3f}",
            "Maintain current allocation. Consider locking in gains by slightly reducing "
            "the highest-weight positions if they have drifted above target.",
            priority=70,
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ALLOCATION / CONCENTRATION RULES
# ─────────────────────────────────────────────────────────────────────────────

def rule_concentration(state, proj) -> Optional[Advice]:
    attrib = state.get("attribution", {})
    if not attrib:
        return None

    weights = {t: v["weight_pct"] for t, v in attrib.items()}
    max_t   = max(weights, key=weights.get)
    max_w   = weights[max_t]

    if max_w >= THR["CONC_ACTION_PCT"]:
        return _advice(
            "ALLOCATION", "ACTION",
            f"Dangerous concentration in {max_t}",
            f"{max_t} represents {max_w:.1f}% of the portfolio — a single-asset concentration "
            f"above the {THR['CONC_ACTION_PCT']}% action limit. This creates idiosyncratic "
            "risk that diversification cannot hedge.",
            f"{max_t} weight = {max_w:.1f}%",
            f"Reduce {max_t} to below {THR['CONC_WARN_PCT']}% by redistributing into "
            "correlated but lower-weight peers or adding a new uncorrelated position.",
            priority=7,
        )
    elif max_w >= THR["CONC_WARN_PCT"]:
        return _advice(
            "ALLOCATION", "WATCH",
            f"Elevated single-asset concentration: {max_t}",
            f"{max_t} at {max_w:.1f}% is approaching the concentration warning threshold. "
            "While not immediately concerning, monitor for further drift.",
            f"{max_t} weight = {max_w:.1f}%",
            f"Set an alert if {max_t} exceeds {THR['CONC_ACTION_PCT']}%. Consider capping "
            "at current level during next rebalance.",
            priority=28,
        )
    return None


def rule_sector_concentration(state, proj) -> Optional[Advice]:
    attrib = state.get("attribution", {})
    if not attrib:
        return None

    sector_weights: dict = {}
    for ticker, vals in attrib.items():
        sector = SECTOR_MAP.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + vals["weight_pct"]

    max_sec = max(sector_weights, key=sector_weights.get)
    max_w   = sector_weights[max_sec]

    if max_w >= THR["SECTOR_CONC_PCT"]:
        return _advice(
            "ALLOCATION", "WATCH",
            f"Sector concentration: {max_sec} at {max_w:.1f}%",
            f"The {max_sec} sector represents {max_w:.1f}% of the portfolio. "
            f"Concentration above {THR['SECTOR_CONC_PCT']}% introduces sector-specific "
            "risk that can amplify losses during sector rotations.",
            f"{max_sec} sector = {max_w:.1f}%",
            f"Diversify by adding positions in underrepresented sectors. Target a maximum "
            "sector weight of 40% for robust diversification.",
            priority=30,
        )
    return None


def rule_diversification_quality(state, proj) -> Optional[Advice]:
    """Check if the portfolio is sufficiently diversified via average correlation."""
    corr_path = ARTIFACTS / "correlation_matrix.csv"
    if not corr_path.exists():
        return None
    try:
        corr = pd.read_csv(corr_path, index_col=0)
        n    = len(corr)
        # Average off-diagonal correlation
        mask     = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = float(corr.values[mask].mean())
        if avg_corr >= THR["HIGH_CORR_WARN"]:
            return _advice(
                "ALLOCATION", "WATCH",
                "High average inter-asset correlation — reduced diversification benefit",
                f"Average pairwise correlation of {avg_corr:.2f} indicates assets are moving "
                "together, reducing the diversification benefit of holding multiple positions. "
                "In a stress event, the portfolio may behave as a concentrated bet.",
                f"Avg correlation = {avg_corr:.2f}",
                "Consider adding assets with low or negative correlation to current holdings "
                "(e.g. Treasuries, gold, low-beta defensives) to restore true diversification.",
                priority=35,
            )
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — REBALANCING SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def rule_weight_drift(state, proj) -> Optional[Advice]:
    """Compare live weights to the classical optimal and flag significant drift."""
    w_class_path = ARTIFACTS / "w_classical.csv"
    if not w_class_path.exists():
        return None
    try:
        w_df     = pd.read_csv(w_class_path, index_col=0)
        attrib   = state.get("attribution", {})
        tickers  = list(attrib.keys())
        if "tangency" not in w_df.columns:
            return None
        w_target = w_df.loc[tickers, "tangency"].values if all(t in w_df.index for t in tickers) \
                   else np.ones(len(tickers)) / len(tickers)
        w_live   = np.array([attrib[t]["weight_pct"] / 100 for t in tickers])
        drift    = np.abs(w_live - w_target)
        max_drift_t = tickers[np.argmax(drift)]
        max_drift   = float(drift.max()) * 100
        total_drift = float(drift.sum()) * 100

        if max_drift >= THR["REBAL_THRESHOLD_PCT"]:
            return _advice(
                "REBALANCING", "WATCH",
                f"Weight drift detected — {max_drift_t} drifted {max_drift:.1f}%",
                f"Current weights have drifted from the optimal Markowitz tangency portfolio. "
                f"{max_drift_t} has drifted {max_drift:.1f}% from its target weight. "
                f"Total portfolio drift is {total_drift:.1f}%.",
                f"Max drift = {max_drift:.1f}% ({max_drift_t})",
                "Schedule a rebalance to bring weights back to the optimal allocation. "
                "Consider transaction costs — only rebalance if drift exceeds cost threshold.",
                priority=40,
            )
    except Exception:
        pass
    return None


def rule_momentum_signals(state, proj) -> Optional[Advice]:
    """Surface strong momentum signals for individual assets."""
    momentum = state.get("momentum", {})
    attrib   = state.get("attribution", {})
    if not momentum:
        return None

    bullish  = [(t, m["20d"]) for t, m in momentum.items()
                if m["signal"] == "bullish" and m["20d"] >= THR["MOMENTUM_STRONG_PCT"]]
    bearish  = [(t, m["20d"]) for t, m in momentum.items()
                if m["signal"] == "bearish" and m["20d"] <= THR["MOMENTUM_WEAK_PCT"]]

    parts = []
    actions = []

    if bearish:
        worst  = sorted(bearish, key=lambda x: x[1])[:3]
        for t, ret in worst:
            w_pct = attrib.get(t, {}).get("weight_pct", 0)
            parts.append(f"{t} ({ret:+.1f}% 20d, weight {w_pct:.1f}%)")
        actions.append(
            f"Consider reducing exposure to: {', '.join(t for t, _ in worst)}. "
            "Bearish momentum tends to persist over 1–3 month horizons."
        )

    if bullish:
        best = sorted(bullish, key=lambda x: -x[1])[:3]
        for t, ret in best:
            w_pct = attrib.get(t, {}).get("weight_pct", 0)
            parts.append(f"{t} ({ret:+.1f}% 20d, weight {w_pct:.1f}%)")
        actions.append(
            f"Strong momentum in: {', '.join(t for t, _ in best)}. "
            "Consider tilting allocation toward these within risk limits."
        )

    if not parts:
        return None

    sev = "WATCH" if bearish else "INFO"
    return _advice(
        "REBALANCING", sev,
        f"Momentum signals: {len(bearish)} bearish, {len(bullish)} bullish",
        f"Significant momentum signals detected. {'; '.join(parts[:4])}.",
        f"{len(bearish)} bearish / {len(bullish)} bullish assets",
        " ".join(actions),
        priority=45 if bearish else 65,
    )


def rule_rebalance_opportunity(state, proj) -> Optional[Advice]:
    """Positive rebalancing signal when Sharpe is strong and vol is low."""
    opt    = state.get("optimization", {})
    risk   = state.get("risk", {})
    sharpe = opt.get("ann_sharpe", 0)
    vol    = risk.get("sigma_p_annual_pct", 99)

    if sharpe >= THR["SHARPE_STRONG"] and vol < THR["VOL_WARN_ANN_PCT"]:
        return _advice(
            "REBALANCING", "INFO",
            "Favourable conditions for strategic rebalancing",
            f"With a Sharpe of {sharpe:.3f} and annualised volatility of {vol:.1f}%, "
            "the portfolio is operating efficiently in a low-risk environment. "
            "This is an optimal time to rebalance any drifted positions.",
            f"Sharpe={sharpe:.3f}, vol={vol:.1f}%",
            "Execute any pending rebalancing trades. Transaction costs are minimised "
            "in low-volatility regimes.",
            priority=75,
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PROJECTION-BASED RULES
# ─────────────────────────────────────────────────────────────────────────────

def rule_loss_probability(state, proj) -> Optional[Advice]:
    if not proj:
        return None
    mc   = proj.get("monte_carlo", {})
    cp   = mc.get("checkpoints", {}).get("63d", {})   # 3-month horizon
    prob = cp.get("prob_loss_10pct", 0)
    mean = cp.get("mean", 1.0)

    if prob >= THR["PROB_LOSS_10PCT_ACT"]:
        return _advice(
            "PROJECTION", "ACTION",
            f"High probability of 10%+ loss over 3 months ({prob*100:.0f}%)",
            f"Monte Carlo simulation ({mc.get('n_paths',0):,} paths) shows a "
            f"{prob*100:.0f}% probability of a 10%+ loss over the next 3 months. "
            f"Expected portfolio value: {mean:.3f}× current value.",
            f"P(loss>10%, 3M) = {prob*100:.0f}%",
            "Consider defensive positioning: increase cash or short-duration bonds, "
            "add tail-risk hedges (puts or low-vol ETFs), or reduce overall equity exposure.",
            priority=8,
        )
    elif prob >= THR["PROB_LOSS_10PCT_WARN"]:
        return _advice(
            "PROJECTION", "WATCH",
            f"Elevated tail risk over 3-month horizon ({prob*100:.0f}% P(loss>10%))",
            f"There is a {prob*100:.0f}% probability of a 10%+ loss over the next "
            f"3 months. Expected value is {mean:.3f}× current, but the downside tail is wide.",
            f"P(loss>10%, 3M) = {prob*100:.0f}%",
            "Review your allocation to the highest-CRC assets. Consider whether "
            "current risk exposure aligns with your investment mandate.",
            priority=15,
        )
    elif mean >= 1.05:
        return _advice(
            "PROJECTION", "INFO",
            f"Positive 3-month expected return ({(mean-1)*100:.1f}%)",
            f"Monte Carlo projects a {(mean-1)*100:.1f}% expected gain over 3 months "
            f"with a {prob*100:.0f}% probability of a 10%+ loss. "
            "The risk/reward profile looks favourable.",
            f"E[return, 3M] = {(mean-1)*100:.1f}%",
            "No immediate action needed. Maintain current allocation and monitor "
            "for changes in the macro environment.",
            priority=80,
        )
    return None


def rule_scenario_worst_case(state, proj) -> Optional[Advice]:
    if not proj:
        return None
    scenarios = proj.get("scenarios", {})
    if not scenarios:
        return None

    # Find the worst expected value scenario (excluding base case)
    worst_name, worst_data = None, None
    for name, data in scenarios.items():
        if name == "Base Case":
            continue
        if worst_data is None or data["expected_value"] < worst_data["expected_value"]:
            worst_name, worst_data = name, data

    if worst_name is None:
        return None

    ev      = worst_data["expected_value"]
    p_loss  = worst_data["prob_loss_10pct"]
    loss_pct = (1 - ev) * 100

    if loss_pct >= 20:
        return _advice(
            "PROJECTION", "WATCH",
            f"Severe downside in worst scenario: {worst_name} ({loss_pct:.0f}% loss)",
            f"Under the '{worst_name}' scenario, the portfolio would lose approximately "
            f"{loss_pct:.0f}% of value over 3 months. Probability of 10%+ loss: "
            f"{p_loss*100:.0f}%. Description: {worst_data['description']}",
            f"Worst scenario loss = {loss_pct:.0f}% ({worst_name})",
            "Assess portfolio resilience to this scenario. If it represents a plausible "
            "near-term risk, consider building in tail hedges now while premiums are low.",
            priority=35,
        )
    return None


def rule_soft_landing_opportunity(state, proj) -> Optional[Advice]:
    if not proj:
        return None
    scenarios   = proj.get("scenarios", {})
    soft        = scenarios.get("Soft Landing", {})
    base        = scenarios.get("Base Case", {})
    if not soft or not base:
        return None

    upside_pct = (soft.get("expected_value", 1.0) - base.get("expected_value", 1.0)) * 100
    if upside_pct >= 3.0:
        return _advice(
            "PROJECTION", "INFO",
            f"Soft landing scenario offers {upside_pct:.1f}% upside vs. base case",
            f"If macro conditions improve to a soft-landing scenario, the portfolio "
            f"would generate {upside_pct:.1f}% more than the base case projection. "
            "Current positioning appears well-suited to capture this upside.",
            f"Soft landing upside = {upside_pct:.1f}% vs. base",
            "Consider tilting slightly toward growth and cyclical exposures "
            "to capture soft-landing alpha while maintaining downside protection.",
            priority=85,
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RULE REGISTRY — add new rules here
# ─────────────────────────────────────────────────────────────────────────────

RULE_REGISTRY = [
    # Risk rules (highest priority)
    rule_max_drawdown,
    rule_sharpe_quality,
    rule_var_level,
    rule_volatility_regime,
    rule_loss_probability,
    # Allocation rules
    rule_concentration,
    rule_sector_concentration,
    rule_diversification_quality,
    # Rebalancing rules
    rule_weight_drift,
    rule_momentum_signals,
    rule_rebalance_opportunity,
    # Projection rules
    rule_scenario_worst_case,
    rule_soft_landing_opportunity,
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SUMMARY NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(advices: list, state: dict, proj: dict) -> dict:
    """
    Generate a top-level portfolio health summary with:
      - Overall health score (0–100)
      - One-sentence headline
      - Recommended immediate action count
    """
    n_action = sum(1 for a in advices if a.severity == "ACTION")
    n_watch  = sum(1 for a in advices if a.severity == "WATCH")
    n_info   = sum(1 for a in advices if a.severity == "INFO")

    opt    = state.get("optimization", {})
    risk   = state.get("risk", {})
    sharpe = opt.get("ann_sharpe", 0)
    vol    = risk.get("sigma_p_annual_pct", 20)
    var    = risk.get("var_95_pct", 3)

    # Health score: starts at 100, deducted for each issue
    score = 100
    score -= n_action * 20
    score -= n_watch  * 8
    score -= max(0, (var  - THR["VAR_WARN_PCT"])    * 10)
    score -= max(0, (vol  - THR["VOL_WARN_ANN_PCT"]) * 1.5)
    score += max(0, (sharpe - 0.4) * 15)
    score  = max(0, min(100, score))

    if score >= 75:
        health_label = "HEALTHY"
        health_color = "green"
        headline     = (f"Portfolio is performing well with a Sharpe of {sharpe:.2f} "
                        f"and controlled risk levels. No immediate action required.")
    elif score >= 50:
        health_label = "CAUTION"
        health_color = "gold"
        headline     = (f"{n_watch} risk factors require monitoring. "
                        f"Review the rebalancing signals and projection scenarios.")
    elif score >= 25:
        health_label = "ELEVATED RISK"
        health_color = "orange"
        headline     = (f"{n_action} items require immediate attention. "
                        f"Portfolio risk is above target — review action items urgently.")
    else:
        health_label = "CRITICAL"
        health_color = "red"
        headline     = (f"Portfolio is in a high-risk state with {n_action} critical alerts. "
                        "Immediate review and likely rebalancing is warranted.")

    return {
        "health_score":  round(score),
        "health_label":  health_label,
        "health_color":  health_color,
        "headline":      headline,
        "n_action":      n_action,
        "n_watch":       n_watch,
        "n_info":        n_info,
        "total_signals": len(advices),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_advisory() -> dict:
    """
    Load state + projections, evaluate all rules, write advice.json.
    """
    # Load inputs
    state = {}
    proj  = {}

    if LIVE_STATE.exists():
        with open(LIVE_STATE) as f:
            state = json.load(f)
        if "error" in state:
            print(f"[ADV] live_state.json has error: {state['error']} — using empty state")
            state = {}
    else:
        print("[ADV] live_state.json not found — run live_optimizer.py first")

    if PROJ_FILE.exists():
        with open(PROJ_FILE) as f:
            proj = json.load(f)
    else:
        print("[ADV] projections.json not found — run projection_engine.py for projection rules")

    if not state:
        print("[ADV] No state data — cannot generate advice")
        print("[ADV] Run: python3 live_optimizer.py --once")
        return {}

    # Validate that state has the expected keys before running rules
    missing_keys = [k for k in ("optimization", "risk", "tickers", "attribution")
                    if k not in state]
    if missing_keys:
        print(f"[ADV] live_state.json is missing keys: {missing_keys}")
        print("[ADV] The live optimizer may have failed mid-run. Re-run:")
        print("[ADV]   python3 live_optimizer.py --once")
        # Still try to run rules — each one guards its own keys
        print("[ADV] Attempting partial advisory with available data …")

    # Evaluate all rules
    t0      = time.perf_counter()
    advices = []
    for rule in RULE_REGISTRY:
        try:
            result = rule(state, proj)
            if result is not None:
                advices.append(result)
        except Exception as e:
            print(f"[ADV] Rule {rule.__name__} failed: {type(e).__name__}: {e}")

    if not advices:
        print("[ADV] No rules triggered — all metrics within thresholds, or state incomplete")
        print(f"[ADV] State keys available: {list(state.keys())}")

    # Sort by priority (lower = more important)
    advices.sort(key=lambda a: a.priority)

    # Generate summary
    summary = generate_summary(advices, state, proj)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"[ADV] Health: {summary['health_label']} (score={summary['health_score']})  "
          f"Signals: {summary['n_action']} action / {summary['n_watch']} watch / "
          f"{summary['n_info']} info  ({elapsed:.1f}ms)")

    output = {
        "generated_at":  datetime.now().isoformat(),
        "portfolio_timestamp": state.get("timestamp", "unknown"),
        "summary":       summary,
        "advice": [
            {
                "category": a.category,
                "severity": a.severity,
                "title":    a.title,
                "body":     a.body,
                "metric":   a.metric,
                "action":   a.action,
                "priority": a.priority,
            }
            for a in advices
        ],
        "thresholds_used": THR,
        "runtime_ms":    round(elapsed, 2),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp = ADVICE_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    tmp.replace(ADVICE_OUT)
    print(f"[ADV] Advice saved → {ADVICE_OUT}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Rule-based portfolio advisory engine")
    parser.add_argument("--watch", action="store_true",
                        help="Run continuously, regenerating advice every 60s")
    parser.add_argument("--interval", type=int, default=60,
                        help="Watch mode interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.watch:
        import signal as _sig
        running = [True]
        _sig.signal(_sig.SIGINT, lambda s, f: running.__setitem__(0, False))
        print(f"[ADV] Watch mode — regenerating every {args.interval}s. Ctrl+C to stop.")
        while running[0]:
            run_advisory()
            for _ in range(args.interval):
                if not running[0]:
                    break
                time.sleep(1)
    else:
        run_advisory()


if __name__ == "__main__":
    main()