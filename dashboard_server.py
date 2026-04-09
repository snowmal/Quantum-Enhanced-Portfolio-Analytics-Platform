"""
dashboard_server.py
===================
Interactive metrics dashboard for the Classical Portfolio Risk System.
Built with Dash + Plotly. Run from the project root:

    python dashboard_server.py

Then open:  http://localhost:8050
"""

import os, json, warnings
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ARTIFACTS = os.path.join(os.path.dirname(__file__), "portfolio_training", "artifacts")
LIVE_REFRESH_MS = 15_000   # milliseconds between live tab auto-refreshes (15s)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(
    bg      = "#07090f",
    panel   = "#0e1320",
    border  = "#1c2640",
    cyan    = "#38bdf8",
    gold    = "#f59e0b",
    green   = "#4ade80",
    red     = "#f87171",
    violet  = "#a78bfa",
    teal    = "#2dd4bf",
    muted   = "#64748b",
    white   = "#e8f0fe",
    text    = "#94a3b8",
    orange  = "#fb923c",
)

FONT = "'IBM Plex Mono', 'Fira Code', monospace"
BODY_FONT = "'IBM Plex Sans', 'Segoe UI', sans-serif"

# ── Load artefacts ─────────────────────────────────────────────────────────────

def _load_json(name):
    path = os.path.join(ARTIFACTS, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv(name):
    path = os.path.join(ARTIFACTS, name)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()


def load_all():
    data = {}
    data["metrics"]    = _load_json("classical_metrics.json")
    data["fhe"]        = _load_json("fhe_comparison.json")
    data["enc_classical"] = _load_json("encrypted_classical_results.json")
    data["live"]       = _load_json("live_state.json")
    data["projections"]= _load_json("projections.json")
    data["advice"]     = _load_json("advice.json")
    data["pipeline"]   = _load_json("pipeline_report.json")
    data["optimizer"]  = _load_json("optimizer_report.json")
    data["risk"]       = _load_json("risk_metrics_report.json")
    data["factor"]     = _load_json("factor_model_report.json")

    data["returns"]    = _load_csv("returns.csv")
    data["bt_returns"] = _load_csv("backtest_portfolio_returns.csv")
    data["frontier"]   = _load_csv("efficient_frontier.csv").reset_index(drop=True) \
                         if os.path.exists(os.path.join(ARTIFACTS, "efficient_frontier.csv")) \
                         else pd.DataFrame()
    data["rolling_risk"]    = _load_csv("rolling_risk.csv")
    data["rolling_weights"] = _load_csv("rolling_weights.csv")
    data["backtest_df"]     = pd.read_csv(os.path.join(ARTIFACTS, "backtest_results.csv"),
                                          parse_dates=["test_start","test_end"]) \
                              if os.path.exists(os.path.join(ARTIFACTS, "backtest_results.csv")) \
                              else pd.DataFrame()
    data["corr"]       = _load_csv("correlation_matrix.csv")
    data["rd"]         = _load_csv("risk_decomposition.csv")
    data["w_class"]    = _load_csv("w_classical.csv")
    return data


# ── Insight engine ─────────────────────────────────────────────────────────────

def generate_insights(data):
    insights = []
    m = data["metrics"].get("overall", {})
    cov = data["metrics"].get("var_coverage", {})
    crisis = data["metrics"].get("crisis_performance", {})
    opt = data["optimizer"]

    sharpe = m.get("sharpe")
    mdd    = m.get("max_drawdown")
    turn   = m.get("avg_turnover")

    if sharpe is not None:
        if sharpe > 0.7:
            insights.append(("✓ Strong risk-adjusted return",
                f"Out-of-sample Sharpe of {sharpe:.2f} exceeds the 0.7 quantum target threshold. "
                "Classical baseline is performing well — sets a high bar for the VQC layer.",
                C["green"], "positive"))
        elif sharpe > 0.4:
            insights.append(("◎ Moderate risk-adjusted return",
                f"Sharpe of {sharpe:.2f} is within the expected 0.4–0.7 classical range. "
                "The quantum VQC layer should aim to push this above 0.7 via nonlinear risk modelling.",
                C["gold"], "neutral"))
        else:
            insights.append(("⚠ Below-target Sharpe",
                f"Sharpe of {sharpe:.2f} is below the 0.4 lower bound. Consider reviewing "
                "your asset universe for survivorship bias or extending the lookback window.",
                C["red"], "warning"))

    if mdd is not None:
        mdd_pct = abs(mdd) * 100
        if mdd_pct > 40:
            insights.append(("⚠ High maximum drawdown",
                f"Max drawdown of {mdd_pct:.1f}% indicates significant tail exposure. "
                "The nonlinear VQC risk penalty should reduce this by better capturing "
                "regime transitions that linear covariance misses.",
                C["red"], "warning"))
        else:
            insights.append(("✓ Drawdown within acceptable range",
                f"Max drawdown of {mdd_pct:.1f}%. Reasonably controlled for a long-only portfolio "
                "spanning 2005–2023 including GFC and COVID.",
                C["green"], "positive"))

    coverage = cov.get("assessment", "")
    br = cov.get("mean_breach_rate")
    if br is not None:
        if coverage == "well-calibrated":
            insights.append(("✓ VaR model well-calibrated",
                f"Parametric VaR breach rate of {br*100:.2f}% is close to the 5% target. "
                "Gaussian assumption is holding for this universe.",
                C["green"], "positive"))
        elif coverage == "under-estimating":
            insights.append(("⚠ VaR under-estimating tail risk",
                f"Breach rate of {br*100:.2f}% exceeds the 5% target. "
                "Fat tails in the return distribution are not fully captured by Gaussian VaR — "
                "this is exactly the gap the historical ES and VQC nonlinear model will address.",
                C["red"], "warning"))
        else:
            insights.append(("◎ VaR overly conservative",
                f"Breach rate of {br*100:.2f}% — VaR is over-estimating risk. "
                "Capital efficiency could be improved with a better-calibrated model.",
                C["gold"], "neutral"))

    if turn is not None:
        if turn > 0.5:
            insights.append(("⚠ High portfolio turnover",
                f"Average rebalance turnover of {turn:.2f} implies significant transaction costs. "
                "Consider adding a turnover penalty λ_t * Σ|Δw_i| to the Markowitz objective.",
                C["gold"], "warning"))
        else:
            insights.append(("✓ Low turnover",
                f"Average turnover of {turn:.2f} per rebalance — portfolio is stable and "
                "transaction-cost-efficient.",
                C["green"], "positive"))

    for name, cs in crisis.items():
        sr = cs.get("sharpe", 0)
        if sr < -0.5:
            insights.append((f"⚠ Severe underperformance: {name}",
                f"Sharpe of {sr:.2f} during {name}. The linear factor model failed to hedge "
                "the nonlinear regime shift. This motivates the VQC's conditional covariance "
                "modelling capability.",
                C["red"], "warning"))

    bl = opt.get("bl_applied", False)
    if bl:
        insights.append(("◎ Black-Litterman applied",
            "Ill-conditioned covariance matrix triggered BL shrinkage. This is normal for "
            "small N portfolios. The Woodbury form kept inversion stable.",
            C["gold"], "neutral"))

    return insights


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _fig_layout(fig, title="", height=340):
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=13, color=C["white"]), x=0.01),
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(family=FONT, color=C["text"], size=11),
        margin=dict(l=48, r=16, t=44, b=36),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                    font=dict(size=10)),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                   tickfont=dict(size=10)),
    )
    return fig


def fig_cumulative_returns(bt_returns, returns):
    fig = go.Figure()

    if not bt_returns.empty:
        col = bt_returns.columns[0] if hasattr(bt_returns, 'columns') else None
        r = bt_returns.iloc[:, 0] if col else bt_returns
        cum = (1 + r).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name="Markowitz OOS",
            line=dict(color=C["gold"], width=2),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.06)",
        ))

    if not returns.empty:
        eq_w = np.ones(len(returns.columns)) / len(returns.columns)
        r_eq = returns @ eq_w
        cum_eq = (1 + r_eq).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_eq.index, y=cum_eq.values,
            name="Equal Weight",
            line=dict(color=C["muted"], width=1, dash="dot"),
        ))

    # Crisis shading
    for name, (cs, ce) in {
        "GFC": ("2008-09-01","2009-03-31"),
        "COVID": ("2020-02-15","2020-05-15"),
        "Bear '22": ("2022-01-01","2022-12-31"),
    }.items():
        fig.add_vrect(x0=cs, x1=ce,
                      fillcolor="rgba(248,113,113,0.07)",
                      line_width=0,
                      annotation_text=name,
                      annotation_position="top left",
                      annotation_font=dict(size=9, color=C["red"]))

    _fig_layout(fig, "Cumulative Portfolio Return (Out-of-Sample)", height=320)
    fig.update_yaxes(title="Growth of $1")
    return fig


def fig_rolling_sharpe(backtest_df):
    if backtest_df.empty or "sharpe" not in backtest_df.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_hline(y=0.7, line_dash="dash", line_color=C["cyan"],
                  annotation_text="Quantum target (0.7)",
                  annotation_font=dict(size=9, color=C["cyan"]))
    fig.add_hline(y=0, line_color=C["red"], line_dash="dot", line_width=1)
    fig.add_trace(go.Bar(
        x=backtest_df["test_end"],
        y=backtest_df["sharpe"],
        name="Rolling Sharpe",
        marker_color=[C["green"] if v > 0 else C["red"]
                      for v in backtest_df["sharpe"]],
        marker_line_width=0,
    ))
    _fig_layout(fig, "Rolling Out-of-Sample Sharpe Ratio", height=280)
    return fig


def fig_rolling_var(rolling_risk):
    if rolling_risk.empty:
        return go.Figure()
    fig = go.Figure()
    if "hist_var_95" in rolling_risk.columns:
        fig.add_trace(go.Scatter(
            x=rolling_risk.index, y=rolling_risk["hist_var_95"] * 100,
            name="Historical VaR 95%",
            line=dict(color=C["red"], width=1.5),
        ))
    if "param_var_95" in rolling_risk.columns:
        fig.add_trace(go.Scatter(
            x=rolling_risk.index, y=rolling_risk["param_var_95"] * 100,
            name="Parametric VaR 95%",
            line=dict(color=C["gold"], width=1.5, dash="dash"),
        ))
    if "hist_es_95" in rolling_risk.columns:
        fig.add_trace(go.Scatter(
            x=rolling_risk.index, y=rolling_risk["hist_es_95"] * 100,
            name="Historical ES 95%",
            line=dict(color=C["violet"], width=1, dash="dot"),
        ))
    _fig_layout(fig, "Rolling VaR & ES (95% Confidence, Daily %)", height=300)
    fig.update_yaxes(title="Daily Loss %")
    return fig


def fig_efficient_frontier(frontier, w_class):
    if frontier.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["ann_vol"] * 100,
        y=frontier["ann_return"] * 100,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color=C["cyan"], width=2),
    ))

    # Key portfolios
    if not w_class.empty and "tangency" in w_class.columns:
        sigma = None
        try:
            sigma = np.load(os.path.join(ARTIFACTS, "sigma_full.npy"))
            mu    = np.load(os.path.join(ARTIFACTS, "mu_annual.npy"))
            w_t   = w_class["tangency"].values
            v_t   = float(np.sqrt(w_t @ sigma @ w_t)) * np.sqrt(252) * 100
            r_t   = float(w_t @ mu) * 100
            fig.add_trace(go.Scatter(x=[v_t], y=[r_t], mode="markers",
                name="Tangency",
                marker=dict(color=C["gold"], size=12, symbol="star")))
            w_m = w_class["min_variance"].values
            v_m = float(np.sqrt(w_m @ sigma @ w_m)) * np.sqrt(252) * 100
            r_m = float(w_m @ mu) * 100
            fig.add_trace(go.Scatter(x=[v_m], y=[r_m], mode="markers",
                name="Min Variance",
                marker=dict(color=C["teal"], size=10, symbol="diamond")))
        except Exception:
            pass

    _fig_layout(fig, "Efficient Frontier", height=320)
    fig.update_xaxes(title="Annual Volatility %")
    fig.update_yaxes(title="Annual Return %")
    return fig


def fig_correlation_heatmap(corr):
    if corr.empty:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, C["red"]], [0.5, C["panel"]], [1, C["cyan"]]],
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    _fig_layout(fig, "Asset Correlation Matrix", height=360)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    return fig


def fig_risk_decomposition(rd):
    if rd.empty or "CRC_pct" not in rd.columns:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=rd.index.tolist(),
        y=rd["CRC_pct"].values,
        marker_color=C["violet"],
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in rd["CRC_pct"].values],
        textposition="outside",
        textfont=dict(size=9, color=C["text"]),
    ))
    _fig_layout(fig, "Component Risk Contribution % (Equal Weight)", height=280)
    fig.update_yaxes(title="% of Portfolio Risk")
    return fig


def fig_rolling_vol(rolling_risk):
    if rolling_risk.empty or "rolling_vol" not in rolling_risk.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_risk.index,
        y=rolling_risk["rolling_vol"] * 100,
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.07)",
        line=dict(color=C["cyan"], width=1.5),
        name="Rolling Ann. Vol %",
    ))
    _fig_layout(fig, "Rolling 252-Day Annualised Volatility (%)", height=260)
    fig.update_yaxes(title="Annual Vol %")
    return fig


def fig_weight_evolution(rolling_weights):
    if rolling_weights.empty:
        return go.Figure()
    fig = go.Figure()
    colors = [C["cyan"], C["gold"], C["green"], C["violet"], C["teal"],
              C["red"], "#f97316", "#ec4899", "#84cc16", "#06b6d4",
              "#8b5cf6", "#14b8a6", "#f59e0b", "#10b981", "#6366f1"]
    for i, col in enumerate(rolling_weights.columns):
        fig.add_trace(go.Scatter(
            x=rolling_weights.index,
            y=rolling_weights[col] * 100,
            name=col,
            stackgroup="one",
            line=dict(width=0),
            fillcolor=colors[i % len(colors)].replace(")", ",0.7)").replace("rgb(", "rgba(")
                       if colors[i % len(colors)].startswith("rgb") else colors[i % len(colors)],
        ))
    _fig_layout(fig, "Rolling Optimal Weight Allocation (%)", height=300)
    fig.update_yaxes(title="Weight %")
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0,
        font=dict(size=9)))
    return fig


def fig_crisis_bar(crisis):
    if not crisis:
        return go.Figure()
    names   = list(crisis.keys())
    sharpes = [crisis[n]["sharpe"]      for n in names]
    returns = [crisis[n]["total_return"] for n in names]
    mdd     = [crisis[n]["max_drawdown"] for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Sharpe", "Total Return %", "Max Drawdown %"])
    colors_s = [C["green"] if v >= 0 else C["red"] for v in sharpes]
    colors_r = [C["green"] if v >= 0 else C["red"] for v in returns]

    fig.add_trace(go.Bar(x=names, y=sharpes, marker_color=colors_s,
                         marker_line_width=0, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=returns, marker_color=colors_r,
                         marker_line_width=0, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=mdd,     marker_color=C["red"],
                         marker_line_width=0, showlegend=False), row=1, col=3)

    _fig_layout(fig, "Crisis-Period Performance", height=280)
    for i in range(1, 4):
        fig.update_xaxes(tickangle=-25, tickfont=dict(size=9), row=1, col=i)
    return fig


# ── KPI card factory ───────────────────────────────────────────────────────────

def kpi_card(label, value, unit="", color=C["cyan"], hint=""):
    return html.Div([
        html.Div(label, style={
            "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
            "textTransform": "uppercase", "color": C["muted"], "marginBottom": "6px",
        }),
        html.Div([
            html.Span(value, style={"color": color, "fontSize": "28px",
                                     "fontWeight": "700", "fontFamily": FONT}),
            html.Span(unit,  style={"color": C["muted"], "fontSize": "13px",
                                     "marginLeft": "4px", "fontFamily": FONT}),
        ]),
        html.Div(hint, style={"fontSize": "10px", "color": C["muted"],
                               "marginTop": "4px", "lineHeight": "1.4",
                               "fontFamily": BODY_FONT}),
    ], style={
        "background": C["panel"],
        "border":     f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding":    "18px 20px",
        "flex":       "1",
        "minWidth":   "140px",
    })


def insight_card(title, body, color, kind):
    border_l = {
        "positive": C["green"],
        "neutral":  C["gold"],
        "warning":  C["red"],
    }.get(kind, C["muted"])
    return html.Div([
        html.Div(title, style={
            "fontFamily": FONT, "fontSize": "11px", "fontWeight": "700",
            "color": color, "marginBottom": "6px",
        }),
        html.Div(body, style={
            "fontSize": "12px", "color": C["text"],
            "lineHeight": "1.6", "fontFamily": BODY_FONT,
        }),
    ], style={
        "background":    C["panel"],
        "border":        f"1px solid {C['border']}",
        "borderLeft":    f"3px solid {border_l}",
        "borderRadius":  "6px",
        "padding":       "14px 16px",
        "marginBottom":  "8px",
    })


def section_header(title, subtitle=""):
    return html.Div([
        html.Div(title, style={
            "fontFamily": FONT, "fontSize": "12px", "letterSpacing": "2px",
            "textTransform": "uppercase", "color": C["cyan"], "fontWeight": "700",
        }),
        html.Div(subtitle, style={
            "fontFamily": BODY_FONT, "fontSize": "12px",
            "color": C["muted"], "marginTop": "4px",
        }),
        html.Hr(style={"borderColor": C["border"], "margin": "10px 0 16px"}),
    ], style={"marginTop": "32px", "marginBottom": "8px"})




# ── FHE helper functions ───────────────────────────────────────────────────────

def _fhe_kpi_row(fhe):
    plain = fhe.get("classical_plaintext", {}) if fhe else {}
    enc   = fhe.get("classical_encrypted", {}) if fhe else {}
    cmp   = fhe.get("comparison", {}) if fhe else {}
    has   = bool(plain and enc and "error" not in enc)
    def _f(d, k, mult=1, prec=4):
        v = d.get(k)
        return f"{float(v)*mult:.{prec}f}" if v is not None else "—"
    rel_err  = cmp.get("variance_relative_error_pct")
    overhead = cmp.get("fhe_overhead_ms")
    err_col  = (C["teal"] if (rel_err or 1) < 0.01 else
                C["gold"] if (rel_err or 1) < 0.1  else C["red"]) if rel_err else C["muted"]
    return html.Div([
        kpi_card("Plaintext σ²_p",  _f(plain,"variance",1,8) if has else "—", "", C["gold"],   "Exact w⊤Σw"),
        kpi_card("Encrypted σ²_p",  _f(enc,  "variance",1,8) if has else "—", "", C["teal"],   "Decrypted CKKS"),
        kpi_card("Variance Error",  f"{rel_err:.4f}" if rel_err is not None else "—", "%", err_col, "Target < 0.1%"),
        kpi_card("FHE Overhead",    f"{overhead:.0f}" if overhead is not None else "—", "ms", C["cyan"], "vs plaintext"),
        kpi_card("Plaintext Sharpe",_f(plain,"sharpe",1,4) if has else "—", "", C["gold"],   "Classical baseline"),
        kpi_card("Encrypted Sharpe",_f(enc,  "sharpe",1,4) if has else "—", "", C["teal"],   "Should match ↑"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"20px"})


def _fig_fhe_metrics(fhe):
    plain = fhe.get("classical_plaintext", {}) if fhe else {}
    enc   = fhe.get("classical_encrypted", {}) if fhe else {}
    if not plain or not enc or "error" in enc:
        fig = go.Figure()
        fig.add_annotation(text="Run: python3 run_fhe_comparison.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Plaintext vs. Encrypted — Risk Metrics", height=320)
        return fig
    keys   = [("Sharpe","sharpe",1,4),("VaR 95%","VaR_95",100,3),
              ("ES Gauss","ES_gaussian",100,3),("σ_p %","sigma_p",100,3),("μ_p %","mu_p",100,4)]
    labels = [k[0] for k in keys]
    pv = [float(plain.get(k[1],0) or 0)*k[2] for k in keys]
    ev = [float(enc.get(k[1],0)   or 0)*k[2] for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Plaintext", x=labels, y=pv,
        marker_color=C["gold"], marker_line_width=0,
        text=[f"{v:.4f}" for v in pv], textposition="outside", textfont=dict(size=9)))
    if bool(enc and "error" not in enc) and ev:
        fig.add_trace(go.Bar(name="CKKS Encrypted", x=labels, y=ev,
            marker_color=C["teal"], marker_line_width=0,
            text=[f"{v:.4f}" for v in ev], textposition="outside", textfont=dict(size=9)))
    if not bool(enc and "error" not in enc):
        fig.add_annotation(
            text="Install TenSEAL for encrypted bars  (pip install tenseal)",
            xref="paper", yref="paper", x=0.5, y=-0.3, showarrow=False,
            font=dict(family=FONT, size=10, color=C["gold"]))
    fig.update_layout(barmode="group")
    _fig_layout(fig, "Risk Metrics — Plaintext vs. CKKS Encrypted", height=340)
    return fig


def _fig_fhe_timing(fhe):
    enc = fhe.get("classical_encrypted", {}) if fhe else {}
    if not enc or "error" in enc:
        fig = go.Figure()
        fig.add_annotation(text="Timing data will appear after running encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "FHE Runtime Breakdown (no data yet)", height=280)
        return _clean_empty_fig(fig)
    t = enc.get("timing", {})
    plain_ms = (fhe.get("classical_plaintext", {}) or {}).get("runtime_ms", 0) or 0
    steps = [
        ("Plaintext",        plain_ms,                               C["gold"]),
        ("Context build",    t.get("context_build_ms",0)    or 0,   C["cyan"]),
        ("Encrypt (Alice)",  t.get("encryption_ms",0)       or 0,   C["violet"]),
        ("Carol CKKS eval",  t.get("carol_evaluation_ms",0) or 0,   C["teal"]),
        ("Decrypt (Alice)",  t.get("decryption_ms",0)       or 0,   C["green"]),
    ]
    fig = go.Figure(go.Bar(
        x=[s[0] for s in steps], y=[s[1] for s in steps],
        marker_color=[s[2] for s in steps], marker_line_width=0,
        text=[f"{s[1]:.1f}ms" for s in steps], textposition="outside",
        textfont=dict(size=10, color=C["text"]),
    ))
    total = enc.get("runtime_ms", 0) or 0
    if total:
        fig.add_hline(y=total, line_dash="dot", line_color=C["red"],
            annotation_text=f"Total encrypted: {total:.0f}ms",
            annotation_font=dict(size=9, color=C["red"]))
    _fig_layout(fig, "FHE Runtime Breakdown (ms)", height=300)
    fig.update_yaxes(title="Milliseconds")
    return fig


def _fig_fhe_variance_accuracy(fhe):
    plain = fhe.get("classical_plaintext", {}) if fhe else {}
    enc   = fhe.get("classical_encrypted", {}) if fhe else {}
    cmp   = fhe.get("comparison", {}) if fhe else {}
    if not plain or not enc or "error" in enc:
        fig = go.Figure()
        fig.add_annotation(text="Run run_fhe_comparison.py to generate data.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Variance Accuracy: Exact vs. Encrypted", height=300)
        return fig
    exact  = float(plain.get("variance", 0) or 0)
    encr   = float(enc.get("variance",  0) or 0)
    rel_e  = float(cmp.get("variance_relative_error_pct", 0) or 0)
    pad    = max(exact * 0.05, 1e-8)
    color  = C["teal"] if rel_e < 0.01 else C["gold"] if rel_e < 0.1 else C["red"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[exact-pad, exact+pad], y=[exact-pad, exact+pad],
        mode="lines", name="Perfect accuracy",
        line=dict(color=C["muted"], dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=[exact], y=[encr], mode="markers",
        name=f"Encrypted (err={rel_e:.4f}%)",
        marker=dict(color=color, size=14, symbol="circle",
                    line=dict(color=C["white"], width=1))))
    fig.add_annotation(x=exact, y=encr, text=f"  err: {rel_e:.6f}%",
        showarrow=False, xanchor="left",
        font=dict(family=FONT, size=10, color=color))
    _fig_layout(fig, "σ²_p: Exact (x) vs. Decrypted CKKS (y)", height=300)
    fig.update_xaxes(title="Plaintext σ²_p")
    fig.update_yaxes(title="Encrypted σ²_p")
    return fig


def _fig_fhe_crc(fhe):
    plain = fhe.get("classical_plaintext", {}) if fhe else {}
    enc   = fhe.get("classical_encrypted", {}) if fhe else {}
    crc_p = plain.get("CRC_pct", []) if plain else []
    crc_e = enc.get("CRC_pct", [])   if (enc and "error" not in enc) else []
    if not crc_p:
        fig = go.Figure()
        fig.add_annotation(text="Risk attribution will appear after running encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Risk Attribution: Plaintext vs. Encrypted (no data yet)", height=280)
        return _clean_empty_fig(fig)
    n = len(crc_p)
    labels = [f"Asset {i+1}" for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Plaintext", x=labels,
        y=[v*100 for v in crc_p],
        marker_color=C["gold"], marker_line_width=0))
    if crc_e:
        fig.add_trace(go.Bar(name="CKKS Encrypted", x=labels,
            y=[v*100 for v in crc_e],
            marker_color=C["teal"], marker_line_width=0, opacity=0.8))
    fig.update_layout(barmode="group")
    _fig_layout(fig, "Component Risk Attribution % — Plaintext vs. Encrypted", height=300)
    fig.update_yaxes(title="% of Portfolio Risk")
    return fig


def _fhe_accuracy_table(fhe):
    if not fhe:
        return html.Div("Run: python3 run_fhe_comparison.py",
            style={"color": C["muted"], "fontFamily": FONT, "fontSize":"11px", "padding":"20px"})
    plain = fhe.get("classical_plaintext", {})
    enc   = fhe.get("classical_encrypted", {})
    cmp   = fhe.get("comparison", {})
    if not plain or not enc or "error" in enc:
        err = enc.get("error","No data") if enc else "No data"
        return html.Div([
            html.Div(f"FHE not yet run. Error: {err}",
                style={"color": C["red"], "fontFamily": FONT, "fontSize":"11px"}),
            html.Div("Run: python3 run_fhe_comparison.py",
                style={"color": C["cyan"], "fontFamily": FONT, "fontSize":"10px", "marginTop":"8px"}),
        ], style={"padding":"20px"})

    def row(label, pk, ek, mult=1, prec=6, threshold=None):
        pv = plain.get(pk); ev = enc.get(ek)
        if pv is None or ev is None: return None
        delta = abs(float(ev)-float(pv))
        if threshold is not None:
            rel = delta/(abs(float(pv))+1e-14)*100
            ok  = rel < threshold
            sc  = C["teal"] if ok else C["red"]
            st  = f"{'✓' if ok else '✗'} {rel:.4f}%"
        else:
            sc = C["muted"]; st = f"{delta*mult:.2e}"
        return html.Tr([
            html.Td(label, style={"color":C["muted"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"20px","paddingBottom":"8px"}),
            html.Td(f"{float(pv)*mult:.{prec}f}", style={"color":C["gold"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"20px","paddingBottom":"8px"}),
            html.Td(f"{float(ev)*mult:.{prec}f}", style={"color":C["teal"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"20px","paddingBottom":"8px"}),
            html.Td(st, style={"color":sc,"fontFamily":FONT,"fontSize":"10px","fontWeight":"700","paddingBottom":"8px"}),
        ])

    rows = [r for r in [
        row("Variance σ²_p",      "variance",    "variance",    1,   8, 0.1),
        row("Volatility σ_p %",   "sigma_p",     "sigma_p",     100, 6, 0.1),
        row("Return μ_p %",       "mu_p",        "mu_p",        100, 6, 0.5),
        row("Sharpe Ratio",       "sharpe",      "sharpe",      1,   4, 1.0),
        row("VaR 95% %",          "VaR_95",      "VaR_95",      100, 4, 0.5),
        row("ES 95% (Gauss) %",   "ES_gaussian", "ES_gaussian", 100, 4, 0.5),
    ] if r is not None]

    rel_err  = cmp.get("variance_relative_error_pct", 0) or 0
    overhead = cmp.get("fhe_overhead_ms", 0) or 0
    gc = C["teal"] if rel_err < 0.1 else C["red"]
    gt = (f"✓ PASS — Variance relative error {rel_err:.4f}% < 0.1%"
          if rel_err < 0.1 else
          f"✗ FAIL — Variance relative error {rel_err:.4f}% ≥ 0.1%")

    return html.Div([
        html.Div("METRIC-BY-METRIC ACCURACY", style={
            "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
            "color":C["muted"],"marginBottom":"12px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"fontFamily":FONT,"fontSize":"10px",
                    "paddingBottom":"8px","paddingRight":"20px","textAlign":"left",
                    "color":col})
                for h, col in [("Metric",C["white"]),("Plaintext",C["gold"]),
                               ("Encrypted",C["teal"]),("Δ / Rel. Error",C["white"])]
            ])),
            html.Tbody(rows),
        ]),
        html.Hr(style={"borderColor":C["border"],"margin":"12px 0"}),
        html.Div(gt, style={"fontFamily":FONT,"fontSize":"11px","color":gc,"fontWeight":"700","marginBottom":"6px"}),
        html.Div(f"FHE overhead: {overhead:.0f}ms  |  poly_mod_degree=16384  scale=2^40  depth=4",
            style={"fontFamily":FONT,"fontSize":"10px","color":C["muted"]}),
    ], style={"padding":"20px"})


def _build_fhe_tab(fhe):
    has = bool(fhe and fhe.get("classical_plaintext") and
               fhe.get("classical_encrypted") and
               "error" not in fhe.get("classical_encrypted", {}))
    return html.Div(style={"marginTop": "20px"}, children=[
        _fhe_kpi_row(fhe),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            html.Div([
                html.Div([
                    html.Div("Metrics: Plaintext vs. Encrypted", style={
                        "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                        "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                    import_dcc_graph(_fig_fhe_metrics(fhe)),
                ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
                          "borderRadius":"10px","marginBottom":"16px"}),
                html.Div([
                    html.Div("FHE Runtime Breakdown", style={
                        "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                        "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                    import_dcc_graph(_fig_fhe_timing(fhe)),
                ], style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
            ]),
            html.Div([
                html.Div([_fhe_accuracy_table(fhe)],
                    style={"background":C["panel"],"border":f"1px solid {C['border']}",
                           "borderRadius":"10px","marginBottom":"16px"}),
                html.Div([
                    html.Div("Variance Accuracy", style={
                        "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                        "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                    import_dcc_graph(_fig_fhe_variance_accuracy(fhe)),
                ], style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
            ]),
        ]),
        html.Div([
            html.Div("Risk Attribution: Plaintext vs. Encrypted", style={
                "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
            import_dcc_graph(_fig_fhe_crc(fhe)),
        ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
                  "borderRadius":"10px","marginTop":"16px"}),
        html.Div([
            html.Div("HOW TO POPULATE THIS TAB", style={
                "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                "color":C["muted"],"marginBottom":"10px"}),
            html.Pre(
                "# After classical pipeline completes:\n"
                "python3 main_classical.py --fhe-only\n\n"
                "# Quick demo (no real data needed):\n"
                "python3 run_fhe_comparison.py --demo",
                style={"background":C["panel"],"border":f"1px solid {C['border']}",
                       "borderRadius":"6px","padding":"16px","fontSize":"11px",
                       "fontFamily":FONT,"color":C["cyan"],"margin":"0"}),
        ], style={"marginTop":"16px"} if not has else {"display":"none"}),
    ])




# ── ENCRYPTED CLASSICAL comparison helpers ─────────────────────────────────────

def _ec_status_badge(ec: dict) -> str:
    if not ec:
        return "not run"
    enc = ec.get("classical_encrypted", {})
    if enc and "error" in enc:
        if "tenseal" in enc.get("error","").lower():
            return "plaintext only  (pip install tenseal  to enable encryption)"
        return f"error: {enc.get('error','')[:60]}"
    if ec.get("overall_pass") is True:
        return f"PASS  {ec.get('portfolio','?')}  ({ec.get('generated_at','')[:19]})"
    if ec.get("overall_pass") is False:
        return "FAIL  check CKKS params"
    plain = ec.get("classical_plaintext", {})
    if plain:
        return f"plaintext computed  ({ec.get('generated_at','')[:19]})"
    return "loaded"


def _fig_ec_metrics(ec: dict):
    plain   = ec.get("classical_plaintext", {}) if ec else {}
    enc     = ec.get("classical_encrypted", {}) if ec else {}
    has_enc = bool(enc and "error" not in enc)
    if not plain:
        fig = go.Figure()
        fig.add_annotation(
            text="Run: python3 encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Plaintext vs. Encrypted -- All Metrics", height=340)
        return fig
    metrics = [
        ("Sharpe",   "ann_sharpe",     1, 4),
        ("VaR 95%",  "var_95",         100, 3),
        ("ES Gauss", "es_95_gaussian", 100, 3),
        ("ES Hist",  "es_95_hist",     100, 3),
        ("Vol %",    "ann_vol_pct",    1,   3),
        ("Ret %",    "ann_return_pct", 1,   3),
    ]
    labels = [m[0] for m in metrics]
    pv = [float(plain.get(m[1], 0) or 0) * m[2] for m in metrics]
    ev = [float(enc.get(m[1], 0) or 0) * m[2] for m in metrics] if has_enc else []
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Plaintext", x=labels, y=pv,
        marker_color=C["gold"], marker_line_width=0,
        text=[f"{v:.4f}" for v in pv], textposition="outside",
        textfont=dict(size=9)))
    fig.add_trace(go.Bar(name="CKKS Encrypted", x=labels, y=ev,
        marker_color=C["teal"], marker_line_width=0,
        text=[f"{v:.4f}" for v in ev], textposition="outside",
        textfont=dict(size=9)))
    fig.update_layout(barmode="group")
    _title = ("Risk Metrics: Plaintext vs. CKKS Encrypted" if has_enc
              else "Plaintext Risk Metrics  (pip install tenseal  for encrypted)")
    _fig_layout(fig, _title, height=360)
    return fig


def _fig_ec_accuracy(ec: dict):
    acc = ec.get("accuracy", {}) if ec else {}
    keys = ["variance","sigma_p","mu_p","sharpe","var_95","es_95_gaussian","ann_sharpe"]
    labels, errors, colors = [], [], []
    for k in keys:
        a = acc.get(k)
        if not isinstance(a, dict):
            continue
        labels.append(a.get("label", k)[:20])
        rel = a.get("rel_error_pct", 0)
        errors.append(rel)
        colors.append(C["teal"] if a.get("pass") else C["red"])
    if not labels:
        fig = go.Figure()
        fig.add_annotation(text="Accuracy data will appear after running encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Relative Error % per Metric (no data yet)", height=280)
        return _clean_empty_fig(fig)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=errors,
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.4f}%" for v in errors],
        textposition="outside", textfont=dict(size=9)))
    fig.add_hline(y=0.1, line_dash="dash", line_color=C["gold"],
                  annotation_text="0.1% gate",
                  annotation_font=dict(size=9, color=C["gold"]))
    _fig_layout(fig, "CKKS Relative Error % (all bars below 0.1% = pass)", height=300)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    fig.update_yaxes(title="Relative Error %")
    return fig


def _fig_ec_timing(ec: dict):
    enc = ec.get("classical_encrypted", {}) if ec else {}
    t   = enc.get("timings", {}) if enc else {}
    plain_ms = (ec.get("classical_plaintext", {}) or {}).get("runtime_ms", 0) or 0
    if not t:
        fig = go.Figure()
        fig.add_annotation(text="Timing data will appear after running encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "FHE Runtime Breakdown (no data yet)", height=280)
        return _clean_empty_fig(fig)
    steps = [
        ("Plaintext",       plain_ms,                           C["gold"]),
        ("Context build",   t.get("context_build_ms", 0) or 0, C["cyan"]),
        ("Alice encrypt",   t.get("encryption_ms",    0) or 0, C["violet"]),
        ("Carol poly eval", t.get("carol_eval_ms",    0) or 0, C["teal"]),
        ("Alice decrypt",   t.get("decryption_ms",    0) or 0, C["green"]),
    ]
    fig = go.Figure(go.Bar(
        x=[s[0] for s in steps], y=[s[1] for s in steps],
        marker_color=[s[2] for s in steps], marker_line_width=0,
        text=[f"{s[1]:.1f}ms" for s in steps],
        textposition="outside", textfont=dict(size=10, color=C["text"]),
    ))
    total = enc.get("runtime_ms", 0) or 0
    if total:
        fig.add_hline(y=total, line_dash="dot", line_color=C["red"],
                      annotation_text=f"Total FHE: {total:.0f}ms",
                      annotation_font=dict(size=9, color=C["red"]))
    _fig_layout(fig, "FHE Runtime Breakdown (ms)", height=300)
    fig.update_yaxes(title="Milliseconds")
    return fig


def _fig_ec_crc(ec: dict):
    plain   = ec.get("classical_plaintext", {}) if ec else {}
    enc     = ec.get("classical_encrypted", {}) if ec else {}
    tickers = ec.get("tickers", []) if ec else []
    crc_p   = plain.get("CRC_pct", [])
    crc_e   = enc.get("CRC_pct",   []) if enc and "error" not in enc else []
    if not crc_p or not tickers:
        fig = go.Figure()
        fig.add_annotation(text="Risk attribution will appear after running encrypted_classical.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=FONT, size=12, color=C["muted"]))
        _fig_layout(fig, "Risk Attribution: Plaintext vs. Encrypted (no data yet)", height=280)
        return _clean_empty_fig(fig)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Plaintext", x=tickers,
        y=crc_p, marker_color=C["gold"], marker_line_width=0))
    if crc_e:
        fig.add_trace(go.Bar(name="CKKS Encrypted", x=tickers,
            y=crc_e, marker_color=C["teal"], marker_line_width=0, opacity=0.85))
    fig.update_layout(barmode="group")
    _fig_layout(fig, "Component Risk Attribution %: Plaintext vs. Encrypted", height=300)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    fig.update_yaxes(title="% of Portfolio Risk")
    return fig


def _ec_accuracy_table(ec: dict):
    if not ec:
        return html.Div("Run: python3 encrypted_classical.py",
            style={"color": C["muted"], "fontFamily": FONT,
                   "fontSize": "11px", "padding": "20px"})
    plain = ec.get("classical_plaintext", {})
    enc   = ec.get("classical_encrypted", {})
    acc   = ec.get("accuracy", {})
    if not plain or not enc or "error" in enc:
        err = enc.get("error", "No data") if enc else "No data"
        return html.Div([
            html.Div(f"Error: {err}",
                     style={"color": C["red"], "fontFamily": FONT,
                            "fontSize": "11px", "padding": "16px"}),
            html.Div("Run: python3 encrypted_classical.py",
                     style={"color": C["cyan"], "fontFamily": FONT,
                            "fontSize": "10px", "padding": "0 16px 16px"}),
        ])
    rows = []
    for k in ["variance","sigma_p","mu_p","sharpe","var_95","es_95_gaussian","ann_sharpe"]:
        a = acc.get(k)
        if not isinstance(a, dict):
            continue
        ok     = a.get("pass", False)
        sc     = C["teal"] if ok else C["red"]
        status = ("OK " if ok else "!! ") + f"{a.get('rel_error_pct',0):.4f}%"
        rows.append(html.Tr([
            html.Td(a.get("label", k),
                    style={"color": C["muted"], "fontFamily": FONT, "fontSize": "10px",
                           "paddingRight": "16px", "paddingBottom": "8px"}),
            html.Td(f"{float(a.get('plaintext',0)):.6f}",
                    style={"color": C["gold"], "fontFamily": FONT, "fontSize": "10px",
                           "paddingRight": "16px", "paddingBottom": "8px"}),
            html.Td(f"{float(a.get('encrypted',0)):.6f}",
                    style={"color": C["teal"], "fontFamily": FONT, "fontSize": "10px",
                           "paddingRight": "16px", "paddingBottom": "8px"}),
            html.Td(status, style={"color": sc, "fontFamily": FONT, "fontSize": "10px",
                                    "fontWeight": "700", "paddingBottom": "8px"}),
        ]))
    overall = ec.get("overall_pass", False)
    gc  = C["teal"] if overall else C["red"]
    gt  = "ALL ACCURACY GATES PASSED" if overall else "VARIANCE ERROR EXCEEDS 0.1%"
    ckks = enc.get("ckks_params", {})
    params_str = (f"poly_mod={ckks.get('poly_modulus_degree','?')}  "
                  f"scale=2^{ckks.get('scale_exp','?')}  "
                  f"depth={ckks.get('levels_available','?')}")
    return html.Div([
        html.Div("METRIC-BY-METRIC ACCURACY", style={
            "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
            "color": C["muted"], "marginBottom": "12px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"color": c, "fontFamily": FONT, "fontSize": "10px",
                                  "paddingBottom": "8px", "paddingRight": "16px",
                                  "textAlign": "left"})
                for h, c in [("Metric", C["white"]), ("Plaintext", C["gold"]),
                              ("Encrypted", C["teal"]), ("Error", C["white"])]
            ])),
            html.Tbody(rows),
        ]),
        html.Hr(style={"borderColor": C["border"], "margin": "12px 0"}),
        html.Div(gt, style={"fontFamily": FONT, "fontSize": "11px",
                             "color": gc, "fontWeight": "700", "marginBottom": "6px"}),
        html.Div(params_str, style={"fontFamily": FONT, "fontSize": "10px",
                                     "color": C["muted"]}),
    ], style={"padding": "16px"})


def _ec_kpi_row(ec: dict):
    plain = ec.get("classical_plaintext", {}) if ec else {}
    enc   = ec.get("classical_encrypted", {}) if ec else {}
    acc   = ec.get("accuracy", {}) if ec else {}
    has   = bool(plain and enc and "error" not in enc)
    def _f(d, k, mult=1, prec=4):
        v = d.get(k)
        return f"{float(v)*mult:.{prec}f}" if v is not None else "--"
    rel_err = acc.get("variance", {}).get("rel_error_pct") if isinstance(acc.get("variance"), dict) else None
    overhead = (enc.get("runtime_ms", 0) or 0) - (plain.get("runtime_ms", 0) or 0)
    err_col = (C["teal"] if (rel_err or 1) < 0.01 else
               C["gold"] if (rel_err or 1) < 0.1 else C["red"]) if rel_err else C["muted"]
    portfolio_label = ec.get("portfolio", "--") if ec else "--"
    return html.Div([
        kpi_card("Portfolio",         portfolio_label, "",   C["cyan"],  "Active portfolio"),
        kpi_card("Plaintext Variance",_f(plain,"variance",1,8) if has else "--", "", C["gold"], "Exact w Sigma w"),
        kpi_card("Encrypted Variance",_f(enc,  "variance",1,8) if has else "--", "", C["teal"], "Decrypted CKKS"),
        kpi_card("Variance Error",    f"{rel_err:.4f}" if rel_err is not None else "--", "%", err_col, "Target < 0.1%"),
        kpi_card("FHE Overhead",      f"{overhead:.0f}" if has else "--", "ms", C["cyan"], "vs plaintext"),
        kpi_card("Gate",
                 "PASS" if ec.get("overall_pass") else ("FAIL" if ec else "--"),
                 "", C["teal"] if ec.get("overall_pass") else C["red"], "Accuracy gate"),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"})


def _tenseal_available() -> bool:
    try:
        import tenseal  # noqa
        return True
    except ImportError:
        return False


def _build_fhe_tab(fhe: dict, ec: dict = None):
    ec      = ec or {}
    has_ec  = bool(ec and ec.get("classical_plaintext") and ec.get("classical_encrypted")
                   and "error" not in ec.get("classical_encrypted", {}))
    has_fhe = bool(fhe and fhe.get("classical_plaintext"))
    tenseal_ok = _tenseal_available()

    # ── TenSEAL not installed banner ──────────────────────────────────────────
    tenseal_banner = html.Div() if tenseal_ok else html.Div([
        html.Div([
            html.Div("TENSEAL NOT INSTALLED", style={
                "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "2px",
                "color": C["gold"], "fontWeight": "700", "marginBottom": "8px"}),
            html.Div(
                "TenSEAL (CKKS homomorphic encryption library) is required to run "
                "the encrypted pipeline. The classical and live tabs work without it.",
                style={"fontFamily": BODY_FONT, "fontSize": "12px", "color": C["text"],
                       "lineHeight": "1.6", "marginBottom": "12px"}),
            html.Pre(
                "pip install tenseal",
                style={"background": C["bg"], "border": f"1px solid {C['border']}",
                       "borderRadius": "6px", "padding": "10px 14px",
                       "fontSize": "12px", "fontFamily": FONT,
                       "color": C["cyan"], "display": "inline-block"}),
            html.Div(
                "After installing, run the encrypted pipeline, then restart the dashboard:",
                style={"fontFamily": BODY_FONT, "fontSize": "12px", "color": C["muted"],
                       "marginTop": "12px", "marginBottom": "8px"}),
            html.Pre(
                "python3 encrypted_classical.py --all-portfolios",
                style={"background": C["bg"], "border": f"1px solid {C['border']}",
                       "borderRadius": "6px", "padding": "10px 14px",
                       "fontSize": "12px", "fontFamily": FONT,
                       "color": C["cyan"], "display": "inline-block"}),
        ], style={
            "background": "rgba(245,158,11,0.06)",
            "border": f"1px solid rgba(245,158,11,0.3)",
            "borderLeft": f"4px solid {C['gold']}",
            "borderRadius": "8px", "padding": "20px 24px", "marginBottom": "24px",
        })
    ], style={"marginTop": "20px"})

    instructions = html.Div([
        html.Div("HOW TO POPULATE THIS TAB", style={
            "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
            "color": C["muted"], "marginBottom": "10px"}),
        html.Pre(
            "# 1. Install TenSEAL (once):\n"
            "pip install tenseal\n\n"
            "# 2. Run classical pipeline (if not done):\n"
            "python3 main_classical.py --skip-fhe\n\n"
            "# 3. Build FHE polynomial bridge:\n"
            "python3 build_classical_polynomial.py\n\n"
            "# 4. Run full encrypted classical pipeline:\n"
            "python3 encrypted_classical.py --all-portfolios\n\n"
            "# Or run everything at once from fresh start:\n"
            "python3 run_platform.py",
            style={"background": C["panel"], "border": f"1px solid {C['border']}",
                   "borderRadius": "6px", "padding": "16px", "fontSize": "11px",
                   "fontFamily": FONT, "color": C["cyan"], "margin": "0"}),
    ], style={"marginTop": "16px"})

    section1 = html.Div([tenseal_banner,
        html.Div([
            html.Div("ENCRYPTED CLASSICAL PIPELINE", style={
                "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "2px",
                "color": C["teal"], "fontWeight": "700", "marginBottom": "4px"}),
            html.Div(f"encrypted_classical.py  Status: {_ec_status_badge(ec)}",
                style={"fontFamily": FONT, "fontSize": "9px", "color": C["muted"]}),
        ], style={"marginBottom": "16px"}),
        _ec_kpi_row(ec),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
        children=[
            html.Div([
                html.Div([
                    html.Div("All Risk Metrics: Plaintext vs. Encrypted", style={
                        "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                        "textTransform": "uppercase", "color": C["muted"],
                        "padding": "12px 16px 0"}),
                    import_dcc_graph(_fig_ec_metrics(ec)),
                ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                          "borderRadius": "10px", "marginBottom": "16px"}),
                html.Div([
                    html.Div("FHE Runtime Breakdown", style={
                        "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                        "textTransform": "uppercase", "color": C["muted"],
                        "padding": "12px 16px 0"}),
                    import_dcc_graph(_fig_ec_timing(ec)),
                ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                          "borderRadius": "10px"}),
            ]),
            html.Div([
                html.Div([_ec_accuracy_table(ec)],
                    style={"background": C["panel"], "border": f"1px solid {C['border']}",
                           "borderRadius": "10px", "marginBottom": "16px"}),
                html.Div([
                    html.Div("Relative Error % per Metric", style={
                        "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                        "textTransform": "uppercase", "color": C["muted"],
                        "padding": "12px 16px 0"}),
                    import_dcc_graph(_fig_ec_accuracy(ec)),
                ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                          "borderRadius": "10px"}),
            ]),
        ]),
        html.Div([
            html.Div("Risk Attribution: Plaintext vs. Encrypted", style={
                "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                "textTransform": "uppercase", "color": C["muted"],
                "padding": "12px 16px 0"}),
            import_dcc_graph(_fig_ec_crc(ec)),
        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                  "borderRadius": "10px", "marginTop": "16px"}),
        instructions if not has_ec else html.Div(),
    ], style={"marginBottom": "32px"})

    section2 = html.Div([
        html.Hr(style={"borderColor": C["border"], "margin": "24px 0 20px"}),
        html.Div([
            html.Div("QUICK ACCURACY CHECK  (run_fhe_comparison.py)", style={
                "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "2px",
                "color": C["muted"], "fontWeight": "700", "marginBottom": "4px"}),
            html.Div("Lightweight variance-only comparison using the saved polynomial.",
                style={"fontFamily": FONT, "fontSize": "9px", "color": C["muted"]}),
        ], style={"marginBottom": "16px"}),
        _fhe_kpi_row(fhe),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
        children=[
            html.Div([
                html.Div("Metrics: Plaintext vs. Encrypted", style={
                    "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                    "textTransform": "uppercase", "color": C["muted"],
                    "padding": "12px 16px 0"}),
                import_dcc_graph(_fig_fhe_metrics(fhe)),
            ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                      "borderRadius": "10px"}),
            html.Div([_fhe_accuracy_table(fhe)],
                style={"background": C["panel"], "border": f"1px solid {C['border']}",
                       "borderRadius": "10px"}),
        ]),
    ]) if has_fhe else html.Div()

    return html.Div(style={"marginTop": "20px"}, children=[section1, section2])


def _clean_empty_fig(fig):
    """Remove visible axes from a placeholder empty figure."""
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    return fig


def import_dcc_graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False})



# ── LIVE state helpers ─────────────────────────────────────────────────────────

def _live_status_badge(live: dict):
    if not live or "error" in live:
        txt = "LIVE: offline"; col = C["red"]
    else:
        ts  = live.get("timestamp","")[:19].replace("T"," ")
        mo  = live.get("market_open", False)
        txt = f"LIVE ● {ts}" if mo else f"LIVE (mkt closed) {ts}"
        col = C["teal"] if mo else C["muted"]
    return html.Span(txt, style={"fontFamily": FONT, "fontSize": "9px",
                                  "color": col, "marginLeft": "12px"})


def _live_kpi_row(live: dict):
    if not live or "error" in live:
        return html.Div("Run: python3 live_optimizer.py to populate live data.",
            style={"color": C["muted"], "fontFamily": FONT, "fontSize":"11px",
                   "padding":"20px"})
    opt  = live.get("optimization", {})
    risk = live.get("risk", {})
    prt  = live.get("portfolio_intraday_pct", 0) or 0
    def kv(label, val, unit="", col=C["cyan"], hint=""):
        return kpi_card(label, val, unit, col, hint)
    col_sr = (C["green"] if (opt.get("ann_sharpe",0) or 0) > 0.7 else
              C["gold"]  if (opt.get("ann_sharpe",0) or 0) > 0.2 else C["red"])
    col_id = C["green"] if prt >= 0 else C["red"]
    return html.Div([
        kv("Live Sharpe",        f"{opt.get('ann_sharpe',0):.3f}",        "", col_sr,   "Annualised"),
        kv("Ann. Return",        f"{opt.get('ann_return_pct',0):.2f}",    "%", C["cyan"],"Live estimate"),
        kv("Ann. Vol",           f"{opt.get('ann_vol_pct',0):.2f}",       "%", C["gold"],"Live estimate"),
        kv("Daily VaR 95%",      f"{risk.get('var_95_pct',0):.2f}",       "%", C["red"], "Historical"),
        kv("Portfolio Today",    f"{prt:+.3f}",                            "%", col_id,  "Intraday return"),
        kv("Optimizer ms",       f"{opt.get('runtime_ms',0):.0f}",        "ms",C["muted"],"Last cycle"),
    ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"20px"})


def _fig_live_weights(live: dict):
    if not live or "error" in live or "attribution" not in live:
        return go.Figure()
    attrib  = live["attribution"]
    tickers = list(attrib.keys())
    weights = [attrib[t]["weight_pct"] for t in tickers]
    crc_pct = [attrib[t]["CRC_pct"]   for t in tickers]
    colors  = [C["cyan"], C["gold"], C["green"], C["violet"], C["teal"],
               C["red"], "#f97316", "#ec4899", "#84cc16", "#06b6d4",
               "#8b5cf6", "#14b8a6", "#f59e0b", "#10b981", "#6366f1"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Portfolio Weights (%)", "Risk Contribution (%)"])
    for i, (t, w, c) in enumerate(zip(tickers, weights, crc_pct)):
        col = colors[i % len(colors)]
        fig.add_trace(go.Bar(name=t, x=[t], y=[w], marker_color=col,
                             marker_line_width=0, showlegend=False,
                             text=[f"{w:.1f}%"], textposition="outside",
                             textfont=dict(size=9)), row=1, col=1)
        fig.add_trace(go.Bar(name=t, x=[t], y=[c], marker_color=col,
                             marker_line_width=0, showlegend=True,
                             text=[f"{c:.1f}%"], textposition="outside",
                             textfont=dict(size=9)), row=1, col=2)
    _fig_layout(fig, "Live Optimal Weights & Risk Attribution", height=340)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def _fig_intraday(live: dict):
    if not live or "error" in live:
        return go.Figure()
    intra   = live.get("intraday_return_pct", {})
    tickers = list(intra.keys())
    values  = [intra[t] for t in tickers]
    colors  = [C["green"] if v >= 0 else C["red"] for v in values]
    fig = go.Figure(go.Bar(
        x=tickers, y=values, marker_color=colors, marker_line_width=0,
        text=[f"{v:+.2f}%" for v in values], textposition="outside",
        textfont=dict(size=9, color=C["text"]),
    ))
    fig.add_hline(y=0, line_color=C["muted"], line_width=1)
    _fig_layout(fig, "Intraday Returns by Asset (%)", height=280)
    fig.update_yaxes(title="Intraday %")
    return fig


def _fig_momentum(live: dict):
    if not live or "error" in live or "momentum" not in live:
        return go.Figure()
    mom = live["momentum"]
    tickers = list(mom.keys())
    m5  = [mom[t]["5d"]  for t in tickers]
    m20 = [mom[t]["20d"] for t in tickers]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="5-day return %",  x=tickers, y=m5,
                         marker_color=C["cyan"], marker_line_width=0))
    fig.add_trace(go.Bar(name="20-day return %", x=tickers, y=m20,
                         marker_color=C["gold"], marker_line_width=0))
    fig.add_hline(y=0, line_color=C["muted"], line_width=1)
    fig.update_layout(barmode="group")
    _fig_layout(fig, "Momentum Signals — 5-Day vs 20-Day Return (%)", height=280)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def _live_attribution_table(live: dict):
    if not live or "error" in live or "attribution" not in live:
        return html.Div("No live data.", style={"color":C["muted"],"fontFamily":FONT,
                                                 "fontSize":"11px","padding":"20px"})
    attrib = live["attribution"]
    mom    = live.get("momentum", {})
    rows   = []
    for t, vals in attrib.items():
        sig   = mom.get(t, {}).get("signal", "—")
        sig_c = C["green"] if sig == "bullish" else C["red"] if sig == "bearish" else C["muted"]
        rows.append(html.Tr([
            html.Td(t,    style={"color":C["cyan"],  "fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{vals['weight_pct']:.2f}%", style={"color":C["white"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{vals['CRC_pct']:.2f}%",    style={"color":C["violet"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{vals['MRC']:.5f}",          style={"color":C["gold"],  "fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(sig, style={"color":sig_c,"fontFamily":FONT,"fontSize":"10px","fontWeight":"700","paddingBottom":"8px"}),
        ]))
    return html.Div([
        html.Div("LIVE ATTRIBUTION & MOMENTUM", style={"fontFamily":FONT,"fontSize":"9px",
                 "letterSpacing":"2px","color":C["muted"],"marginBottom":"12px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"color":C["muted"],"fontFamily":FONT,"fontSize":"10px",
                                  "paddingBottom":"8px","paddingRight":"16px","textAlign":"left"})
                for h in ["Ticker","Weight","CRC %","MRC","Momentum"]
            ])),
            html.Tbody(rows),
        ]),
    ], style={"padding":"16px"})


def _build_live_tab(live: dict):
    has = bool(live and "error" not in live and "optimization" in live)
    no_data = html.Div([
        html.Div("LIVE OPTIMIZER NOT RUNNING", style={"fontFamily":FONT,"fontSize":"9px",
                 "letterSpacing":"2px","color":C["muted"],"marginBottom":"10px"}),
        html.Pre(
            "# Start the live optimizer in a separate terminal:\n"
            "python3 live_optimizer.py\n\n"
            "# Or for one cycle (testing):\n"
            "python3 live_optimizer.py --once\n\n"
            "# HPC mode (parallel lambda scan):\n"
            "python3 live_optimizer.py --lambda-scan --n-jobs -1",
            style={"background":C["panel"],"border":f"1px solid {C['border']}",
                   "borderRadius":"6px","padding":"16px","fontSize":"11px",
                   "fontFamily":FONT,"color":C["cyan"],"margin":"0"}),
    ], style={"marginTop":"20px"})

    if not has:
        return html.Div([no_data], style={"marginTop":"20px"})

    return html.Div(style={"marginTop":"20px"}, children=[
        _live_kpi_row(live),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            html.Div([
                html.Div("Live Weights & Risk Attribution", style={
                    "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                    "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                import_dcc_graph(_fig_live_weights(live)),
            ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
                      "borderRadius":"10px","gridColumn":"1 / -1"}),
            html.Div([
                html.Div("Intraday Returns", style={
                    "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                    "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                import_dcc_graph(_fig_intraday(live)),
            ], style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
            html.Div([
                html.Div("Momentum Signals", style={
                    "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                    "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                import_dcc_graph(_fig_momentum(live)),
            ], style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
        ]),
        html.Div([_live_attribution_table(live)],
            style={"background":C["panel"],"border":f"1px solid {C['border']}",
                   "borderRadius":"10px","marginTop":"16px"}),
    ])


# ── ADVISORY helpers ────────────────────────────────────────────────────────────

SEVERITY_COLORS = {"ACTION": C["red"], "WATCH": C["gold"], "INFO": C["green"]}
SEVERITY_BG     = {"ACTION": "rgba(248,113,114,0.06)", "WATCH": "rgba(245,158,11,0.06)",
                    "INFO":   "rgba(74,222,128,0.06)"}

def _health_gauge(summary: dict):
    score = summary.get("health_score", 0)
    label = summary.get("health_label", "UNKNOWN")
    col   = {"HEALTHY": C["green"], "CAUTION": C["gold"],
              "ELEVATED RISK": C["orange"], "CRITICAL": C["red"]}.get(label, C["muted"])
    return html.Div([
        html.Div("PORTFOLIO HEALTH", style={"fontFamily":FONT,"fontSize":"9px",
                 "letterSpacing":"2px","color":C["muted"],"marginBottom":"8px"}),
        html.Div([
            html.Span(f"{score}", style={"fontFamily":FONT,"fontSize":"48px",
                                          "fontWeight":"700","color":col}),
            html.Span("/100", style={"fontFamily":FONT,"fontSize":"16px","color":C["muted"]}),
        ]),
        html.Div(label, style={"fontFamily":FONT,"fontSize":"12px","color":col,
                               "fontWeight":"700","letterSpacing":"2px","marginTop":"4px"}),
        html.Div(summary.get("headline",""), style={"fontFamily":BODY_FONT,"fontSize":"12px",
                 "color":C["text"],"lineHeight":"1.6","marginTop":"10px"}),
        html.Div([
            html.Span(f"▲ {summary.get('n_action',0)} ACTION",
                      style={"color":C["red"],"fontFamily":FONT,"fontSize":"10px","marginRight":"16px"}),
            html.Span(f"◎ {summary.get('n_watch',0)} WATCH",
                      style={"color":C["gold"],"fontFamily":FONT,"fontSize":"10px","marginRight":"16px"}),
            html.Span(f"✓ {summary.get('n_info',0)} INFO",
                      style={"color":C["green"],"fontFamily":FONT,"fontSize":"10px"}),
        ], style={"marginTop":"12px"}),
    ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
              "borderLeft":f"4px solid {col}","borderRadius":"8px",
              "padding":"20px 24px","marginBottom":"20px"})


def _advice_card(a: dict):
    sev = a.get("severity","INFO")
    col = SEVERITY_COLORS.get(sev, C["muted"])
    bg  = SEVERITY_BG.get(sev, "transparent")
    return html.Div([
        html.Div([
            html.Span(f"[{sev}] ", style={"color":col,"fontFamily":FONT,
                                           "fontSize":"10px","fontWeight":"700"}),
            html.Span(f"[{a.get('category','')}]", style={"color":C["muted"],
                      "fontFamily":FONT,"fontSize":"9px","marginLeft":"4px"}),
        ], style={"marginBottom":"6px"}),
        html.Div(a.get("title",""), style={"fontFamily":FONT,"fontSize":"12px",
                                            "fontWeight":"700","color":C["white"],
                                            "marginBottom":"8px"}),
        html.Div(a.get("body",""),  style={"fontFamily":BODY_FONT,"fontSize":"11px",
                                            "color":C["text"],"lineHeight":"1.6",
                                            "marginBottom":"10px"}),
        html.Div([
            html.Span("Metric: ", style={"fontFamily":FONT,"fontSize":"9px","color":C["muted"]}),
            html.Span(a.get("metric",""), style={"fontFamily":FONT,"fontSize":"9px","color":col}),
        ], style={"marginBottom":"6px"}),
        html.Div([
            html.Span("Action: ", style={"fontFamily":FONT,"fontSize":"9px","color":C["muted"]}),
            html.Span(a.get("action",""), style={"fontFamily":BODY_FONT,"fontSize":"11px","color":C["white"]}),
        ]),
    ], style={"background":bg,"border":f"1px solid {col}","borderLeft":f"3px solid {col}",
              "borderRadius":"8px","padding":"16px 18px","marginBottom":"10px"})


def _fig_projections(proj: dict):
    if not proj:
        return go.Figure()
    mc    = proj.get("monte_carlo", {})
    bands = mc.get("bands", {})
    if not bands:
        return go.Figure()
    t_axis = bands.get("time_axis", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("p95",[]), mode="lines",
        line=dict(width=0), name="95th pct", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("p5",[]),  mode="lines",
        line=dict(width=0), name="5th pct",
        fill="tonexty", fillcolor="rgba(56,189,248,0.08)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("p75",[]), mode="lines",
        line=dict(width=0), name="75th pct", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("p25",[]), mode="lines",
        line=dict(width=0), name="25th pct",
        fill="tonexty", fillcolor="rgba(56,189,248,0.14)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("mean",[]), mode="lines",
        line=dict(color=C["cyan"], width=2), name="Mean"))
    fig.add_trace(go.Scatter(
        x=t_axis, y=bands.get("median",[]), mode="lines",
        line=dict(color=C["gold"], width=1.5, dash="dash"), name="Median"))
    fig.add_hline(y=1.0, line_color=C["muted"], line_dash="dot", line_width=1,
                  annotation_text="Break-even", annotation_font=dict(size=9, color=C["muted"]))
    for cp, col_a in [(21,C["teal"]),(63,C["gold"]),(126,C["violet"]),(252,C["red"])]:
        if cp <= mc.get("horizon", 0):
            fig.add_vline(x=cp, line_color=col_a, line_dash="dash", line_width=1,
                          annotation_text=f"{cp}d", annotation_font=dict(size=9, color=col_a))
    _fig_layout(fig, f"Monte Carlo Projection ({mc.get('n_paths',0):,} paths)", height=360)
    fig.update_yaxes(title="Portfolio Value (×initial)")
    fig.update_xaxes(title="Trading Days Forward")
    return fig


def _fig_scenarios(proj: dict):
    if not proj:
        return go.Figure()
    scenarios = proj.get("scenarios", {})
    if not scenarios:
        return go.Figure()
    names  = list(scenarios.keys())
    evs    = [(1 - scenarios[n]["expected_value"]) * -100 for n in names]
    probs  = [scenarios[n]["prob_loss_10pct"] * 100 for n in names]
    col_map = {"cyan":C["cyan"],"gold":C["gold"],"green":C["green"],
               "red":C["red"],"violet":C["violet"],"orange":"#fb923c"}
    colors = [col_map.get(scenarios[n].get("color","muted"), C["muted"]) for n in names]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Expected Return % (3M)", "P(loss > 10%) %"])
    fig.add_trace(go.Bar(x=names, y=evs, marker_color=colors,
                         marker_line_width=0, showlegend=False,
                         text=[f"{v:.1f}%" for v in evs], textposition="outside",
                         textfont=dict(size=9)), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=probs, marker_color=colors,
                         marker_line_width=0, showlegend=False,
                         text=[f"{v:.0f}%" for v in probs], textposition="outside",
                         textfont=dict(size=9)), row=1, col=2)
    _fig_layout(fig, "Scenario Stress Test Results (3-Month Horizon)", height=320)
    for i in range(1, 3):
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=8), row=1, col=i)
    return fig


def _checkpoint_table(proj: dict):
    if not proj:
        return html.Div()
    mc_cp = proj.get("monte_carlo",  {}).get("checkpoints", {})
    pm_cp = proj.get("parametric",   {})
    if not mc_cp:
        return html.Div("Run projection_engine.py to populate.",
            style={"color":C["muted"],"fontFamily":FONT,"fontSize":"11px","padding":"16px"})
    rows = []
    for key in ["21d","63d","126d","252d"]:
        mc = mc_cp.get(key, {}); pm = pm_cp.get(key, {})
        if not mc: continue
        rows.append(html.Tr([
            html.Td(mc.get("horizon_label",key), style={"color":C["cyan"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{(mc.get('mean',1)-1)*100:.2f}%",    style={"color":C["green"],"fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{(1-mc.get('var_95',1))*100:.2f}%",  style={"color":C["red"],  "fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{mc.get('prob_gain',0)*100:.0f}%",    style={"color":C["teal"], "fontFamily":FONT,"fontSize":"10px","paddingRight":"16px","paddingBottom":"8px"}),
            html.Td(f"{mc.get('prob_loss_10pct',0)*100:.0f}%",style={"color":C["gold"],"fontFamily":FONT,"fontSize":"10px","paddingBottom":"8px"}),
        ]))
    return html.Div([
        html.Div("PROJECTION CHECKPOINTS (MONTE CARLO)", style={"fontFamily":FONT,
            "fontSize":"9px","letterSpacing":"2px","color":C["muted"],"marginBottom":"12px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"color":C["muted"],"fontFamily":FONT,"fontSize":"10px",
                    "paddingBottom":"8px","paddingRight":"16px","textAlign":"left"})
                for h in ["Horizon","Exp. Return","VaR 95%","P(gain)","P(loss>10%)"]
            ])),
            html.Tbody(rows),
        ]),
    ], style={"padding":"16px"})


def _build_advisory_tab(advice: dict, proj: dict):
    has_adv  = bool(advice and advice.get("advice"))
    has_proj = bool(proj  and proj.get("monte_carlo"))

    no_adv = html.Div([
        html.Pre(
            "# Generate advisory output:\n"
            "python3 advisory_engine.py\n\n"
            "# Generate projections first:\n"
            "python3 projection_engine.py\n\n"
            "# Both require live_state.json:\n"
            "python3 live_optimizer.py --once",
            style={"background":C["panel"],"border":f"1px solid {C['border']}",
                   "borderRadius":"6px","padding":"16px","fontSize":"11px",
                   "fontFamily":FONT,"color":C["cyan"],"margin":"0"}),
    ], style={"marginTop":"16px"})

    # Show health gauge if advice.json exists at all (even with 0 signals)
    has_advice_json = bool(advice)
    summary_block = html.Div()
    if has_advice_json:
        summary_block = _health_gauge(advice.get("summary", {}))

    advice_cards = html.Div()
    if has_adv:
        advice_cards = html.Div([_advice_card(a) for a in advice.get("advice", [])])
    elif has_advice_json:
        # advice.json exists but produced 0 signals — show positive status
        advice_cards = html.Div([
            html.Div([
                html.Span("✓ ", style={"color": C["green"], "fontFamily": FONT, "fontSize": "12px"}),
                html.Span("No action items — portfolio is within all risk thresholds.",
                          style={"fontFamily": BODY_FONT, "fontSize": "12px", "color": C["text"]}),
            ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                      "borderLeft": f"3px solid {C['green']}", "borderRadius": "6px",
                      "padding": "14px 16px", "marginBottom": "8px"}),
            html.Div(f"Advisory engine last ran: {advice.get('generated_at','unknown')[:19]}",
                     style={"fontFamily": FONT, "fontSize": "10px", "color": C["muted"],
                            "marginTop": "8px"}),
        ])

    proj_block = html.Div()
    if has_proj:
        proj_block = html.Div([
            html.Div([
                html.Div("Monte Carlo Portfolio Projections", style={
                    "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                    "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                import_dcc_graph(_fig_projections(proj)),
            ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
                      "borderRadius":"10px","marginBottom":"16px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                html.Div([
                    html.Div("Scenario Stress Tests", style={
                        "fontFamily":FONT,"fontSize":"9px","letterSpacing":"2px",
                        "textTransform":"uppercase","color":C["muted"],"padding":"12px 16px 0"}),
                    import_dcc_graph(_fig_scenarios(proj)),
                ], style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
                html.Div([_checkpoint_table(proj)],
                    style={"background":C["panel"],"border":f"1px solid {C['border']}","borderRadius":"10px"}),
            ]),
        ])
    else:
        proj_block = html.Div([
            html.Div("No projection data.", style={"color":C["muted"],"fontFamily":FONT,
                     "fontSize":"11px","padding":"16px"}),
            html.Pre("python3 projection_engine.py  # --paths 10000 --n-jobs -1 for HPC",
                style={"background":C["panel"],"border":f"1px solid {C['border']}",
                       "borderRadius":"6px","padding":"12px","fontSize":"11px",
                       "fontFamily":FONT,"color":C["cyan"]}),
        ], style={"background":C["panel"],"border":f"1px solid {C['border']}",
                  "borderRadius":"10px","marginBottom":"16px"})

    return html.Div(style={"marginTop":"20px"}, children=[
        summary_block,
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            html.Div([
                html.Div("ADVICE & SIGNALS", style={"fontFamily":FONT,"fontSize":"9px",
                         "letterSpacing":"2px","color":C["muted"],"marginBottom":"16px"}),
                advice_cards if (has_adv or has_advice_json) else no_adv,
            ]),
            html.Div([proj_block]),
        ]),
    ])

# ── App layout ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Portfolio Risk Dashboard",
                external_stylesheets=[
                    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap"
                ])

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; background: #07090f; color: #e8f0fe; }
      ::-webkit-scrollbar { width: 5px; background: #07090f; }
      ::-webkit-scrollbar-thumb { background: #1c2640; border-radius: 3px; }
      .tab-selected { border-bottom: 2px solid #38bdf8 !important; color: #38bdf8 !important; }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''


def build_layout(data):
    fhe          = data.get("fhe", {})
    enc_classical = data.get("enc_classical", {})
    live    = data.get("live", {})
    proj    = data.get("projections", {})
    advice  = data.get("advice", {})
    m       = data["metrics"].get("overall", {})
    cov     = data["metrics"].get("var_coverage", {})
    crisis  = data["metrics"].get("crisis_performance", {})
    opt     = data["optimizer"]
    insights = generate_insights(data)

    sharpe  = m.get("sharpe")
    mdd     = m.get("max_drawdown")
    hvar    = m.get("hist_var_95")
    hes     = m.get("hist_es_95")
    turn    = m.get("avg_turnover")
    n_win   = data["metrics"].get("n_backtest_windows", "—")
    max_sr  = opt.get("max_sharpe")

    def _fmt(v, mult=1, prec=2, suffix=""):
        return f"{v*mult:.{prec}f}{suffix}" if v is not None else "—"

    tab_style = {
        "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "1.5px",
        "textTransform": "uppercase", "background": "transparent",
        "color": C["muted"], "border": "none", "borderBottom": f"2px solid {C['border']}",
        "padding": "10px 18px", "cursor": "pointer",
    }
    tab_sel_style = {**tab_style, "color": C["cyan"], "borderBottom": f"2px solid {C['cyan']}"}

    return html.Div(style={
        "background": C["bg"], "minHeight": "100vh",
        "fontFamily": BODY_FONT, "padding": "0",
    }, children=[

        # ── Live refresh interval + data store ───────────────────────────────
        dcc.Interval(
            id="live-interval",
            interval=LIVE_REFRESH_MS,
            n_intervals=0,
            disabled=False,
        ),
        dcc.Store(id="live-store"),
        dcc.Store(id="advice-store"),

        # ── Top bar ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("◈ ", style={"color": C["cyan"]}),
                html.Span("PORTFOLIO RISK SYSTEM", style={
                    "fontFamily": FONT, "fontSize": "13px",
                    "fontWeight": "700", "letterSpacing": "2px", "color": C["white"],
                }),
                html.Span("  Classical Baseline + FHE Comparison", style={
                    "fontFamily": FONT, "fontSize": "10px",
                    "color": C["muted"], "marginLeft": "12px",
                }),
            ]),
            html.Div([
                html.Span("Phase 1–6 Complete" if (fhe and fhe.get("classical_plaintext")) else "Phase 1–5 Complete", style={
                    "fontFamily": FONT, "fontSize": "9px",
                    "color": C["green"], "background": "rgba(74,222,128,0.1)",
                    "padding": "3px 10px", "borderRadius": "20px",
                    "border": f"1px solid rgba(74,222,128,0.3)",
                }),
                html.Span(
                    "  FHE: ✓ Ready" if (fhe and not fhe.get("classical_encrypted", {}).get("error")) else "  FHE: pending",
                    style={"fontFamily": FONT, "fontSize": "9px", "marginLeft": "12px",
                           "color": C["teal"] if (fhe and not fhe.get("classical_encrypted", {}).get("error")) else C["muted"]}
                ),
                html.Span("  Quantum layer: pending", style={
                    "fontFamily": FONT, "fontSize": "9px",
                    "color": C["muted"], "marginLeft": "12px",
                }),
            ]),
        ], style={
            "background": C["panel"], "borderBottom": f"1px solid {C['border']}",
            "padding": "14px 32px", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center",
        }),

        # ── Main content ─────────────────────────────────────────────────────
        html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px 28px"},
        children=[

            # ── KPI row ──────────────────────────────────────────────────────
            html.Div([
                kpi_card("OOS Sharpe",        _fmt(sharpe, prec=3),       "",  
                         C["green"] if (sharpe or 0) > 0.7 else C["gold"] if (sharpe or 0) > 0.4 else C["red"],
                         "Target >0.7 for quantum layer"),
                kpi_card("Max Drawdown",       _fmt(mdd, mult=100, prec=1), "%",
                         C["red"],   "Peak-to-trough over full backtest"),
                kpi_card("Hist VaR 95%",       _fmt(hvar, mult=100, prec=2), "%/day",
                         C["gold"],  "Daily loss threshold exceeded 5% of days"),
                kpi_card("Hist ES 95%",        _fmt(hes,  mult=100, prec=2), "%/day",
                         C["violet"],"Avg loss when VaR is breached"),
                kpi_card("Avg Turnover",       _fmt(turn, prec=3),          "",
                         C["teal"],  "Per-rebalance weight change Σ|Δw|"),
                kpi_card("Frontier Sharpe",    _fmt(max_sr, prec=3),         "",
                         C["cyan"],  "Best Sharpe on efficient frontier"),
                kpi_card("Backtest Windows",   str(n_win),                   "",
                         C["white"], "3yr train / 1yr test / quarterly step"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                      "marginBottom": "24px"}),

            # ── Tabs ─────────────────────────────────────────────────────────
            dcc.Tabs(id="tabs", value="overview", style={
                "background": "transparent", "border": "none",
            }, children=[

                # ── Tab 1: Overview ──────────────────────────────────────────
                dcc.Tab(label="OVERVIEW", value="overview",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[
                    html.Div(style={"display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "16px", "marginTop": "20px"}, children=[

                        html.Div([
                            html.Div("Cumulative Return", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_cumulative_returns(data["bt_returns"], data["returns"]),
                                      config={"displayModeBar": False},
                                      style={"borderRadius": "8px"}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        html.Div([
                            html.Div("Rolling Sharpe", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_rolling_sharpe(data["backtest_df"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        html.Div([
                            html.Div("Rolling VaR & ES", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_rolling_var(data["rolling_risk"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        html.Div([
                            html.Div("Rolling Volatility", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_rolling_vol(data["rolling_risk"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),
                    ]),
                ]),

                # ── Tab 2: Risk Analysis ─────────────────────────────────────
                dcc.Tab(label="RISK ANALYSIS", value="risk",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[
                    html.Div(style={"marginTop": "20px", "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[

                        html.Div([
                            html.Div("Crisis-Period Performance", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_crisis_bar(crisis),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px", "gridColumn": "1 / -1"}),

                        html.Div([
                            html.Div("Asset Correlation Matrix", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_correlation_heatmap(data["corr"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        html.Div([
                            html.Div("Risk Decomposition (Equal Weight)", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_risk_decomposition(data["rd"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        # VaR coverage stats table
                        html.Div([
                            html.Div("VaR Coverage Summary", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 8px",
                            }),
                            html.Div([
                                html.Table([
                                    html.Tbody([
                                        html.Tr([
                                            html.Td(k, style={"color": C["muted"], "paddingRight": "24px",
                                                              "fontFamily": FONT, "fontSize": "10px",
                                                              "paddingBottom": "10px"}),
                                            html.Td(str(v), style={"color": C["white"],
                                                                    "fontFamily": FONT, "fontSize": "11px",
                                                                    "fontWeight": "700",
                                                                    "paddingBottom": "10px"}),
                                        ])
                                        for k, v in cov.items()
                                    ])
                                ])
                            ], style={"padding": "0 16px 16px"})
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),
                    ]),
                ]),

                # ── Tab 3: Portfolio Construction ────────────────────────────
                dcc.Tab(label="PORTFOLIO", value="portfolio",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[
                    html.Div(style={"marginTop": "20px", "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        html.Div([
                            html.Div("Efficient Frontier", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_efficient_frontier(data["frontier"], data["w_class"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),

                        html.Div([
                            html.Div("Rolling Weight Allocation", style={
                                "fontFamily": FONT, "fontSize": "9px",
                                "letterSpacing": "2px", "textTransform": "uppercase",
                                "color": C["muted"], "padding": "12px 16px 0",
                            }),
                            dcc.Graph(figure=fig_weight_evolution(data["rolling_weights"]),
                                      config={"displayModeBar": False}),
                        ], style={"background": C["panel"], "border": f"1px solid {C['border']}",
                                  "borderRadius": "10px"}),
                    ]),
                ]),



                # ── Tab 4: LIVE ──────────────────────────────────────────────
                dcc.Tab(label="LIVE", value="live",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[html.Div(id="live-tab-content",
                                           children=[_build_live_tab(live)])]),

                # ── Tab 5: ADVISORY ──────────────────────────────────────────
                dcc.Tab(label="ADVISORY", value="advisory",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[html.Div(id="advisory-tab-content",
                                           children=[_build_advisory_tab(advice, proj)])]),

                # ── Tab 6: FHE COMPARISON ────────────────────────────────────
                dcc.Tab(label="FHE COMPARISON", value="fhe",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[_build_fhe_tab(fhe, enc_classical)]),

                # ── Tab 7: Actionable Insights ───────────────────────────────
                dcc.Tab(label="INSIGHTS", value="insights",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[
                    html.Div(style={"marginTop": "24px",
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "16px",
                                    "alignItems": "start"},
                    children=[
                        html.Div([
                            html.Div("MODEL DIAGNOSTICS & ACTIONABLE INSIGHTS", style={
                                "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                                "color": C["muted"], "marginBottom": "16px",
                            }),
                            *[insight_card(t, b, c, k) for t, b, c, k in insights],
                        ], style={"gridColumn": "1 / -1" if len(insights) <= 4 else "1"}),

                        # Quantum readiness checklist
                        html.Div([
                            html.Div("QUANTUM LAYER READINESS", style={
                                "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                                "color": C["muted"], "marginBottom": "16px",
                            }),
                            *[html.Div([
                                html.Span(icon + " ", style={"color": col}),
                                html.Span(label, style={"fontFamily": BODY_FONT,
                                                         "fontSize": "12px", "color": C["text"]}),
                                html.Div(desc, style={"fontFamily": BODY_FONT, "fontSize": "11px",
                                                       "color": C["muted"], "marginTop": "3px",
                                                       "paddingLeft": "20px", "lineHeight": "1.5"}),
                            ], style={"marginBottom": "14px",
                                      "background": C["panel"],
                                      "border": f"1px solid {C['border']}",
                                      "borderRadius": "6px", "padding": "12px 14px"})
                            for icon, col, label, desc in [
                                ("✓", C["green"],  "classical_metrics.json generated",
                                 "Baseline Sharpe, VaR, ES, drawdown locked in for comparison."),
                                ("✓", C["green"],  "Sqrt Chebyshev approximation validated",
                                 "FHE-compatible √(w'Σw) polynomial fitted and coeffs saved."),
                                ("✓", C["green"],  "Scaler artifact ready",
                                 "scaler_params.json saved for FHE encoding in alice_portfolio.py."),
                                ("✓", C["green"],  "Historical VaR/ES computed",
                                 "Nonlinear targets available for VQC regression training."),
                                ("✓", C["green"],  "Rolling covariance Σ(t) saved",
                                 "252-day rolling Σ available for quantum layer warm-start."),
                                ("◉", C["gold"],   "Next: vqc_portfolio.py",
                                 "Build 6–8 qubit PennyLane VQC — regression on hist ES/vol."),
                                ("◉", C["gold"],   "Next: surrogate_portfolio.py",
                                 "Fit degree-2 polynomial to VQC outputs. Target R² ≥ 0.95."),
                                ("✓" if (fhe and fhe.get("classical_plaintext")) else "◉",
                                 C["teal"] if (fhe and fhe.get("classical_plaintext")) else C["gold"],
                                 "FHE classical comparison complete",
                                 "fhe_comparison.json generated — check FHE COMPARISON tab." if (fhe and fhe.get("classical_plaintext")) else "Run: python3 run_fhe_comparison.py"),
                                ("◉", C["gold"],   "Next: vqc_portfolio.py",
                                 "Build 6–8 qubit PennyLane VQC — regression on hist ES/vol."),
                                ("◉", C["gold"],   "Next: surrogate_portfolio.py",
                                 "Fit degree-2 polynomial to VQC outputs. Target R² ≥ 0.95."),
                                ("◎", C["muted"],  "Pending: alice_portfolio.py",
                                 "Full 4-mode pipeline: classical + quantum × plaintext + CKKS."),
                                ("◎", C["muted"],  "Pending: carol_portfolio_listener.py",
                                 "Encrypted polynomial evaluation — linear + quadratic CKKS ops."),
                            ]],
                        ], style={"gridColumn": "2" if len(insights) > 4 else "1 / -1"}),
                    ]),
                ]),

                # ── Tab 8: Raw Metrics ───────────────────────────────────────
                dcc.Tab(label="RAW DATA", value="raw",
                        style=tab_style, selected_style=tab_sel_style,
                        children=[
                    html.Div(style={"marginTop": "20px"}, children=[
                        html.Div("FULL METRICS JSON", style={
                            "fontFamily": FONT, "fontSize": "9px", "letterSpacing": "2px",
                            "color": C["muted"], "marginBottom": "10px",
                        }),
                        html.Pre(
                            json.dumps(data["metrics"], indent=2, default=str)
                            + "\n\n--- FHE COMPARISON ---\n"
                            + (json.dumps(data["fhe"], indent=2, default=str) if data["fhe"] else "Run: python3 run_fhe_comparison.py"),
                            style={
                                "background":  C["panel"],
                                "border":      f"1px solid {C['border']}",
                                "borderRadius": "8px",
                                "padding":     "20px",
                                "fontSize":    "11px",
                                "fontFamily":  FONT,
                                "color":       C["text"],
                                "overflowX":   "auto",
                                "maxHeight":   "600px",
                                "overflowY":   "auto",
                            }
                        ),
                    ])
                ]),
            ]),
        ]),
    ])




# ── Live refresh callback ───────────────────────────────────────────────────────
# Fires every LIVE_REFRESH_MS milliseconds.
# Re-reads live_state.json, projections.json, advice.json from disk and
# rebuilds the Live and Advisory tab contents in-place.
# No page reload required — only the two dynamic tabs are updated.

@app.callback(
    Output("live-tab-content",      "children"),
    Output("advisory-tab-content",  "children"),
    Output("live-badge-container",  "children"),
    Input("live-interval",          "n_intervals"),
    prevent_initial_call=False,
)
def refresh_live_tabs(n_intervals):
    """
    Called automatically every LIVE_REFRESH_MS ms by dcc.Interval.
    Reads the three live artifact files fresh from disk each time.
    Returns rebuilt tab content — Dash diffs and applies only the changes.
    """
    live          = _load_json("live_state.json")
    proj          = _load_json("projections.json")
    advice        = _load_json("advice.json")
    enc_classical = _load_json("encrypted_classical_results.json")

    live_content    = _build_live_tab(live)
    advisory_content = _build_advisory_tab(advice, proj)
    badge           = _live_status_badge(live)

    return live_content, advisory_content, badge

# ── Boot ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Portfolio Risk Dashboard")
    _p.add_argument("--refresh", type=int, default=LIVE_REFRESH_MS,
                    help=f"Live tab refresh interval ms (default: {LIVE_REFRESH_MS})")
    _p.add_argument("--port",    type=int, default=8050)
    _p.add_argument("--debug",   action="store_true")
    _args = _p.parse_args()

    # Apply refresh interval from CLI
    LIVE_REFRESH_MS = _args.refresh

    print("\n[DASH] Loading artefacts …")
    data = load_all()
    live_status = "loaded" if data.get("live") else "not found — run: python3 live_optimizer.py --once"
    fhe_status  = "loaded" if data.get("fhe")  else "not found — run: python3 run_fhe_comparison.py"
    print(f"[DASH] Live state      : {live_status}")
    print(f"[DASH] FHE comparison  : {fhe_status}")
    enc_status  = "loaded" if data.get("enc_classical") else "not found — run: python3 encrypted_classical.py"
    print(f"[DASH] Encrypted class.: {enc_status}")
    print(f"[DASH] Live refresh    : every {LIVE_REFRESH_MS/1000:.0f}s (LIVE + ADVISORY tabs)")
    print("[DASH] Building layout …")
    app.layout = build_layout(data)
    print(f"[DASH] Starting server at http://localhost:{_args.port}\n")
    app.run(debug=_args.debug, host="0.0.0.0", port=_args.port)