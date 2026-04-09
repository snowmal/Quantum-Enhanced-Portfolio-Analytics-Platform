# Quantum-Enhanced Portfolio Risk System

> **A privacy-preserving, quantum-augmented portfolio analytics platform.**  
> Classical Markowitz baseline → Variational Quantum Circuit risk modelling → Fully Homomorphic Encryption → Live dashboard.

---

## Overview

This project extends the architecture of a prior **Quantum-Enhanced Fraud Detection System** — which demonstrated that a Variational Quantum Classifier (VQC) approximated by a degree-2 polynomial surrogate can be evaluated on CKKS-encrypted inputs without ever decrypting the data — into the domain of **portfolio risk management**.

The core thesis is identical: quantum circuits naturally encode nonlinear, entangled feature relationships that classical linear models miss. In fraud detection, those relationships were between transaction features (amount, time, merchant category). Here they are between **portfolio weights, factor exposures, and macroeconomic regime variables** — specifically how joint combinations of these drive tail risk and covariance instability that Gaussian factor models cannot fully capture.

By approximating the trained VQC with a degree-2 polynomial surrogate, the entire risk inference pipeline becomes compatible with CKKS Fully Homomorphic Encryption (FHE). Portfolio weights are encrypted by Alice (the data owner), risk scores are computed by Carol (the encrypted evaluator) who never sees the plaintext, and Alice decrypts only the final output. The portfolio never leaves an encrypted state during computation.

The system is designed as a **complete, production-quality research platform** — not a demo. It includes a live data feed, real-time re-optimization, Monte Carlo forward projections, rule-based advisory signals, and an interactive Dash dashboard. Everything runs from a single command even from a completely fresh environment.

---

## Prior Work: Fraud Detection System

This project is a direct extension of a completed VQC → FHE fraud detection pipeline. That system:

-   Trained a **4-qubit PennyLane VQC** (2 variational layers, CNOT entanglement chain) on financial transaction data to classify fraud vs. legitimate transactions
-   Approximated the VQC output with a **degree-2 polynomial surrogate** (R² ≈ 0.98) using least-squares fitting on VQC evaluation grid
-   Evaluated the surrogate inside **TenSEAL CKKS ciphertexts** (poly\_modulus\_degree=16384, scale=2²⁰) — Carol computed `Enc(f̂(z))` using only additions and multiplications, never decrypting
-   Achieved **~7% higher accuracy** than a classical MLP with identical runtime (~1 min, dominated by FHE operations)
-   Produced per-feature attribution from the polynomial gradient (`C_i = a_i·z_i + z_i·(Qz)_i`)

The portfolio system reuses `alice.py`, `carol_listener.py`, `surrogate_fit.py`, and the CKKS context setup with parameter adjustments. The primary new work is the financial data pipeline, classical baseline models, and extending the VQC from binary classification to continuous regression.

---

## Architecture

text

Copy

```
┌─────────────────────────────────────────────────────────────────────┐│                    THREE-PARTY FHE MODEL                            ││                                                                     ││  ALICE (data owner)          CAROL (encrypted evaluator)            ││  ├─ holds secret key         ├─ holds polynomial JSON only          ││  ├─ standardizes w → z       ├─ never sees plaintext                ││  ├─ encrypts: Enc(z)         ├─ evaluates: Enc(P_risk(w))           ││  ├─ sends to Carol     ────► │   using add + mul on ciphertexts     ││  ├─ receives Enc(score) ◄─── ├─ returns encrypted scores            ││  └─ decrypts → metrics       └─ (zero knowledge of portfolio)       ││                                                                     ││  PUBLIC CONTEXT: poly_modulus_degree=16384, scale=2⁴⁰, depth=4     │└─────────────────────────────────────────────────────────────────────┘Pipeline modes:  classical_plaintext  →  Markowitz w⊤Σw, no encryption  classical_encrypted  →  w⊤Σw polynomial in CKKS ciphertext       ✓ BUILT  quantum_plaintext    →  VQC surrogate, no encryption              ◉ PENDING  quantum_encrypted    →  VQC surrogate in CKKS ciphertext          ◉ PENDING
```

---

## What Is Built (Current State)

### Classical Pipeline (Steps 1–5) ✓

-   **Data pipeline** (`data_pipeline.py`): fetches 15 S&P 500 tickers across 5 sectors (Tech, Finance, Healthcare, Energy, Consumer) from yfinance 2005–2023. Augmented with Fama-French 3-factor data (Mkt-RF, SMB, HML) from the Ken French data library, and FRED macroeconomic variables (VIX, yield curve spread, Fed Funds Rate). Computes daily log returns, rolling 252-day covariance Σ(t), z-score standardisation with ±3σ clipping.
-   **Factor model** (`classical_model.py`): OLS regression r = Bf + ε per asset on a rolling 1-year window. Computes B matrix (N×K factor loadings), portfolio variance σ²\_p = w⊤Σw, Marginal Risk Contribution (MRC = ∂σ\_p/∂w\_i), and Component Risk Contribution (CRC = w\_i · MRC\_i / σ\_p).
-   **Risk metrics** (`risk_metrics.py`): parametric VaR and ES under Gaussian assumption (FHE-compatible), historical VaR and ES from empirical return quantiles (nonlinear targets for VQC regression), and Chebyshev polynomial approximation of √(w⊤Σw) at degree 3–5 for FHE-compatible volatility.
-   **Markowitz optimizer** (`classical_optimizer.py`): SLSQP solver for min\_w w⊤Σw − λw⊤μ subject to Σw\_i = 1, w\_i ≥ 0. Scans λ ∈ \[0,1\] to produce the efficient frontier. Optional Black-Litterman fallback using the Woodbury matrix identity when Σ is near-singular, inverting only the K×K view matrix rather than the N×N covariance.
-   **Backtest** (`evaluate_classical.py`): rolling window backtest with 3-year training / 1-year test / quarterly step. Records out-of-sample Sharpe, max drawdown, VaR coverage rate, realised ES, portfolio turnover, and weight stability. Crisis-period isolation for 2008–09 (GFC), March–May 2020 (COVID), and 2022 (rate cycle).

### FHE Bridge (Steps 6–8) ✓

-   **Polynomial bridge** (`build_classical_polynomial.py`): derives the degree-2 polynomial representation of w⊤Σw in standardised weight space: `f̂(z) = a₀ + Σaᵢzᵢ + Σ_{i,j} Q_{ij}zᵢzⱼ` where z = scaler.transform(w). Fits a StandardScaler on 203 weight vectors (3 frontier portfolios + 200 noisy augmented copies). Exports `classical_polynomial_model.json` and `scaler.pkl`. Validates the Chebyshev sqrt approximation against exact √(w⊤Σw).
-   **FHE accuracy check** (`run_fhe_comparison.py`): lightweight plaintext-vs-CKKS comparison for quick validation. Computes variance, σ\_p, μ\_p, Sharpe, VaR, ES in both modes and measures relative error. Target: variance relative error < 0.1%.
-   **Full encrypted classical pipeline** (`encrypted_classical.py`): production Alice → Carol → Alice pipeline. Alice standardises and encrypts weights via CKKS. Carol evaluates the degree-2 polynomial on the ciphertext using only add and mul operations, consuming 1 multiplicative level. Alice decrypts and computes all downstream metrics: σ²\_p, σ\_p, μ\_p, Sharpe, parametric VaR, Gaussian ES, historical ES, per-asset MRC/CRC/CRC%, annualised return/vol/Sharpe, and full accuracy analysis. Runs all three portfolios (tangency, minimum variance, equal weight). Outputs `encrypted_classical_results.json`.

### Live Engines ✓

-   **Live optimizer** (`live_optimizer.py`): polls yfinance every N seconds (default: 300), recomputes rolling covariance from the latest window including today’s partial session, re-runs Markowitz SLSQP, computes intraday returns and 5/20-day momentum signals per asset, and writes `live_state.json` atomically. HPC mode: parallel lambda scan via joblib with n\_jobs=-1.
-   **Projection engine** (`projection_engine.py`): Monte Carlo GBM simulation (N paths × H trading days forward) parallelised via joblib. Reports portfolio value distributions at 1M/3M/6M/12M checkpoints with full percentile bands (p5, p25, p75, p95, mean, median). Seven named scenario stress tests: Base Case, GFC-Style Crash (-40% return, 3.5× vol), COVID Shock (-25%, 2.8×), Rate Spike +300bps (-15%, 1.8×), Inflation Persistence (-10%, 1.4×), Tech Selloff (-20%, 2.0×), Soft Landing (+5%, 0.8×). HPC: 10,000 paths in ~8s on one core, ~0.3s on 32 cores.
-   **Advisory engine** (`advisory_engine.py`): 13 rule-based signal functions across four categories — Risk (VaR level, volatility regime, max drawdown, Sharpe quality), Allocation (single-asset concentration, sector concentration, average pairwise correlation), Rebalancing (weight drift from optimal, momentum signals, favourable rebalancing conditions), and Projections (3-month loss probability, worst-case scenario, soft landing upside). Each rule returns a structured Advice object with severity (ACTION / WATCH / INFO), metric, and concrete action step. Portfolio health score 0–100.

### Dashboard ✓

-   **8-tab Dash application** (`dashboard_server.py`) with 15-second auto-refresh on the Live and Advisory tabs:
    -   **Overview** — cumulative return vs. equal weight, rolling Sharpe bar chart with 0.7 quantum target line, rolling VaR/ES (95%, both parametric and historical), rolling 252-day annualised volatility. Crisis shading for GFC, COVID, Bear 2022.
    -   **Risk Analysis** — crisis-period Sharpe/return/drawdown, asset correlation heatmap, component risk contribution bar chart, VaR coverage summary table.
    -   **Portfolio** — efficient frontier with tangency and minimum variance portfolio markers, rolling weight stacked area chart.
    -   **Live** — six KPI cards (live Sharpe, annualised return/vol, VaR, intraday P&L, optimizer runtime), live weight and CRC side-by-side bar charts, intraday returns by asset, 5-day vs. 20-day momentum signals, per-asset attribution table with momentum direction.
    -   **Advisory** — portfolio health gauge (0–100 with label), all triggered advice cards colour-coded by severity, Monte Carlo fan chart with percentile bands and checkpoint markers, scenario comparison bar charts, checkpoint table (expected return, VaR, P(gain), P(loss>10%) at 1M/3M/6M/12M).
    -   **FHE Comparison** — two sections: (1) full encrypted classical results — six KPI cards, all-metrics grouped bar chart, timing breakdown, per-metric accuracy table with pass/fail, relative error % chart, CRC attribution comparison; (2) legacy quick accuracy check from `run_fhe_comparison.py`.
    -   **Insights** — model diagnostics with insight cards, pipeline readiness checklist showing which steps are complete and what is pending.
    -   **Raw Data** — full JSON dumps of `classical_metrics.json` and `fhe_comparison.json`.

---

## Quantum Layer (Pending — Phase 2)

The quantum layer slots in as a drop-in replacement for the risk/return objective function. The infrastructure is already wired: `run_platform.py` detects `vqc_portfolio.py` and `surrogate_portfolio.py` and runs them automatically when they exist. The FHE comparison tab has placeholder slots for quantum\_plaintext and quantum\_encrypted modes.

### VQC Design (`vqc_portfolio.py` — to be built)

-   **Qubits**: 6–8 (supports 6–8 input features after PCA/factor compression to control circuit depth)
-   **Input encoding**: RY angle encoding — `θ_i = α · clip(z_i, −k, k)`, same as fraud detection VQC (`k=3`, `α=π/2`)
-   **Variational layers**: 2–3 layers of trainable RY/RZ rotations + CNOT entanglement chain
-   **Output**: expectation value `⟨Z₀⟩ ∈ [−1, 1]` reinterpreted as a risk or return score
-   **Training objective**: MSE loss against historical VaR/ES or realised portfolio volatility — regression, not classification
-   **Two separate VQCs**: `P_risk(w, f)` trained on realised volatility/ES, `P_return(w, f)` trained on realised returns
-   **SDK**: PennyLane with `default.qubit` simulator; `lightning.gpu` backend for GPU-accelerated training on HPC

### Polynomial Surrogate (`surrogate_portfolio.py` — to be built)

-   Same degree-2 form: `f̂(w) = a₀ + Σaᵢwᵢ + Σ_{i,j} Q_{ij}wᵢwⱼ`
-   Note that `σ²_p = w⊤Σw` is already degree-2 — the VQC adds the nonlinear residual beyond linear covariance
-   Design matrix approach: evaluate trained VQC on a dense grid of (w, f) combinations, solve least-squares for polynomial coefficients
-   Target: R² ≥ 0.95 (same standard as R² ≈ 0.98 achieved in fraud detection)
-   Export `P_risk.json` and `P_return.json` in the same format as `classical_polynomial_model.json`

### Encrypted Quantum Optimization

-   Encrypted objective: Alice minimises `Enc(P_risk(w)) − λ · Enc(P_return(w))` via gradient descent in encrypted space
-   Alice decrypts, computes `∇P_risk(w) − λ∇P_return(w)`, updates w, re-encrypts, repeats until convergence
-   Chebyshev sqrt approximation (degree 3–5) for `√(P_risk(w))` needed for VaR computation inside CKKS
-   CKKS parameters may need extension to `coeff_mod_bit_sizes=[60,40,40,40,40,60]` for the additional multiplicative depth required by the Chebyshev approximation

### Quantum Evaluation Targets

| Metric | Classical Baseline | Quantum Target |
| --- | --- | --- |
| Out-of-sample Sharpe | 0.4–0.7 | \> 0.7 |
| Crisis Sharpe (2020) | Typically negative | Less negative via nonlinear hedge |
| VaR Coverage | ~95% (Gaussian) | \> 95% (nonparametric) |
| Max Drawdown | Markowitz baseline | Lower via nonlinear risk penalty |
| Surrogate R² | N/A | ≥ 0.95 |
| Runtime per optimization | < 1s (classical) | ~1–2 min with FHE (acceptable) |

---

## Planned Enhancements

### Black-Litterman Integration

Standard Markowitz is already implemented. If extreme weight instability or near-singular covariance causes optimizer failures (common with small N and real data), the system switches to Black-Litterman using the Woodbury matrix identity:

`μ_BL = π + τΣP⊤(PτΣP⊤ + Ω)⁻¹(q − Pπ)`

This inverts only the K×K view matrix (K = 1–10 views), keeping N arbitrarily large. The full encrypted version would encrypt prior returns π, investor views q, and the confidence matrix Ω, enabling secure private-view-based portfolio construction.

### Higher-Moment Modelling

The VQC’s expressiveness enables modelling of portfolio skewness and excess kurtosis — statistics that matter significantly for tail risk but are invisible to Gaussian/linear models. Planned: third and fourth moment polynomial surrogates for ES refinement and stress scenario calibration.

### Cross-Factor Nonlinear Sensitivities

Current factor model: linear `r = Bf + ε`. Planned: VQC captures nonlinear cross-factor sensitivities (e.g., how the Mkt-RF × VIX interaction drives conditional correlation during market stress) — the kind of relationship that Gaussian copulas miss and that drives underestimation of systemic risk.

### Real-Time Data Feed Upgrade

Current live feed uses yfinance (15-minute delayed). Planned integration with [Polygon.io](http://Polygon.io) or Alpaca real-time API for true intraday re-optimization. The live optimizer is already structured with a clean data source abstraction (`fetch_live_returns`) — swapping the source requires only changing one function.

### Adaptive Advisory Engine

The current advisory engine is rule-based with fixed thresholds. Planned: a second-pass LLM call that takes the structured advice output and generates natural-language client-facing summaries in the tone and detail appropriate for the portfolio’s mandate (e.g., institutional vs. retail, conservative vs. aggressive). The rule layer provides structure and auditability; the language layer provides communication quality.

### MPI-Based HPC Distribution

Monte Carlo path simulation is currently parallelised via `joblib.Parallel`. For cluster environments (SLURM/PBS), planned `mpi4py` integration allowing true distributed memory parallelism — relevant for 10M+ path simulations or rolling backtest parallelisation. `mpi4py` is already listed as an optional dependency.

### GPU-Accelerated VQC Training

PennyLane’s `lightning.gpu` backend uses NVIDIA CUDA for circuit simulation. The VQC training step (planned `vqc_portfolio.py`) is structured to accept a `--backend` argument that switches between `default.qubit` (CPU) and `lightning.gpu` (CUDA). On a GPU node, VQC training time drops from hours to minutes.

---

## Repository Structure

text

Copy

```
portfolio_system/│├── run_platform.py                  ← SINGLE ENTRY POINT — runs everything├── main_classical.py                ← sequential pipeline runner (steps 1–8)│├── portfolio_training/              ← core analytical modules│   ├── data_pipeline.py             ← yfinance + FRED + Fama-French fetch│   ├── classical_model.py           ← OLS factor model, Σ, MRC/CRC│   ├── risk_metrics.py              ← VaR, ES, Chebyshev sqrt approx│   ├── classical_optimizer.py       ← Markowitz SLSQP + Black-Litterman│   ├── evaluate_classical.py        ← rolling backtest, crisis stats│   ├── vqc_portfolio.py             ← [PENDING] PennyLane VQC regression│   ├── surrogate_portfolio.py       ← [PENDING] degree-2 poly surrogate fit│   └── artifacts/                   ← all outputs (gitignored)│├── build_classical_polynomial.py    ← FHE bridge: derives poly from Σ├── run_fhe_comparison.py            ← quick plaintext vs. CKKS check├── encrypted_classical.py           ← full Alice→Carol→Alice pipeline│├── live_optimizer.py                ← background: live weights + momentum├── projection_engine.py             ← background: Monte Carlo projections├── advisory_engine.py               ← background: rule-based advice│├── dashboard_server.py              ← 8-tab Dash app (http://localhost:8050)├── requirements.txt└── README.md
```

**Artifact outputs** (written to `portfolio_training/artifacts/`, not committed):

| File | Written by | Contains |
| --- | --- | --- |
| `returns.csv` | data\_pipeline | Daily log returns (T × N) |
| `sigma_full.npy` | classical\_model | Full-period covariance matrix |
| `mu_annual.npy` | classical\_model | Expected annual returns |
| `w_classical.csv` | classical\_optimizer | Tangency, min-variance, equal-weight portfolios |
| `classical_metrics.json` | evaluate\_classical | Backtest Sharpe, drawdown, VaR coverage |
| `classical_polynomial_model.json` | build\_classical\_polynomial | Degree-2 FHE polynomial coefficients |
| `scaler.pkl` | build\_classical\_polynomial | Weight StandardScaler for FHE encoding |
| `chebyshev_sqrt_coeffs.json` | risk\_metrics | Sqrt polynomial approximation coefficients |
| `fhe_comparison.json` | run\_fhe\_comparison | Quick plaintext vs. CKKS accuracy check |
| `encrypted_classical_results.json` | encrypted\_classical | Full encrypted pipeline metrics + accuracy |
| `live_state.json` | live\_optimizer | Current weights, intraday, momentum |
| `projections.json` | projection\_engine | MC paths, checkpoints, scenarios |
| `advice.json` | advisory\_engine | Advice cards, health score |

---

## Installation

bash

Copy

```bash
git clone https://github.com/your-username/quantum-portfolio-risk.gitcd quantum-portfolio-risk/portfolio_systempip install -r requirements.txt# Optional — encrypted modes (requires C++ build tools):pip install tenseal# Optional — quantum layer (when VQC modules are built):pip install pennylane# Optional — FRED macroeconomic data:export FRED_API_KEY="your_key_here"   # free at fred.stlouisfed.org
```

---

## Quick Start

### Fresh start — everything from zero

bash

Copy

```bash
python3 run_platform.py
```

This runs all 11 pipeline steps, generates initial engine outputs, starts background engines, and opens the dashboard at `http://localhost:8050`. Total runtime on first run: ~15–30 minutes (data fetch dominates). Subsequent runs with cached data: ~5–10 minutes.

### Data already downloaded

bash

Copy

```bash
python3 run_platform.py --skip-data
```

### Classical only (no FHE, fastest)

bash

Copy

```bash
python3 run_platform.py --skip-fhe --no-engines --no-dashboard
```

### Dashboard only (pipeline already ran)

bash

Copy

```bash
python3 run_platform.py --dashboard-only
```

### Restart engines only

bash

Copy

```bash
python3 run_platform.py --engines-only
```

### HPC mode

bash

Copy

```bash
python3 run_platform.py --lambda-scan --mc-paths 50000 --n-jobs 32
```

### Quantum mode (when VQC modules are ready)

bash

Copy

```bash
python3 run_platform.py --mode quantum-fhe
```

---

## Data Sources

| Source | Access | Contents | Used for |
| --- | --- | --- | --- |
| yfinance | Free, no auth | 15 S&P 500 tickers, 2005–2023, daily OHLCV | Asset returns, intraday prices |
| Ken French Data Library | Free, direct CSV | Mkt-RF, SMB, HML daily factors | Factor model baseline |
| FRED (fredapi) | Free, API key | VIX, yield curve spread, Fed Funds Rate | Macro regime variables |

All sources are free and reproducible. No proprietary data is required.

---

## Key Technical Decisions

**Why a polynomial surrogate instead of evaluating the VQC directly under FHE?** CKKS supports only polynomial arithmetic (additions and multiplications). A quantum circuit is not directly expressible as polynomial operations. The surrogate bridge — fit a degree-2 polynomial to the VQC’s output over a grid of inputs — is what makes FHE compatibility possible while preserving the expressive power of the quantum model. R² ≈ 0.98 in the fraud detection system; ≥ 0.95 is the target here.

**Why degree-2 and not higher?** Each polynomial multiplication consumes one CKKS multiplicative level. With `coeff_mod_bit_sizes=[60,40,40,40,60]`, the context has 4 levels available. Degree-2 consumes 1 level for the quadratic terms, leaving 3 spare for the Chebyshev sqrt approximation of VaR and any additional operations. Degree-3 would require additional coefficient primes and reduce precision.

**Why Chebyshev for the square root?** `σ_p = √(w⊤Σw)` requires a square root, which is non-polynomial and therefore incompatible with CKKS. Taylor expansion around the expected variance works but is only accurate near the expansion point. A Chebyshev polynomial fit of degree 3–5 on the observed variance range `[0, max_var]` achieves accuracy across the full domain. Both approaches are implemented and compared.

**Why Black-Litterman as a fallback?** Standard Markowitz inverts the N×N covariance matrix, which becomes numerically unstable for small N and real data. The Woodbury identity reformulation inverts only the K×K view matrix (K = 1–10 investor views), making it stable regardless of N. Both forms produce identical optimal weights when the views are uninformative.

---

## Performance Notes

**Monte Carlo scaling** (projection\_engine.py): The GBM path simulation is embarrassingly parallel — each path is independent. With `joblib.Parallel(n_jobs=-1)`:

| Paths | 1 core (laptop) | 8 cores | 32 cores (HPC) |
| --- | --- | --- | --- |
| 10,000 | ~8s | ~1.2s | ~0.3s |
| 50,000 | ~40s | ~6s | ~1.5s |
| 500,000 | ~400s | ~55s | ~14s |

**FHE overhead** (encrypted\_classical.py): The dominant cost is context build (~1–2s, one-time per session) and the Carol polynomial evaluation (~5–30s depending on N). Per-inference overhead after context is built: ~100–500ms — acceptable at portfolio-frequency computation (daily or weekly rebalancing). Not suitable for tick-by-tick trading.

**VQC training** (planned): Estimated ~2–4 hours on CPU for 6-qubit circuit with 2 variational layers on a 4,500-day dataset. With PennyLane `lightning.gpu` on an NVIDIA A100: ~8–15 minutes.

---

## Licence

MIT. See `LICENSE` for details.

---

## Acknowledgements

This project builds on:

-   **PennyLane** (Xanadu) for variational quantum circuit training
-   **TenSEAL** (OpenMined) for CKKS homomorphic encryption
-   **Ken French Data Library** for Fama-French factor data
-   **FRED** (Federal Reserve Bank of St. Louis) for macroeconomic time series
-   The **VQC → FHE Fraud Detection System** that established the core architecture this project extends
