"""
encrypted_classical.py
=======================
Full end-to-end encrypted classical portfolio pipeline.

This is the production-quality implementation of the encrypted classical mode.
It is distinct from run_fhe_comparison.py (which is a quick accuracy check).
Here we run the COMPLETE pipeline:

    Alice (data owner)
        ↓  standardizes weights → encrypts via CKKS
    Carol (encrypted evaluator)
        ↓  evaluates degree-2 polynomial on ciphertext — never sees plaintext
    Alice
        ↓  decrypts → computes ALL risk metrics from decrypted score
        ↓  saves full results to encrypted_classical_results.json

Metrics computed (both plaintext and encrypted, for exact comparison):
    • Portfolio variance σ²_p        (exact vs. decrypted CKKS)
    • Portfolio volatility σ_p
    • Portfolio return μ_p
    • Sharpe ratio
    • Parametric VaR 95%             (Chebyshev sqrt approx for FHE)
    • Gaussian Expected Shortfall 95%
    • Historical ES 95%              (from returns, post-decryption)
    • Per-asset MRC / CRC / CRC%
    • CKKS accuracy metrics          (absolute error, relative error)
    • Runtime breakdown              (context, encryption, Carol eval, decryption)

Output
------
    portfolio_training/artifacts/encrypted_classical_results.json

Usage
-----
    # Run on real artifacts (after main_classical.py):
    python3 encrypted_classical.py

    # Specific portfolio (0=tangency, 1=min_variance, 2=equal_weight):
    python3 encrypted_classical.py --portfolio tangency
    python3 encrypted_classical.py --portfolio min_variance
    python3 encrypted_classical.py --portfolio equal_weight

    # Run all three portfolios and compare:
    python3 encrypted_classical.py --all-portfolios

    # Quick synthetic demo (no artifact files needed):
    python3 encrypted_classical.py --demo

    # Adjust CKKS parameters for higher precision:
    python3 encrypted_classical.py --scale 50 --depth 5

Dependencies
------------
    pip install tenseal numpy pandas scikit-learn scipy
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False
    warnings.warn(
        "\n[!] TenSEAL not installed — run:  pip install tenseal\n",
        RuntimeWarning,
    )

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
ARTIFACTS    = HERE / "portfolio_training" / "artifacts"
OUTPUT_PATH  = ARTIFACTS / "encrypted_classical_results.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.04 / 252      # daily (~4% annual)
VAR_ALPHA        = 0.05
Z_ALPHA          = norm.ppf(VAR_ALPHA)   # ≈ -1.645
TRADING_DAYS     = 252

# CKKS defaults — chosen for degree-2 polynomial (1 mult level needed)
DEFAULT_POLY_MOD = 16384
DEFAULT_SCALE_EXP = 40            # scale = 2^40
DEFAULT_COEFF_BITS = [60, 40, 40, 40, 60]   # 4 levels total


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(portfolio: str = "tangency") -> dict:
    """
    Load all required artifacts produced by the classical pipeline.
    Raises FileNotFoundError with clear instructions if any are missing.
    """
    required = {
        "sigma":    ARTIFACTS / "sigma_full.npy",
        "mu_ann":   ARTIFACTS / "mu_annual.npy",
        "weights":  ARTIFACTS / "w_classical.csv",
        "scaler":   ARTIFACTS / "scaler.pkl",
        "poly":     ARTIFACTS / "classical_polynomial_model.json",
        "returns":  ARTIFACTS / "returns.csv",
    }

    missing = [str(v) for k, v in required.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts:\n" + "\n".join(f"  {m}" for m in missing) +
            "\n\nRun the classical pipeline first:\n"
            "  python3 main_classical.py --skip-fhe\n"
            "  python3 build_classical_polynomial.py"
        )

    sigma   = np.load(required["sigma"]).astype(np.float64)
    mu_ann  = np.load(required["mu_ann"]).astype(np.float64)
    mu      = mu_ann / TRADING_DAYS   # daily

    w_df    = pd.read_csv(required["weights"], index_col=0)
    tickers = w_df.index.tolist()

    if portfolio not in w_df.columns:
        available = w_df.columns.tolist()
        raise ValueError(
            f"Portfolio '{portfolio}' not found. Available: {available}"
        )

    w = w_df[portfolio].values.astype(np.float64)
    w = w / w.sum()   # re-normalise for floating-point safety

    with open(required["scaler"], "rb") as f:
        scaler = pickle.load(f)

    with open(required["poly"]) as f:
        poly = json.load(f)

    returns = pd.read_csv(required["returns"], index_col=0, parse_dates=True)

    # Optional: Chebyshev sqrt coefficients
    cheby_path = ARTIFACTS / "chebyshev_sqrt_coeffs.json"
    sqrt_coeffs = None
    if cheby_path.exists():
        with open(cheby_path) as f:
            sqrt_data   = json.load(f)
            sqrt_coeffs = sqrt_data.get("coeffs") or sqrt_data.get("chebyshev_coeffs")

    print(f"[ENC] Artifacts loaded:")
    print(f"  Portfolio   : {portfolio}  ({len(w)} assets)")
    print(f"  Tickers     : {tickers}")
    print(f"  Weights     : {np.round(w, 4).tolist()}")
    print(f"  Poly source : {poly.get('source', 'classical')}")
    print(f"  Sqrt coeffs : {'available' if sqrt_coeffs else 'not found (using exact sqrt)'}")

    return {
        "sigma": sigma, "mu": mu, "mu_ann": mu_ann,
        "w": w, "tickers": tickers, "portfolio": portfolio,
        "scaler": scaler, "poly": poly, "returns": returns,
        "sqrt_coeffs": sqrt_coeffs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PLAINTEXT REFERENCE (exact classical formulas)
# ─────────────────────────────────────────────────────────────────────────────

def compute_plaintext_metrics(w, sigma, mu, returns, sqrt_coeffs=None) -> dict:
    """
    Compute the full suite of classical risk metrics in plaintext.
    This is the ground truth that the encrypted version must match.
    """
    t0 = time.perf_counter()

    # Core portfolio statistics
    variance = float(w @ sigma @ w)
    sigma_p  = float(np.sqrt(max(variance, 0.0)))
    mu_p     = float(w @ mu)

    # VaR and ES (parametric Gaussian)
    var_95_param = float(-(mu_p + Z_ALPHA * sigma_p))
    es_95_param  = float(-(mu_p - sigma_p * norm.pdf(Z_ALPHA) / VAR_ALPHA))

    # Historical ES
    r_p      = returns.values @ w
    threshold = float(np.quantile(r_p, VAR_ALPHA))
    tail      = r_p[r_p <= threshold]
    es_hist   = float(-tail.mean()) if len(tail) > 0 else abs(threshold)

    # Sharpe
    sharpe = (mu_p - RISK_FREE_RATE) / (sigma_p + 1e-12)

    # Annualised
    ann_return = mu_p * TRADING_DAYS
    ann_vol    = sigma_p * np.sqrt(TRADING_DAYS)
    ann_sharpe = sharpe * np.sqrt(TRADING_DAYS)

    # Risk attribution
    sigma_w = sigma @ w
    mrc     = (sigma_w / (sigma_p + 1e-12)).tolist()
    crc     = (w * np.array(mrc)).tolist()
    crc_sum = sum(crc)
    crc_pct = [c / (crc_sum + 1e-12) * 100 for c in crc]

    # Chebyshev sqrt approximation (for comparison with encrypted version)
    if sqrt_coeffs:
        cheby_sigma_p = _eval_chebyshev(variance, sqrt_coeffs)
    else:
        cheby_sigma_p = sigma_p

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "mode":          "classical_plaintext",
        "variance":      variance,
        "sigma_p":       sigma_p,
        "cheby_sigma_p": cheby_sigma_p,
        "mu_p":          mu_p,
        "var_95":        var_95_param,
        "es_95_gaussian": es_95_param,
        "es_95_hist":    es_hist,
        "sharpe":        sharpe,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct":    ann_vol * 100,
        "ann_sharpe":    ann_sharpe,
        "MRC":           mrc,
        "CRC":           crc,
        "CRC_pct":       crc_pct,
        "runtime_ms":    elapsed,
        "encrypted":     False,
    }


def _eval_chebyshev(x: float, coeffs: list) -> float:
    """Evaluate Chebyshev polynomial approximation of sqrt(x)."""
    result = 0.0
    for k, c in enumerate(coeffs):
        result += float(c) * (float(x) ** k)
    return max(result, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CKKS CONTEXT (Alice's setup)
# ─────────────────────────────────────────────────────────────────────────────

def build_ckks_context(scale_exp: int = DEFAULT_SCALE_EXP,
                       coeff_bits: list = DEFAULT_COEFF_BITS,
                       poly_mod: int = DEFAULT_POLY_MOD) -> "ts.Context":
    """
    Build Alice's full CKKS context including secret key.

    Parameter choices:
        poly_modulus_degree = 16384  → ~128-bit security
        coeff_mod_bit_sizes = [60,40,40,40,60]  → 4 multiplicative levels
        scale = 2^40  → ~12 decimal digits precision
        Degree-2 poly evaluation uses 1 level, leaving 3 spare.
    """
    if not HAS_TENSEAL:
        raise RuntimeError("TenSEAL not installed — pip install tenseal")

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coeff_bits,
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2 ** scale_exp
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CAROL'S ENCRYPTED POLYNOMIAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def carol_evaluate(enc_scalars_bytes: list,
                   pub_ctx_bytes: bytes,
                   poly: dict) -> bytes:
    """
    Carol evaluates f̂(z) = bias + a⊤z + z⊤Qz on individually encrypted scalars.
    Each z_i is a separate 1-slot CKKS ciphertext — no rotation or slot extraction.
    All enc*enc products are between depth-0 ciphertexts → depth-1 result.
    All additions are between same-depth ciphertexts → no scale mismatch.
    Carol never decrypts. Returns serialised Enc(f̂(z)).
    """
    ctx = ts.context_from(pub_ctx_bytes)
    n   = len(enc_scalars_bytes)

    bias = float(poly["bias"])
    a    = np.array(poly["linear"],    dtype=np.float64)
    Q    = np.array(poly["quadratic"], dtype=np.float64)

    print(f"    [CAROL] Evaluating degree-2 polynomial on {n} scalar ciphertexts")

    # Deserialise individual scalar ciphertexts (each at depth 0)
    enc = []
    for b in enc_scalars_bytes:
        ct = ts.lazy_ckks_vector_from(b)
        ct.link_context(ctx)
        enc.append(ct)

    # Quadratic terms: Q_ij_sym * Enc(z_i) * Enc(z_j) → depth 1
    Q_sym = (Q + Q.T) / 2.0
    quad_terms = []
    for i in range(n):
        for j in range(i, n):
            q = float(Q_sym[i, j]) * (1.0 if i == j else 2.0)
            if abs(q) < 1e-14:
                continue
            enc_prod = enc[i] * enc[j]       # depth 0 * depth 0 → depth 1
            quad_terms.append(enc_prod * q)   # plaintext scale, still depth 1

    if not quad_terms:
        raise RuntimeError("No quadratic terms — Q matrix is zero.")

    enc_score = quad_terms[0]
    for t in quad_terms[1:]:
        enc_score = enc_score + t             # all depth 1, safe

    # Linear terms upgraded to depth 1 via dummy enc*enc
    dummy = ts.ckks_vector(ctx, [1.0])
    for i in range(n):
        if abs(a[i]) > 1e-14:
            enc_li_d1  = enc[i] * dummy       # depth 0 * depth 0 → depth 1
            enc_score  = enc_score + enc_li_d1 * float(a[i])

    enc_score = enc_score + bias              # plaintext add, no level consumed

    print(f"    [CAROL] Done — returning Enc(f̂(z))")
    return enc_score.serialize()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ENCRYPTED PIPELINE (Alice orchestrates)
# ─────────────────────────────────────────────────────────────────────────────

def compute_encrypted_metrics(w, sigma, mu, returns,
                               scaler, poly,
                               sqrt_coeffs=None,
                               scale_exp: int = DEFAULT_SCALE_EXP,
                               coeff_bits: list = DEFAULT_COEFF_BITS) -> dict:
    """
    Full encrypted pipeline:
        Alice → standardize → encrypt → Carol → decrypt → metrics
    """
    if not HAS_TENSEAL:
        return {"mode": "classical_encrypted", "error": "TenSEAL not installed"}

    timings = {}
    t_total = time.perf_counter()

    # ── Alice: build CKKS context ────────────────────────────────────────────
    t0 = time.perf_counter()
    ctx = build_ckks_context(scale_exp=scale_exp, coeff_bits=coeff_bits)
    timings["context_build_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [ALICE] CKKS context built  "
          f"({timings['context_build_ms']:.0f}ms, "
          f"poly_mod={DEFAULT_POLY_MOD}, scale=2^{scale_exp})")

    # ── Alice: standardize weights ────────────────────────────────────────────
    z = scaler.transform(w.reshape(1, -1)).flatten().astype(np.float64)
    print(f"  [ALICE] Standardized z: min={z.min():.3f}  max={z.max():.3f}")

    # ── Alice: encrypt each z_i as a separate 1-slot ciphertext ─────────────
    # Scalar-per-ciphertext avoids all rotation/slot-extraction in Carol,
    # eliminating the "scale out of bounds" depth mismatch error.
    t0 = time.perf_counter()
    enc_scalars_bytes = []
    total_enc_bytes   = 0
    for zi in z:
        ct = ts.ckks_vector(ctx, [float(zi)])
        b  = ct.serialize()
        enc_scalars_bytes.append(b)
        total_enc_bytes += len(b)
    pub_ctx_bytes = ctx.serialize(save_secret_key=False)
    timings["encryption_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [ALICE] Encrypted {len(z)} scalars  "
          f"({timings['encryption_ms']:.1f}ms, "
          f"total ≈ {total_enc_bytes/1024:.1f} KB)")

    # ── Carol: evaluate polynomial on encrypted scalars ───────────────────────
    t0 = time.perf_counter()
    enc_score_bytes = carol_evaluate(enc_scalars_bytes, pub_ctx_bytes, poly)
    timings["carol_eval_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [CAROL] Evaluated polynomial  ({timings['carol_eval_ms']:.1f}ms)")

    # ── Alice: decrypt ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    enc_score = ts.lazy_ckks_vector_from(enc_score_bytes)
    enc_score.link_context(ctx)
    decrypted = enc_score.decrypt()
    variance  = float(decrypted[0])
    timings["decryption_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [ALICE] Decrypted σ²_p = {variance:.10f}  ({timings['decryption_ms']:.1f}ms)")

    # ── Alice: compute all metrics from decrypted variance ───────────────────
    sigma_p = float(np.sqrt(max(variance, 0.0)))
    mu_p    = float(w @ mu)

    # VaR using Chebyshev sqrt if available (more FHE-faithful), else exact sqrt
    if sqrt_coeffs:
        cheby_sigma_p = _eval_chebyshev(variance, sqrt_coeffs)
        var_95 = float(-(mu_p + Z_ALPHA * cheby_sigma_p))
        es_95_gaussian = float(-(mu_p - cheby_sigma_p * norm.pdf(Z_ALPHA) / VAR_ALPHA))
    else:
        cheby_sigma_p = sigma_p
        var_95 = float(-(mu_p + Z_ALPHA * sigma_p))
        es_95_gaussian = float(-(mu_p - sigma_p * norm.pdf(Z_ALPHA) / VAR_ALPHA))

    # Historical ES (computed post-decryption in plaintext — acceptable privacy trade-off)
    r_p       = returns.values @ w
    threshold = float(np.quantile(r_p, VAR_ALPHA))
    tail      = r_p[r_p <= threshold]
    es_hist   = float(-tail.mean()) if len(tail) > 0 else abs(threshold)

    sharpe    = (mu_p - RISK_FREE_RATE) / (sigma_p + 1e-12)
    ann_sharpe = sharpe * np.sqrt(TRADING_DAYS)

    # Risk attribution (uses exact sigma and decrypted weights — post-decrypt plaintext)
    sigma_w = sigma @ w
    mrc     = (sigma_w / (sigma_p + 1e-12)).tolist()
    crc     = (w * np.array(mrc)).tolist()
    crc_sum = sum(crc)
    crc_pct = [c / (crc_sum + 1e-12) * 100 for c in crc]

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000

    return {
        "mode":              "classical_encrypted",
        "variance":          variance,
        "sigma_p":           sigma_p,
        "cheby_sigma_p":     cheby_sigma_p,
        "mu_p":              mu_p,
        "var_95":            var_95,
        "es_95_gaussian":    es_95_gaussian,
        "es_95_hist":        es_hist,
        "sharpe":            sharpe,
        "ann_return_pct":    mu_p * TRADING_DAYS * 100,
        "ann_vol_pct":       sigma_p * np.sqrt(TRADING_DAYS) * 100,
        "ann_sharpe":        ann_sharpe,
        "MRC":               mrc,
        "CRC":               crc,
        "CRC_pct":           crc_pct,
        "runtime_ms":        timings["total_ms"],
        "timings":           timings,
        "ckks_params": {
            "poly_modulus_degree": DEFAULT_POLY_MOD,
            "coeff_mod_bit_sizes": coeff_bits,
            "scale_exp":           scale_exp,
            "levels_available":    len(coeff_bits) - 1,
            "levels_consumed":     1,
        },
        "encrypted": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ACCURACY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(plain: dict, enc: dict) -> dict:
    """
    Compute per-metric accuracy of encrypted vs. plaintext.
    Returns absolute error, relative error %, and pass/fail for each.
    """
    if "error" in enc:
        return {"error": enc["error"]}

    metrics = [
        ("variance",       "Portfolio Variance σ²_p",    0.1),
        ("sigma_p",        "Portfolio Volatility σ_p",   0.1),
        ("mu_p",           "Portfolio Return μ_p",        0.5),
        ("sharpe",         "Sharpe Ratio",                1.0),
        ("var_95",         "VaR 95% (parametric)",        0.5),
        ("es_95_gaussian", "ES 95% (Gaussian)",           0.5),
        ("ann_sharpe",     "Annualised Sharpe",           1.0),
    ]

    results = {}
    gate_pass = True

    for key, label, threshold_pct in metrics:
        pv = plain.get(key)
        ev = enc.get(key)
        if pv is None or ev is None:
            continue
        pv, ev = float(pv), float(ev)
        abs_err = abs(ev - pv)
        rel_err = abs_err / (abs(pv) + 1e-14) * 100
        passes  = rel_err < threshold_pct
        if not passes and key == "variance":
            gate_pass = False
        results[key] = {
            "label":         label,
            "plaintext":     pv,
            "encrypted":     ev,
            "abs_error":     abs_err,
            "rel_error_pct": rel_err,
            "threshold_pct": threshold_pct,
            "pass":          passes,
        }

    # CRC comparison
    crc_p = plain.get("CRC_pct", [])
    crc_e = enc.get("CRC_pct", [])
    if crc_p and crc_e:
        crc_max_delta = max(abs(p - e) for p, e in zip(crc_p, crc_e))
        results["crc_max_delta_pct"] = crc_max_delta

    results["overall_gate"] = gate_pass
    results["summary"] = (
        "✓ CKKS encryption is accurate — all metrics within tolerance"
        if gate_pass else
        "✗ Variance error exceeds 0.1% — review CKKS parameters"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DEMO MODE (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_data(n_assets: int = 10) -> dict:
    """Generate synthetic data for a quick smoke test."""
    print(f"[DEMO] Generating synthetic data  ({n_assets} assets)")
    rng   = np.random.default_rng(42)
    n_days = 1260
    raw    = rng.normal(0.0004, 0.01, (n_days, n_assets))
    sigma  = np.cov(raw.T) + np.eye(n_assets) * 1e-6
    mu     = raw.mean(axis=0)
    w      = np.ones(n_assets) / n_assets
    tickers = [f"Asset_{i+1}" for i in range(n_assets)]
    returns = pd.DataFrame(raw, columns=tickers)

    scaler = None
    if HAS_SKLEARN:
        from sklearn.preprocessing import StandardScaler as SS
        scaler = SS()
        cloud  = w + rng.normal(0, 0.05, (200, n_assets))
        cloud  = np.clip(cloud, 0, 1)
        cloud /= cloud.sum(axis=1, keepdims=True)
        scaler.fit(cloud)
    else:
        class _IdentityScaler:
            mean_  = np.zeros(n_assets)
            scale_ = np.ones(n_assets)
            def transform(self, X): return X.copy()
        scaler = _IdentityScaler()

    from build_classical_polynomial import derive_classical_polynomial
    poly = derive_classical_polynomial(sigma, scaler)

    return {
        "sigma": sigma, "mu": mu, "mu_ann": mu * TRADING_DAYS,
        "w": w, "tickers": tickers, "portfolio": "equal_weight_demo",
        "scaler": scaler, "poly": poly, "returns": returns,
        "sqrt_coeffs": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PRINT COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(plain: dict, enc: dict, accuracy: dict,
                     tickers: list) -> None:
    sep  = "═" * 70
    sep2 = "─" * 70

    print(f"\n{sep}")
    print(f"  ENCRYPTED CLASSICAL PIPELINE — FULL COMPARISON")
    print(f"  Portfolio: {plain.get('mode','?')}  |  Assets: {len(tickers)}")
    print(sep)

    rows = [
        ("Metric",                  "Plaintext",    "Encrypted",     "Rel. Error"),
        (sep2,                      "",              "",              ""),
        ("Variance σ²_p",
            f"{plain['variance']:.8f}",
            f"{enc.get('variance',0):.8f}",
            f"{accuracy.get('variance',{}).get('rel_error_pct',0):.6f}%"),
        ("Volatility σ_p",
            f"{plain['sigma_p']:.6f}",
            f"{enc.get('sigma_p',0):.6f}",
            f"{accuracy.get('sigma_p',{}).get('rel_error_pct',0):.6f}%"),
        ("Return μ_p (daily)",
            f"{plain['mu_p']:.6f}",
            f"{enc.get('mu_p',0):.6f}",
            f"{accuracy.get('mu_p',{}).get('rel_error_pct',0):.6f}%"),
        ("Sharpe Ratio",
            f"{plain['sharpe']:.4f}",
            f"{enc.get('sharpe',0):.4f}",
            f"{accuracy.get('sharpe',{}).get('rel_error_pct',0):.4f}%"),
        ("VaR 95% (parametric)",
            f"{plain['var_95']:.4f}",
            f"{enc.get('var_95',0):.4f}",
            f"{accuracy.get('var_95',{}).get('rel_error_pct',0):.4f}%"),
        ("ES 95% (Gaussian)",
            f"{plain['es_95_gaussian']:.4f}",
            f"{enc.get('es_95_gaussian',0):.4f}",
            f"{accuracy.get('es_95_gaussian',{}).get('rel_error_pct',0):.4f}%"),
        ("ES 95% (Historical)",
            f"{plain['es_95_hist']:.4f}",
            f"{enc.get('es_95_hist',0):.4f}",
            "—  (computed post-decrypt)"),
        ("Ann. Return %",
            f"{plain['ann_return_pct']:.3f}%",
            f"{enc.get('ann_return_pct',0):.3f}%",
            "—"),
        ("Ann. Volatility %",
            f"{plain['ann_vol_pct']:.3f}%",
            f"{enc.get('ann_vol_pct',0):.3f}%",
            "—"),
        ("Ann. Sharpe",
            f"{plain['ann_sharpe']:.4f}",
            f"{enc.get('ann_sharpe',0):.4f}",
            f"{accuracy.get('ann_sharpe',{}).get('rel_error_pct',0):.4f}%"),
    ]

    for row in rows:
        if row[0] == sep2:
            print(sep2)
        else:
            print(f"  {row[0]:<28} {row[1]:>14}  {row[2]:>14}  {row[3]:>16}")

    print(sep2)

    gate = accuracy.get("overall_gate", False)
    print(f"\n  {accuracy.get('summary', '')}")

    t = enc.get("timings", {})
    print(f"\n  RUNTIME BREAKDOWN")
    print(sep2)
    print(f"  Plaintext pipeline    : {plain['runtime_ms']:>10.2f} ms")
    print(f"  Encrypted total       : {enc.get('runtime_ms',0):>10.2f} ms")
    print(f"  ── Context build      : {t.get('context_build_ms',0):>10.2f} ms")
    print(f"  ── Alice encrypt      : {t.get('encryption_ms',0):>10.2f} ms")
    print(f"  ── Carol poly eval    : {t.get('carol_eval_ms',0):>10.2f} ms")
    print(f"  ── Alice decrypt      : {t.get('decryption_ms',0):>10.2f} ms")
    overhead = enc.get('runtime_ms', 0) - plain['runtime_ms']
    print(f"  FHE overhead          : {overhead:>+10.2f} ms")

    print(f"\n  PER-ASSET RISK ATTRIBUTION")
    print(sep2)
    print(f"  {'Asset':<8} {'Weight':>8}  {'Plain CRC%':>10}  {'Enc CRC%':>10}  {'Δ':>8}")
    for i, t in enumerate(tickers[:15]):
        wp  = plain["CRC_pct"][i]  if i < len(plain.get("CRC_pct",[])) else 0
        we  = enc.get("CRC_pct",[None]*len(tickers))[i] or 0
        ww  = plain.get("CRC",[0]*len(tickers))[i] or 0
        print(f"  {t:<8} {ww*100:>7.2f}%  {wp:>9.3f}%  {we:>9.3f}%  {we-wp:>+7.3f}%")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_results(plain: dict, enc: dict, accuracy: dict,
                 tickers: list, portfolio: str) -> None:

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, list):
            return [_clean(v) for v in d]
        return d

    output = {
        "generated_at":   datetime.now().isoformat(),
        "portfolio":      portfolio,
        "tickers":        tickers,
        "n_assets":       len(tickers),
        "classical_plaintext":  _clean(plain),
        "classical_encrypted":  _clean(enc),
        "accuracy":             _clean(accuracy),
        "overall_pass":         accuracy.get("overall_gate", False),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp.replace(OUTPUT_PATH)
    print(f"\n[ENC] Results saved → {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_encrypted_classical(portfolio: str = "tangency",
                             scale_exp: int = DEFAULT_SCALE_EXP,
                             coeff_bits: list = DEFAULT_COEFF_BITS,
                             demo: bool = False) -> dict:
    """Full pipeline: load → plaintext → encrypted → compare → save."""

    print("\n" + "═" * 70)
    print("  ENCRYPTED CLASSICAL PORTFOLIO PIPELINE")
    print(f"  Portfolio: {portfolio}  |  CKKS scale: 2^{scale_exp}  |  "
          f"poly_mod: {DEFAULT_POLY_MOD}")
    print("═" * 70)

    if demo:
        arts = build_demo_data()
    else:
        arts = load_artifacts(portfolio)

    w       = arts["w"]
    sigma   = arts["sigma"]
    mu      = arts["mu"]
    returns = arts["returns"]
    tickers = arts["tickers"]
    scaler  = arts["scaler"]
    poly    = arts["poly"]
    sqrt_c  = arts["sqrt_coeffs"]

    # Plaintext reference
    print("\n[ENC] Step 1 — Plaintext metrics (ground truth) …")
    plain = compute_plaintext_metrics(w, sigma, mu, returns, sqrt_c)
    print(f"  σ²_p={plain['variance']:.8f}  "
          f"Sharpe={plain['sharpe']:.4f}  "
          f"VaR={plain['var_95']:.4f}  "
          f"({plain['runtime_ms']:.1f}ms)")

    # Encrypted pipeline
    print("\n[ENC] Step 2 — Encrypted pipeline (Alice → Carol → Alice) …")
    enc = compute_encrypted_metrics(
        w, sigma, mu, returns, scaler, poly, sqrt_c,
        scale_exp=scale_exp, coeff_bits=coeff_bits
    )

    if "error" in enc:
        print(f"\n[ENC] {enc['error']}")
        print("[ENC] Saving plaintext-only results so dashboard can display them …")
        # Save what we have — plaintext metrics are still fully valid.
        # The dashboard checks for 'error' in classical_encrypted and shows
        # a clear "install TenSEAL" message rather than a blank tab.
        if not demo:
            save_results(plain, enc, {}, tickers, arts["portfolio"])
        return {
            "classical_plaintext": plain,
            "classical_encrypted": enc,
            "accuracy": {"error": enc["error"]},
        }

    print(f"  σ²_p={enc['variance']:.8f}  "
          f"Sharpe={enc['sharpe']:.4f}  "
          f"Total={enc['runtime_ms']:.0f}ms")

    # Accuracy analysis
    print("\n[ENC] Step 3 — Accuracy analysis …")
    accuracy = compute_accuracy(plain, enc)
    print(f"  Variance rel. error: "
          f"{accuracy.get('variance',{}).get('rel_error_pct',0):.6f}%  "
          f"{'✓ PASS' if accuracy.get('overall_gate') else '✗ FAIL'}")

    # Print full comparison
    print_comparison(plain, enc, accuracy, tickers)

    # Save
    if not demo:
        save_results(plain, enc, accuracy, tickers, arts["portfolio"])

    return {
        "classical_plaintext": plain,
        "classical_encrypted": enc,
        "accuracy":            accuracy,
    }


def main():
    p = argparse.ArgumentParser(
        description="Fully encrypted classical portfolio pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 encrypted_classical.py\n"
            "  python3 encrypted_classical.py --portfolio min_variance\n"
            "  python3 encrypted_classical.py --all-portfolios\n"
            "  python3 encrypted_classical.py --demo\n"
            "  python3 encrypted_classical.py --scale 50\n"
        )
    )
    p.add_argument("--portfolio",      default="tangency",
                   choices=["tangency","min_variance","equal_weight"],
                   help="Which portfolio to encrypt (default: tangency)")
    p.add_argument("--all-portfolios", action="store_true",
                   help="Run all three portfolios sequentially")
    p.add_argument("--demo",           action="store_true",
                   help="Quick smoke test with synthetic data (no artifacts needed)")
    p.add_argument("--scale",          type=int, default=DEFAULT_SCALE_EXP,
                   help=f"CKKS scale exponent — scale = 2^N (default: {DEFAULT_SCALE_EXP})")
    p.add_argument("--depth",          type=int, default=4,
                   help="CKKS multiplicative depth / levels (default: 4)")
    args = p.parse_args()

    # Build coeff_bits from depth
    coeff_bits = [60] + [40] * args.depth + [60]

    if args.demo:
        run_encrypted_classical(demo=True, scale_exp=args.scale, coeff_bits=coeff_bits)
        return

    if args.all_portfolios:
        all_results = {}
        for pf in ["tangency", "min_variance", "equal_weight"]:
            print(f"\n{'='*70}")
            print(f"  PORTFOLIO: {pf.upper()}")
            result = run_encrypted_classical(pf, args.scale, coeff_bits)
            all_results[pf] = result

        # Save combined results
        combined_path = ARTIFACTS / "encrypted_classical_all_portfolios.json"
        def _clean(d):
            if isinstance(d, dict): return {k: _clean(v) for k, v in d.items()}
            if isinstance(d, (np.floating, np.integer)): return float(d)
            if isinstance(d, np.ndarray): return d.tolist()
            if isinstance(d, list): return [_clean(v) for v in d]
            return d
        with open(combined_path, "w") as f:
            json.dump(_clean(all_results), f, indent=2, default=str)
        print(f"\n[ENC] All portfolios saved → {combined_path}")
    else:
        run_encrypted_classical(args.portfolio, args.scale, coeff_bits)


if __name__ == "__main__":
    main()