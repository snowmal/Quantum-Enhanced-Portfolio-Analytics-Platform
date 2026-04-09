"""
run_fhe_comparison.py
=====================
Implements CKKS/FHE encryption on the classical portfolio pipeline and
produces a full side-by-side comparison between:

    [A] Classical Plaintext   — exact Markowitz formulas, no encryption
    [B] Classical Encrypted   — same math, weights encrypted via CKKS

Both modes compute:
    • Portfolio variance  (σ²_p = w⊤Σw)
    • Portfolio return    (μ_p  = w⊤μ)
    • VaR at 95%         (parametric Gaussian)
    • Expected Shortfall  (historical tail average)
    • Sharpe Ratio
    • Per-asset Risk Attribution (MRC, CRC)
    • Encryption overhead (runtime delta)

The degree-2 polynomial surrogate of w⊤Σw is what Carol evaluates
in ciphertext — identical math, just computed on encrypted values.

Usage
-----
    python3 run_fhe_comparison.py \
        --weights  portfolio_training/artifacts/w_classical.csv \
        --returns  portfolio_training/artifacts/returns.csv \
        --cov      portfolio_training/artifacts/covariance.npz \
        --mu       portfolio_training/artifacts/expected_returns.npy \
        --scaler   portfolio_training/artifacts/scaler.pkl \
        --row      0          # which row of w_classical.csv to use
        --output   portfolio_training/artifacts/fhe_comparison.json

    # Quick synthetic demo (no artifact files needed):
    python3 run_fhe_comparison.py --demo

Dependencies
------------
    pip install tenseal numpy pandas scikit-learn scipy tabulate
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False
    warnings.warn(
        "\n[!] TenSEAL not installed — encrypted mode will be skipped.\n"
        "    pip install tenseal\n",
        RuntimeWarning,
    )

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.04 / 252      # ~4% annual, dailised
VAR_ALPHA      = 0.05            # 5% VaR / ES level
Z_ALPHA        = norm.ppf(VAR_ALPHA)  # ≈ -1.645


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CKKS CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def build_ckks_context() -> "ts.Context":
    """
    Build Alice's full CKKS context (includes secret key).

    Budget:
      poly_modulus_degree = 16384  → max ~200-bit security
      coeff_mod_bit_sizes = [60, 40, 40, 40, 60]  → 4 levels
      scale = 2**40  → ~12 decimal digits of precision

    Degree-2 polynomial needs 1 multiplicative level (z_i * z_j).
    4 levels is conservative and leaves headroom for Chebyshev sqrt
    when we extend to the quantum pipeline.
    """
    if not HAS_TENSEAL:
        raise RuntimeError("TenSEAL is required for encrypted mode.")

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2 ** 40
    return ctx


def public_context_bytes(ctx: "ts.Context") -> bytes:
    """Serialize the context WITHOUT the secret key (Carol's view)."""
    return ctx.serialize(save_secret_key=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — POLYNOMIAL SURROGATE FOR w⊤Σw
#
# w⊤Σw is already a degree-2 polynomial in w, so we don't need to fit
# anything — we can read the coefficients directly from Σ.
#
# f(w) = Σ_{i,j} Σ_{ij} · w_i · w_j
#       = Σ_i Σ_{ii} · w_i²  +  2 Σ_{i<j} Σ_{ij} · w_i · w_j
#
# In the standard form  f̂(z) = bias + a⊤z + z⊤Qz  we have:
#   bias = 0
#   a    = 0  (no linear terms in w⊤Σw)
#   Q    = Σ  (the covariance matrix itself)
#
# After standardizing w → z via the scaler, the polynomial becomes:
#   f̂(z) = bias_s + a_s⊤z + z⊤Q_s z
# where the coefficients absorb the scaler's mean and scale.
# ─────────────────────────────────────────────────────────────────────────────

def build_classical_polynomial(sigma: np.ndarray,
                                scaler) -> dict:
    """
    Derive the degree-2 polynomial for w⊤Σw in standardized weight space.

    If scaler is None, assumes weights are already standardized (z = w).

    The scaler transforms  z = (w - mean) / scale, so
        w = z * scale + mean
        w⊤Σw = (z*scale + mean)⊤ Σ (z*scale + mean)
             = z⊤ (diag(scale) Σ diag(scale)) z
               + 2 mean⊤ Σ diag(scale) z
               + mean⊤ Σ mean

    Returns dict with keys: bias (float), linear (list), quadratic (list of lists)
    """
    n = sigma.shape[0]

    if scaler is not None:
        s    = np.array(scaler.scale_, dtype=np.float64)    # (n,)
        m    = np.array(scaler.mean_,  dtype=np.float64)    # (n,)
        S    = np.diag(s)
        Q_s  = S @ sigma @ S
        a_s  = 2.0 * (sigma @ m) * s        # 2 * diag(s) * Σ * m
        bias = float(m @ sigma @ m)
    else:
        Q_s  = sigma.copy()
        a_s  = np.zeros(n, dtype=np.float64)
        bias = 0.0

    return {
        "bias":      bias,
        "linear":    a_s.tolist(),
        "quadratic": Q_s.tolist(),
        "n":         n,
        "source":    "classical_w_sigma_w",
    }


def evaluate_poly_plaintext(z: np.ndarray, poly: dict) -> float:
    """f̂(z) = bias + a⊤z + z⊤Qz — plaintext reference."""
    bias = float(poly["bias"])
    a    = np.array(poly["linear"],    dtype=np.float64)
    Q    = np.array(poly["quadratic"], dtype=np.float64)
    return bias + float(a @ z) + float(z @ Q @ z)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CAROL'S ENCRYPTED POLYNOMIAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def carol_evaluate_encrypted(enc_scalars_bytes: list,
                              pub_ctx_bytes: bytes,
                              poly: dict) -> bytes:
    """
    Carol evaluates f̂(z) = bias + a⊤z + z⊤Qz on individually encrypted scalars.

    Alice sends each z_i as a separate 1-slot CKKS ciphertext (enc_scalars_bytes
    is a list of n serialised ciphertexts). Carol receives no secret key.

    All operations are scalar × scalar:
      - enc_i * float(a_i)           depth 0 → 0 (plaintext multiply)
      - enc_i * enc_j                depth 0+0 → 1 (enc × enc, costs 1 level)
      - enc_prod * float(Q_ij)       depth 1 → 1 (plaintext multiply)
      - enc + enc (same depth)       safe
      - enc + float (bias)           safe

    No rotation. No slot extraction. No scale mismatch.
    """
    if not HAS_TENSEAL:
        raise RuntimeError("TenSEAL required for Carol's encrypted evaluation.")

    ctx = ts.context_from(pub_ctx_bytes)
    n   = len(enc_scalars_bytes)

    bias = float(poly["bias"])
    a    = np.array(poly["linear"],    dtype=np.float64)
    Q    = np.array(poly["quadratic"], dtype=np.float64)

    print(f"    [CAROL] Evaluating degree-2 polynomial on {n} scalar ciphertexts")

    # Deserialise individual scalar ciphertexts (each is depth 0)
    enc = []
    for b in enc_scalars_bytes:
        ct = ts.lazy_ckks_vector_from(b)
        ct.link_context(ctx)
        enc.append(ct)

    # Linear terms: a_i * Enc(z_i) — plaintext multiply, stays depth 0
    linear_terms = []
    for i in range(n):
        if abs(a[i]) > 1e-14:
            linear_terms.append(enc[i] * float(a[i]))

    # Quadratic terms: Q_ij_sym * Enc(z_i) * Enc(z_j) — one enc×enc, depth 1
    Q_sym = (Q + Q.T) / 2.0
    quad_terms = []
    for i in range(n):
        for j in range(i, n):
            q = float(Q_sym[i, j]) * (1.0 if i == j else 2.0)
            if abs(q) < 1e-14:
                continue
            enc_prod = enc[i] * enc[j]          # depth 0 * depth 0 → depth 1
            quad_terms.append(enc_prod * q)      # plaintext multiply, depth 1

    # Accumulate quadratic terms (all depth 1)
    if not quad_terms:
        raise RuntimeError("No quadratic terms — check Q matrix.")
    enc_score = quad_terms[0]
    for t in quad_terms[1:]:
        enc_score = enc_score + t

    # Upgrade linear terms from depth 0 to depth 1 by multiplying a dummy
    # depth-0 ones-ciphertext (enc * enc costs 1 level).
    # We reuse enc[0] * enc[0] / enc[0] ... too complex.
    # Simplest: linear contribution is small; compute it as a plaintext
    # scalar added post-hoc. Since Carol cannot decrypt, we add the linear
    # contribution using enc[i] scaled to depth 1 via a dummy mul:
    # enc[i] * enc[i] gives Enc(z_i^2) — wrong value.
    # Correct: dummy = ts.ckks_vector(ctx, [1.0]) then enc[i] * dummy → depth 1.
    dummy = ts.ckks_vector(ctx, [1.0])
    for i in range(n):
        if abs(a[i]) > 1e-14:
            enc_li_d1 = enc[i] * dummy           # enc * enc → depth 1
            enc_score = enc_score + enc_li_d1 * float(a[i])

    # Add bias (plaintext constant, no level consumed)
    enc_score = enc_score + bias

    print(f"    [CAROL] Done — returning Enc(f̂(z))")
    return enc_score.serialize()



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RISK METRICS (shared, applied post-decryption or plaintext)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_metrics(w: np.ndarray,
                          sigma: np.ndarray,
                          mu: np.ndarray,
                          variance: float,
                          returns: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute full suite of risk metrics given a variance estimate.
    Works identically whether variance came from plaintext or decrypted FHE.

    Parameters
    ----------
    w        : portfolio weights (N,)
    sigma    : covariance matrix (N, N) — used only for MRC/CRC
    mu       : expected daily returns (N,)
    variance : σ²_p — the key scalar, either plaintext or decrypted
    returns  : daily returns DataFrame for historical ES (optional)
    """
    n       = len(w)
    mu_p    = float(w @ mu)
    sigma_p = float(np.sqrt(max(variance, 0.0)))

    # ── Parametric VaR and ES (Gaussian) ─────────────────────────────────────
    var_95  = -(mu_p + Z_ALPHA * sigma_p)          # positive = loss
    es_gauss = -(mu_p + norm.pdf(Z_ALPHA) / VAR_ALPHA * sigma_p)

    # ── Historical ES (if returns provided) ──────────────────────────────────
    es_hist = None
    if returns is not None:
        port_ret  = returns.values @ w
        threshold = np.quantile(port_ret, VAR_ALPHA)
        tail      = port_ret[port_ret <= threshold]
        es_hist   = float(-tail.mean()) if len(tail) > 0 else None

    # ── Sharpe ratio ─────────────────────────────────────────────────────────
    sharpe = (mu_p - RISK_FREE_RATE) / sigma_p if sigma_p > 1e-12 else 0.0

    # ── Risk attribution ─────────────────────────────────────────────────────
    sigma_w = sigma @ w
    mrc     = (sigma_w / sigma_p).tolist() if sigma_p > 1e-12 else [0.0] * n
    crc     = (w * np.array(mrc)).tolist()
    crc_sum = sum(crc)
    crc_pct = [c / crc_sum for c in crc] if abs(crc_sum) > 1e-12 else [0.0] * n

    return {
        "variance":   variance,
        "sigma_p":    sigma_p,
        "mu_p":       mu_p,
        "VaR_95":     var_95,
        "ES_gaussian": es_gauss,
        "ES_historical": es_hist,
        "sharpe":     sharpe,
        "MRC":        mrc,
        "CRC":        crc,
        "CRC_pct":    crc_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PIPELINE A: CLASSICAL PLAINTEXT
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_plaintext(w: np.ndarray,
                             sigma: np.ndarray,
                             mu: np.ndarray,
                             returns: Optional[pd.DataFrame]) -> dict:
    print("\n  [A] Classical Plaintext")
    print("  " + "─" * 48)

    t0       = time.perf_counter()
    variance = float(w @ sigma @ w)
    metrics  = compute_risk_metrics(w, sigma, mu, variance, returns)
    elapsed  = time.perf_counter() - t0

    print(f"      σ²_p (exact)  = {variance:.8f}")
    print(f"      σ_p           = {metrics['sigma_p']:.6f}")
    print(f"      μ_p           = {metrics['mu_p']:.6f}")
    print(f"      Sharpe        = {metrics['sharpe']:.4f}")
    print(f"      VaR 95%       = {metrics['VaR_95']:.4f}")
    print(f"      Runtime       = {elapsed*1000:.2f} ms")

    return {"mode": "classical_plaintext", "runtime_ms": elapsed * 1000, **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PIPELINE B: CLASSICAL ENCRYPTED (CKKS)
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_encrypted(w: np.ndarray,
                             sigma: np.ndarray,
                             mu: np.ndarray,
                             returns: Optional[pd.DataFrame],
                             scaler) -> dict:
    print("\n  [B] Classical Encrypted (CKKS)")
    print("  " + "─" * 48)

    if not HAS_TENSEAL:
        print("      [SKIP] TenSEAL not available.")
        return {"mode": "classical_encrypted", "error": "TenSEAL not installed"}

    t0 = time.perf_counter()

    # ── 1. Build polynomial from Σ ───────────────────────────────────────────
    poly = build_classical_polynomial(sigma, scaler)
    print(f"      Polynomial built from Σ  (n={poly['n']}, bias={poly['bias']:.6f})")

    # ── 2. Standardize weights ───────────────────────────────────────────────
    if scaler is not None:
        z = scaler.transform(w.reshape(1, -1)).flatten().astype(np.float64)
    else:
        z = w.astype(np.float64)
    print(f"      Standardized z: min={z.min():.3f}  max={z.max():.3f}")

    # ── 3. Build CKKS context (Alice) ────────────────────────────────────────
    t_ctx = time.perf_counter()
    ctx   = build_ckks_context()
    t_ctx = time.perf_counter() - t_ctx
    print(f"      CKKS context built  ({t_ctx*1000:.0f} ms)")

    # ── 4. Alice encrypts each z_i as a separate 1-slot ciphertext ──────────
    # Encrypting scalars individually avoids all rotation/slot-extraction
    # in Carol's evaluation, which is the root cause of scale mismatch errors.
    t_enc = time.perf_counter()
    n_assets = len(z)
    enc_scalars_bytes = []
    total_bytes = 0
    for zi in z:
        ct    = ts.ckks_vector(ctx, [float(zi)])
        b     = ct.serialize()
        enc_scalars_bytes.append(b)
        total_bytes += len(b)
    pub_bytes = public_context_bytes(ctx)
    t_enc = time.perf_counter() - t_enc
    print(f"      Alice encrypted {n_assets} scalars  ({t_enc*1000:.1f} ms, "
          f"total ≈ {total_bytes/1024:.1f} KB)")

    # ── 5. Carol evaluates polynomial on encrypted scalars ───────────────────
    t_carol = time.perf_counter()
    enc_score_bytes = carol_evaluate_encrypted(enc_scalars_bytes, pub_bytes, poly)
    t_carol = time.perf_counter() - t_carol
    print(f"      Carol evaluated polynomial  ({t_carol*1000:.1f} ms)")

    # ── 6. Alice decrypts ────────────────────────────────────────────────────
    t_dec = time.perf_counter()
    enc_out = ts.lazy_ckks_vector_from(enc_score_bytes)
    enc_out.link_context(ctx)
    decrypted = enc_out.decrypt()
    variance  = float(decrypted[0])
    t_dec = time.perf_counter() - t_dec
    print(f"      Alice decrypted  ({t_dec*1000:.1f} ms)")
    print(f"      σ²_p (encrypted) = {variance:.8f}")

    # ── 7. Compute risk metrics (post-decryption, plaintext math) ────────────
    metrics = compute_risk_metrics(w, sigma, mu, variance, returns)
    elapsed = time.perf_counter() - t0

    print(f"      σ_p           = {metrics['sigma_p']:.6f}")
    print(f"      μ_p           = {metrics['mu_p']:.6f}")
    print(f"      Sharpe        = {metrics['sharpe']:.4f}")
    print(f"      VaR 95%       = {metrics['VaR_95']:.4f}")
    print(f"      Total runtime = {elapsed*1000:.1f} ms")

    return {
        "mode":       "classical_encrypted",
        "runtime_ms": elapsed * 1000,
        "timing": {
            "context_build_ms":    t_ctx  * 1000,
            "encryption_ms":       t_enc  * 1000,
            "carol_evaluation_ms": t_carol * 1000,
            "decryption_ms":       t_dec  * 1000,
        },
        **metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(plain: dict, enc: dict, n_assets: int) -> None:
    """
    Print a structured side-by-side comparison of the two modes.
    """
    enc_ok = "error" not in enc

    # Absolute error on variance (the key encrypted quantity)
    var_error   = abs(enc["variance"] - plain["variance"]) if enc_ok else None
    var_rel_err = var_error / abs(plain["variance"]) * 100 if (var_error is not None and abs(plain["variance"]) > 1e-14) else None

    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print("  CLASSICAL PLAINTEXT  vs.  CLASSICAL ENCRYPTED (CKKS)")
    print(f"  Portfolio: {n_assets} assets    CKKS scale: 2^40    poly_mod_degree: 16384")
    print(sep)

    rows = [
        ("Portfolio Variance σ²_p",
            f"{plain['variance']:.8f}",
            f"{enc['variance']:.8f}"  if enc_ok else "N/A"),
        ("Portfolio Volatility σ_p",
            f"{plain['sigma_p']:.6f}",
            f"{enc['sigma_p']:.6f}"   if enc_ok else "N/A"),
        ("Portfolio Return μ_p (daily)",
            f"{plain['mu_p']:.6f}",
            f"{enc['mu_p']:.6f}"      if enc_ok else "N/A"),
        ("Sharpe Ratio",
            f"{plain['sharpe']:.4f}",
            f"{enc['sharpe']:.4f}"    if enc_ok else "N/A"),
        ("VaR 95% (parametric)",
            f"{plain['VaR_95']:.4f}",
            f"{enc['VaR_95']:.4f}"    if enc_ok else "N/A"),
        ("ES  95% (Gaussian)",
            f"{plain['ES_gaussian']:.4f}",
            f"{enc['ES_gaussian']:.4f}" if enc_ok else "N/A"),
        ("ES  95% (Historical)",
            f"{plain['ES_historical']:.4f}"  if plain['ES_historical'] else "N/A",
            f"{enc['ES_historical']:.4f}"    if (enc_ok and enc['ES_historical']) else "N/A"),
    ]

    if HAS_TABULATE:
        print(tabulate(
            rows,
            headers=["Metric", "Plaintext", "Encrypted"],
            tablefmt="simple",
            colalign=("left", "right", "right"),
        ))
    else:
        hdr = f"  {'Metric':<34}  {'Plaintext':>14}  {'Encrypted':>14}"
        print(hdr)
        print(sep2)
        for label, pval, eval_ in rows:
            print(f"  {label:<34}  {pval:>14}  {eval_:>14}")

    print(sep2)

    if enc_ok:
        print(f"\n  ENCRYPTION ACCURACY")
        print(sep2)
        print(f"  Variance absolute error   : {var_error:.2e}")
        print(f"  Variance relative error   : {var_rel_err:.4f}%")
        print(f"  {'PASS ✓' if var_rel_err < 0.1 else 'WARN ✗ — error > 0.1%, check CKKS params'}")

        print(f"\n  RUNTIME BREAKDOWN (ms)")
        print(sep2)
        t = enc.get("timing", {})
        plain_ms = plain["runtime_ms"]
        enc_ms   = enc["runtime_ms"]
        print(f"  Plaintext total           : {plain_ms:>10.2f} ms")
        print(f"  Encrypted total           : {enc_ms:>10.2f} ms")
        print(f"  ── Context build          : {t.get('context_build_ms', 0):>10.2f} ms")
        print(f"  ── Encryption (Alice)     : {t.get('encryption_ms', 0):>10.2f} ms")
        print(f"  ── Poly eval  (Carol)     : {t.get('carol_evaluation_ms', 0):>10.2f} ms")
        print(f"  ── Decryption (Alice)     : {t.get('decryption_ms', 0):>10.2f} ms")
        overhead = enc_ms - plain_ms
        print(f"  FHE overhead              : {overhead:>+10.2f} ms")

    print(f"\n  PER-ASSET RISK ATTRIBUTION  (Component Risk Contribution %)")
    print(sep2)
    crc_p  = plain["CRC_pct"]
    crc_e  = enc["CRC_pct"] if enc_ok else [None] * len(crc_p)
    for i, (cp, ce) in enumerate(zip(crc_p, crc_e)):
        plain_pct = f"{cp*100:>6.2f}%"
        enc_pct   = f"{ce*100:>6.2f}%" if ce is not None else "  N/A "
        delta     = f"Δ{(ce-cp)*100:+.4f}%" if ce is not None else ""
        print(f"  Asset {i+1:>2}  Plaintext: {plain_pct}   Encrypted: {enc_pct}   {delta}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(plain: dict, enc: dict, output_path: str) -> None:
    """Save both result dicts to a single JSON for downstream use."""

    def _clean(d: dict) -> dict:
        """Convert numpy types to native Python for JSON serialisation."""
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                out[k] = _clean(v)
            else:
                out[k] = v
        return out

    payload = {
        "classical_plaintext":  _clean(plain),
        "classical_encrypted":  _clean(enc),
        "comparison": {
            "variance_absolute_error": abs(enc["variance"] - plain["variance"])
                                       if "error" not in enc else None,
            "variance_relative_error_pct":
                abs(enc["variance"] - plain["variance"]) / abs(plain["variance"]) * 100
                if ("error" not in enc and abs(plain["variance"]) > 1e-14) else None,
            "sharpe_delta":
                enc["sharpe"] - plain["sharpe"] if "error" not in enc else None,
            "VaR_delta":
                enc["VaR_95"] - plain["VaR_95"] if "error" not in enc else None,
            "fhe_overhead_ms":
                enc["runtime_ms"] - plain["runtime_ms"] if "error" not in enc else None,
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DEMO MODE (synthetic data, no artifact files needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_data(n_assets: int = 10,
                    n_days:   int = 1260) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, pd.DataFrame]:
    """
    Generate realistic synthetic market data for a quick smoke test.
    Returns: w, sigma, mu, returns_df
    """
    print(f"\n  [DEMO] Generating synthetic data  "
          f"({n_assets} assets, {n_days} trading days)")

    rng = np.random.default_rng(42)

    # Correlated returns via a 3-factor model
    n_factors  = 3
    F_loadings = rng.uniform(0.2, 0.9, (n_assets, n_factors))
    factor_ret = rng.normal(0.0004, 0.01, (n_days, n_factors))
    idio_ret   = rng.normal(0.0001, 0.008, (n_days, n_assets))
    raw_returns = factor_ret @ F_loadings.T + idio_ret   # (T, N)

    returns_df = pd.DataFrame(
        raw_returns,
        columns=[f"Asset_{i+1}" for i in range(n_assets)],
    )

    # Covariance from returns
    sigma = np.cov(raw_returns.T) + np.eye(n_assets) * 1e-6  # regularise

    # Expected returns (annualised then dailised)
    mu = raw_returns.mean(axis=0)

    # Equal-weight portfolio (normalised)
    w = np.ones(n_assets) / n_assets

    return w, sigma, mu, returns_df


def build_demo_scaler(w: np.ndarray) -> Optional[object]:
    """Build a minimal scaler from demo weights."""
    if not HAS_SKLEARN:
        return None
    scaler = StandardScaler()
    # Fit on a small perturbation cloud around w so scaler sees variation
    rng    = np.random.default_rng(0)
    cloud  = w + rng.normal(0, 0.05, (200, len(w)))
    cloud  = np.clip(cloud, 0, 1)
    cloud /= cloud.sum(axis=1, keepdims=True)
    scaler.fit(cloud)
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FHE comparison: classical plaintext vs. CKKS encrypted"
    )
    p.add_argument("--demo",    action="store_true",
                   help="Run on synthetic data (no artifact files needed)")
    p.add_argument("--weights", type=str,
                   default="portfolio_training/artifacts/w_classical.csv")
    p.add_argument("--returns", type=str,
                   default="portfolio_training/artifacts/returns.csv")
    p.add_argument("--cov",     type=str,
                   default="portfolio_training/artifacts/covariance.npz")
    p.add_argument("--mu",      type=str,
                   default="portfolio_training/artifacts/expected_returns.npy")
    p.add_argument("--scaler",  type=str,
                   default="portfolio_training/artifacts/scaler.pkl")
    p.add_argument("--portfolio", type=str, default="tangency",
                   choices=["tangency", "min_variance", "equal_weight"],
                   help="Portfolio column to use from w_classical.csv (default: tangency)")
    p.add_argument("--row",     type=int, default=0,
                   help="(deprecated, ignored) use --portfolio instead")
    p.add_argument("--output",  type=str,
                   default="portfolio_training/artifacts/fhe_comparison.json")
    p.add_argument("--n-assets", type=int, default=10,
                   help="Number of assets for --demo mode (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 72)
    print("  CLASSICAL FHE COMPARISON — Plaintext vs. CKKS Encrypted")
    print("═" * 72)

    # ── Load or generate data ─────────────────────────────────────────────────
    if args.demo:
        w, sigma, mu, returns_df = build_demo_data(n_assets=args.n_assets)
        scaler = build_demo_scaler(w)
        print(f"  Mode: DEMO  (synthetic {args.n_assets}-asset portfolio)")

    else:
        # Load weights
        if not Path(args.weights).exists():
            print(f"\n  [ERROR] Weights file not found: {args.weights}")
            print("  Run with --demo to use synthetic data, or generate artifacts first.")
            sys.exit(1)

        w_df    = pd.read_csv(args.weights, index_col=0)
        # w_classical.csv is (assets x portfolios): index=tickers, columns=portfolio names
        # Read by column (portfolio), not by row
        available = w_df.columns.tolist()
        portfolio = args.portfolio if args.portfolio in available else available[0]
        w       = w_df[portfolio].values.astype(np.float64)
        w      /= w.sum()   # re-normalise for floating-point safety

        # Load covariance
        cov_data = np.load(args.cov)
        sigma    = cov_data["sigma"].astype(np.float64)

        # Load expected returns
        mu = np.load(args.mu).astype(np.float64)

        # Load returns for historical ES
        returns_df = pd.read_csv(args.returns, index_col=0, parse_dates=True) \
                     if Path(args.returns).exists() else None

        # Load scaler
        scaler = None
        if HAS_SKLEARN and Path(args.scaler).exists():
            with open(args.scaler, "rb") as f:
                scaler = pickle.load(f)
        else:
            print("  [WARN] Scaler not found — using raw weights as z (no standardization)")

        print(f"  Mode: REAL DATA  (portfolio={args.portfolio}, {len(w)} assets)")

    print(f"\n  Weights  : {np.round(w, 4)}")
    print(f"  Sum(w)   : {w.sum():.6f}  (should be 1.0)")

    # ── Run both pipelines ────────────────────────────────────────────────────
    plain_result = run_classical_plaintext(w, sigma, mu, returns_df
                                           if not args.demo else returns_df)
    enc_result   = run_classical_encrypted(w, sigma, mu,
                                           returns_df if not args.demo else returns_df,
                                           scaler)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(plain_result, enc_result, n_assets=len(w))

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(plain_result, enc_result, args.output)


if __name__ == "__main__":
    main()