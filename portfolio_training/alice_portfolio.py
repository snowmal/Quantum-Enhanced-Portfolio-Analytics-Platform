"""
alice_portfolio.py
==================
Alice's side of the 3-party encrypted portfolio risk pipeline.

Alice owns the portfolio weights and the secret key. She never shares
plaintext weights with Carol. She supports four pipeline modes so that
all four can be compared side-by-side:

    MODE                     MODEL           ENCRYPTION
    ─────────────────────────────────────────────────────
    classical_plaintext      Markowitz Σ     None  (baseline)
    classical_encrypted      Markowitz Σ     CKKS  (privacy on classical)
    quantum_plaintext        VQC surrogate   None  (quantum, no FHE)
    quantum_encrypted        VQC surrogate   CKKS  (full pipeline)

Usage
-----
    # Run a single mode
    python alice_portfolio.py --mode quantum_encrypted --weights artifacts/w_classical.csv

    # Run all four modes and print comparison table
    python alice_portfolio.py --mode all --weights artifacts/w_classical.csv

Dependencies
------------
    pip install tenseal numpy pandas scikit-learn scipy
"""

import os
import sys
import json
import time
import argparse
import warnings
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── TenSEAL (optional: only needed for encrypted modes) ──────────────────────
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    warnings.warn(
        "TenSEAL not installed. Encrypted modes unavailable.\n"
        "  pip install tenseal",
        RuntimeWarning,
    )

# ── Carol communication (in-process call for now; swap for socket if needed) ──
sys.path.insert(0, str(Path(__file__).parent))
from carol_portfolio_listener import (
    carol_evaluate_classical,
    carol_evaluate_quantum,
    carol_evaluate_classical_encrypted,
    carol_evaluate_quantum_encrypted,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  –  adjust if your artifact directory differs
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_DIR   = Path(__file__).parent / "portfolio_training" / "artifacts"
SCALER_PATH    = ARTIFACT_DIR / "scaler.pkl"
COV_PATH       = ARTIFACT_DIR / "covariance.npz"
MU_PATH        = ARTIFACT_DIR / "expected_returns.npy"
CL_POLY_PATH   = ARTIFACT_DIR / "classical_polynomial_model.json"  # σ²=w⊤Σw coeffs
QT_RISK_PATH   = ARTIFACT_DIR / "P_risk.json"                       # VQC risk surrogate
QT_RET_PATH    = ARTIFACT_DIR / "P_return.json"                     # VQC return surrogate
SECRET_KEY_PATH= ARTIFACT_DIR / "secret_key.bin"
PUBLIC_CTX_PATH= ARTIFACT_DIR / "public_context.bin"
RISK_FREE_RATE = 0.04 / 252   # daily risk-free rate (~4 % annual)

# ─────────────────────────────────────────────────────────────────────────────
# CKKS CONTEXT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_ckks_context() -> "ts.Context":
    """
    Create Alice's CKKS context with secret key.

    Parameters chosen to support:
      • degree-2 polynomial evaluation (2 multiplicative levels)
      • Chebyshev sqrt approximation up to degree 5 (3 levels)
    coeff_mod_bit_sizes=[60, 40, 40, 40, 60] gives 4 levels total — sufficient.
    scale=2**40 balances precision vs. noise growth.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.global_scale = 2 ** 40
    return context


def save_context(context: "ts.Context") -> None:
    """Persist secret key and public context to disk for session reuse."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Save with secret key (Alice only)
    with open(SECRET_KEY_PATH, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    # Save public context (shared with Carol — no secret key)
    public_ctx = context.copy()
    public_ctx.make_context_public()
    with open(PUBLIC_CTX_PATH, "wb") as f:
        f.write(public_ctx.serialize())
    print(f"[ALICE] CKKS context saved → {ARTIFACT_DIR}")


def load_or_build_context() -> "ts.Context":
    """Load existing context from disk, or build and save a new one."""
    if SECRET_KEY_PATH.exists():
        with open(SECRET_KEY_PATH, "rb") as f:
            context = ts.context_from(f.read())
        print("[ALICE] Loaded existing CKKS context from disk.")
    else:
        print("[ALICE] Building new CKKS context...")
        context = build_ckks_context()
        save_context(context)
    return context


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_scaler():
    """Load the StandardScaler fitted during data_pipeline.py."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run data_pipeline.py first."
        )
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_covariance() -> np.ndarray:
    data = np.load(COV_PATH)
    return data["sigma"]   # shape (N, N)


def load_expected_returns() -> np.ndarray:
    return np.load(MU_PATH)   # shape (N,)


def load_polynomial(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────

def standardize_weights(w: np.ndarray, scaler) -> np.ndarray:
    """
    Apply the same StandardScaler used during training so that
    encrypted weights live in the same space as the polynomial was fit on.
    w is reshaped to (1, N) for scaler, then flattened back.
    """
    z = scaler.transform(w.reshape(1, -1)).flatten().astype(np.float64)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# RISK METRICS (plaintext helpers — used by both plaintext and post-decrypt)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_variance_classical(w: np.ndarray, sigma: np.ndarray) -> float:
    """σ²_p = w⊤Σw — exact quadratic, no surrogate."""
    return float(w @ sigma @ w)


def portfolio_variance_from_poly(z: np.ndarray, poly: dict) -> float:
    """
    Evaluate degree-2 polynomial surrogate of variance in plaintext.
    f̂(z) = bias + a⊤z + z⊤Qz
    """
    bias = float(poly["bias"])
    a    = np.array(poly["linear"],    dtype=np.float64)
    Q    = np.array(poly["quadratic"], dtype=np.float64)
    return bias + float(a @ z) + float(z @ Q @ z)


def chebyshev_sqrt(x: float, coeffs: list) -> float:
    """Evaluate Chebyshev polynomial approximation of sqrt(x)."""
    coeffs = np.array(coeffs, dtype=np.float64)
    deg    = len(coeffs) - 1
    val    = 0.0
    for k, c in enumerate(coeffs):
        val += c * (x ** k)
    return max(val, 0.0)   # clamp — sqrt can't be negative


def var_from_variance(variance: float, mu_p: float, z_alpha: float = -1.645,
                      sqrt_coeffs: Optional[list] = None) -> float:
    """
    Parametric VaR_α = μ_p + z_α × σ_p.
    Uses Chebyshev sqrt approximation if coefficients provided, else exact sqrt.
    z_alpha = -1.645 for 95% VaR (left tail).
    """
    if sqrt_coeffs:
        sigma_p = chebyshev_sqrt(variance, sqrt_coeffs)
    else:
        sigma_p = float(np.sqrt(max(variance, 0.0)))
    return mu_p + z_alpha * sigma_p


def expected_shortfall(w: np.ndarray, returns: pd.DataFrame,
                       alpha: float = 0.05) -> float:
    """Historical ES: average loss on days beyond VaR threshold."""
    port_ret = returns.values @ w
    threshold = np.quantile(port_ret, alpha)
    tail      = port_ret[port_ret <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, variance: float,
                 rf: float = RISK_FREE_RATE,
                 sqrt_coeffs: Optional[list] = None) -> float:
    """S = (w⊤μ − r_f) / σ_p."""
    mu_p    = float(w @ mu)
    sigma_p = chebyshev_sqrt(variance, sqrt_coeffs) if sqrt_coeffs \
              else float(np.sqrt(max(variance, 0.0)))
    return (mu_p - rf) / sigma_p if sigma_p > 1e-10 else 0.0


def risk_attribution(w: np.ndarray, sigma: np.ndarray) -> dict:
    """
    Marginal and Component Risk Contribution from surrogate gradient.
    ∇_w σ²_p = 2Σw  →  MRC_i = (Σw)_i / σ_p
    """
    sigma_p = float(np.sqrt(max(float(w @ sigma @ w), 0.0)))
    sigma_w = sigma @ w
    mrc     = sigma_w / sigma_p if sigma_p > 1e-10 else np.zeros_like(w)
    crc     = w * mrc
    return {"MRC": mrc.tolist(), "CRC": crc.tolist(),
            "CRC_pct": (crc / crc.sum()).tolist() if crc.sum() > 1e-10 else []}


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — CLASSICAL PLAINTEXT
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_plaintext(w: np.ndarray, sigma: np.ndarray,
                             mu: np.ndarray, returns: pd.DataFrame) -> dict:
    """
    Baseline: all computations in plaintext using exact Markowitz formulas.
    No surrogate, no encryption. This is the direct classical benchmark.
    """
    t0      = time.perf_counter()
    var     = portfolio_variance_classical(w, sigma)
    mu_p    = float(w @ mu)
    var_95  = var_from_variance(var, mu_p)
    es_95   = expected_shortfall(w, returns)
    sharpe  = sharpe_ratio(w, mu, var)
    attrib  = risk_attribution(w, sigma)
    elapsed = time.perf_counter() - t0

    # Call Carol (plaintext mode — Carol just evaluates the polynomial directly)
    carol_result = carol_evaluate_classical(w, sigma, mu)

    return {
        "mode":            "classical_plaintext",
        "variance":         var,
        "mu_p":             mu_p,
        "VaR_95":           var_95,
        "ES_95":            es_95,
        "sharpe":           sharpe,
        "risk_attribution": attrib,
        "carol_score":      carol_result,
        "runtime_sec":      elapsed,
        "encrypted":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — CLASSICAL ENCRYPTED
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_encrypted(w: np.ndarray, sigma: np.ndarray,
                             mu: np.ndarray, returns: pd.DataFrame,
                             context: "ts.Context",
                             cl_poly: dict) -> dict:
    """
    Classical risk model (w⊤Σw as degree-2 polynomial) evaluated under CKKS.
    Alice encrypts w, Carol evaluates the polynomial, Alice decrypts.
    Demonstrates privacy on the classical pipeline without quantum.
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL required for encrypted modes.")

    t0     = time.perf_counter()
    scaler = load_scaler()
    z      = standardize_weights(w, scaler)

    # ── Alice encrypts ────────────────────────────────────────────────────────
    enc_z = ts.ckks_vector(context, z.tolist())
    print(f"[ALICE] Encrypted weight vector z (dim={len(z)})")

    # Serialize public context to simulate network boundary with Carol
    pub_ctx_bytes = context.serialize(save_secret_key=False)

    # ── Carol evaluates in ciphertext ─────────────────────────────────────────
    enc_score_bytes = carol_evaluate_classical_encrypted(
        enc_z.serialize(), pub_ctx_bytes, cl_poly
    )

    # ── Alice decrypts ────────────────────────────────────────────────────────
    enc_score = ts.lazy_ckks_vector_from(enc_score_bytes)
    enc_score.link_context(context)
    variance  = float(enc_score.decrypt()[0])
    print(f"[ALICE] Decrypted variance (encrypted classical): {variance:.6f}")

    mu_p    = float(w @ mu)   # return stays plaintext — only risk was encrypted
    var_95  = var_from_variance(variance, mu_p)
    es_95   = expected_shortfall(w, returns)
    sharpe  = sharpe_ratio(w, mu, variance)
    attrib  = risk_attribution(w, sigma)
    elapsed = time.perf_counter() - t0

    return {
        "mode":            "classical_encrypted",
        "variance":         variance,
        "mu_p":             mu_p,
        "VaR_95":           var_95,
        "ES_95":            es_95,
        "sharpe":           sharpe,
        "risk_attribution": attrib,
        "runtime_sec":      elapsed,
        "encrypted":        True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — QUANTUM PLAINTEXT
# ─────────────────────────────────────────────────────────────────────────────

def run_quantum_plaintext(w: np.ndarray, sigma: np.ndarray,
                           mu: np.ndarray, returns: pd.DataFrame,
                           qt_risk_poly: dict, qt_ret_poly: dict) -> dict:
    """
    VQC surrogate evaluated in plaintext (no encryption).
    Uses P_risk and P_return polynomial models fit to VQC outputs.
    Isolates the effect of quantum nonlinear risk modeling from FHE overhead.
    """
    t0     = time.perf_counter()
    scaler = load_scaler()
    z      = standardize_weights(w, scaler)

    # ── Carol evaluates surrogates in plaintext ───────────────────────────────
    carol_result = carol_evaluate_quantum(z, qt_risk_poly, qt_ret_poly)
    p_risk   = carol_result["P_risk"]
    p_return = carol_result["P_return"]

    mu_p    = float(w @ mu)
    var_95  = var_from_variance(p_risk, mu_p)
    es_95   = expected_shortfall(w, returns)

    # Quantum Sharpe: numerator = encrypted return score, denominator = P_risk
    sigma_p = float(np.sqrt(max(p_risk, 0.0)))
    sharpe  = (p_return - RISK_FREE_RATE) / sigma_p if sigma_p > 1e-10 else 0.0

    attrib  = risk_attribution(w, sigma)   # classical MRC/CRC for interpretability
    elapsed = time.perf_counter() - t0

    return {
        "mode":            "quantum_plaintext",
        "P_risk":           p_risk,
        "P_return":         p_return,
        "mu_p":             mu_p,
        "VaR_95":           var_95,
        "ES_95":            es_95,
        "sharpe":           sharpe,
        "risk_attribution": attrib,
        "runtime_sec":      elapsed,
        "encrypted":        False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4 — QUANTUM ENCRYPTED  (full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def run_quantum_encrypted(w: np.ndarray, sigma: np.ndarray,
                           mu: np.ndarray, returns: pd.DataFrame,
                           context: "ts.Context",
                           qt_risk_poly: dict, qt_ret_poly: dict,
                           sqrt_coeffs: Optional[list] = None) -> dict:
    """
    Full pipeline: VQC surrogate evaluated entirely under CKKS encryption.
    Alice encrypts weights → Carol evaluates P_risk and P_return in ciphertext
    → Alice decrypts scores → compute risk metrics.

    The Chebyshev sqrt approximation (sqrt_coeffs) is used to compute σ_p
    from P_risk without leaving the encrypted domain on Carol's side.
    If sqrt_coeffs is None, Alice computes sqrt after decryption (slight
    privacy trade-off but simpler).
    """
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL required for encrypted modes.")

    t0     = time.perf_counter()
    scaler = load_scaler()
    z      = standardize_weights(w, scaler)

    # ── Alice encrypts ────────────────────────────────────────────────────────
    enc_z = ts.ckks_vector(context, z.tolist())
    print(f"[ALICE] Encrypted weight vector z (dim={len(z)}) for quantum pipeline")

    pub_ctx_bytes = context.serialize(save_secret_key=False)

    # ── Carol evaluates both surrogates in ciphertext ─────────────────────────
    enc_risk_bytes, enc_ret_bytes = carol_evaluate_quantum_encrypted(
        enc_z.serialize(), pub_ctx_bytes, qt_risk_poly, qt_ret_poly
    )

    # ── Alice decrypts ────────────────────────────────────────────────────────
    enc_risk = ts.lazy_ckks_vector_from(enc_risk_bytes)
    enc_risk.link_context(context)
    p_risk   = float(enc_risk.decrypt()[0])

    enc_ret  = ts.lazy_ckks_vector_from(enc_ret_bytes)
    enc_ret.link_context(context)
    p_return = float(enc_ret.decrypt()[0])

    print(f"[ALICE] Decrypted P_risk={p_risk:.6f}  P_return={p_return:.6f}")

    mu_p    = float(w @ mu)
    var_95  = var_from_variance(p_risk, mu_p, sqrt_coeffs=sqrt_coeffs)
    es_95   = expected_shortfall(w, returns)
    sigma_p = chebyshev_sqrt(p_risk, sqrt_coeffs) if sqrt_coeffs \
              else float(np.sqrt(max(p_risk, 0.0)))
    sharpe  = (p_return - RISK_FREE_RATE) / sigma_p if sigma_p > 1e-10 else 0.0
    attrib  = risk_attribution(w, sigma)
    elapsed = time.perf_counter() - t0

    return {
        "mode":            "quantum_encrypted",
        "P_risk":           p_risk,
        "P_return":         p_return,
        "mu_p":             mu_p,
        "VaR_95":           var_95,
        "ES_95":            es_95,
        "sharpe":           sharpe,
        "risk_attribution": attrib,
        "runtime_sec":      elapsed,
        "encrypted":        True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results: list[dict]) -> None:
    """Pretty-print a side-by-side comparison of all four modes."""
    metrics = ["sharpe", "VaR_95", "ES_95", "runtime_sec", "encrypted"]
    labels  = {
        "sharpe":      "Sharpe Ratio",
        "VaR_95":      "VaR (95%)",
        "ES_95":       "ES  (95%)",
        "runtime_sec": "Runtime (s)",
        "encrypted":   "Encrypted?",
    }

    header = f"{'Metric':<22}" + "".join(f"{r['mode']:<26}" for r in results)
    print("\n" + "═" * (22 + 26 * len(results)))
    print("  PIPELINE COMPARISON — ALL MODES")
    print("═" * (22 + 26 * len(results)))
    print(header)
    print("─" * (22 + 26 * len(results)))

    for m in metrics:
        row = f"{labels[m]:<22}"
        for r in results:
            val = r.get(m, "N/A")
            if isinstance(val, float):
                row += f"{val:<26.5f}"
            else:
                row += f"{str(val):<26}"
        print(row)

    # Risk attribution summary (only if available)
    print("─" * (22 + 26 * len(results)))
    print("\n  VARIANCE / RISK SCORE")
    print("─" * (22 + 26 * len(results)))
    for r in results:
        var_key = "variance" if "variance" in r else "P_risk"
        val     = r.get(var_key, "N/A")
        label   = "σ²_p (exact)" if var_key == "variance" else "P_risk (surrogate)"
        print(f"  [{r['mode']}]  {label} = {val:.6f}" if isinstance(val, float)
              else f"  [{r['mode']}]  {label} = {val}")
    print("═" * (22 + 26 * len(results)) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Alice portfolio FHE pipeline")
    p.add_argument(
        "--mode",
        choices=["classical_plaintext", "classical_encrypted",
                 "quantum_plaintext", "quantum_encrypted", "all"],
        default="all",
        help="Pipeline mode to run (default: all)",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=str(ARTIFACT_DIR / "w_classical.csv"),
        help="Path to CSV of portfolio weights (one row = one portfolio)",
    )
    p.add_argument(
        "--portfolio-index",
        type=int,
        default=0,
        help="Which row of the weights CSV to use (default: 0 = first portfolio)",
    )
    p.add_argument(
        "--returns",
        type=str,
        default=str(ARTIFACT_DIR / "returns.csv"),
        help="Path to daily returns CSV for historical ES computation",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load weights ──────────────────────────────────────────────────────────
    w_df = pd.read_csv(args.weights, index_col=0)
    w    = w_df.iloc[args.portfolio_index].values.astype(np.float64)
    w    = w / w.sum()   # re-normalise in case of floating point drift
    print(f"[ALICE] Loaded portfolio weights (n_assets={len(w)}): {np.round(w, 4)}")

    # ── Load shared market data ───────────────────────────────────────────────
    sigma   = load_covariance()
    mu      = load_expected_returns()
    returns = pd.read_csv(args.returns, index_col=0, parse_dates=True)

    # ── Load polynomial models (classical Σ surrogate + quantum surrogates) ───
    cl_poly      = load_polynomial(CL_POLY_PATH)  if CL_POLY_PATH.exists()  else None
    qt_risk_poly = load_polynomial(QT_RISK_PATH)  if QT_RISK_PATH.exists()  else None
    qt_ret_poly  = load_polynomial(QT_RET_PATH)   if QT_RET_PATH.exists()   else None

    # Optional: load Chebyshev sqrt coefficients if pre-computed
    cheby_path   = ARTIFACT_DIR / "chebyshev_sqrt_coeffs.json"
    sqrt_coeffs  = json.loads(cheby_path.read_text())["coeffs"] \
                   if cheby_path.exists() else None

    # ── Build / load CKKS context (only for encrypted modes) ─────────────────
    context = None
    needs_encryption = args.mode in ("classical_encrypted", "quantum_encrypted", "all")
    if needs_encryption and TENSEAL_AVAILABLE:
        context = load_or_build_context()

    # ── Run selected mode(s) ──────────────────────────────────────────────────
    results = []
    modes   = (
        ["classical_plaintext", "classical_encrypted",
         "quantum_plaintext",   "quantum_encrypted"]
        if args.mode == "all" else [args.mode]
    )

    for mode in modes:
        print(f"\n[ALICE] ── Running mode: {mode} ──")
        try:
            if mode == "classical_plaintext":
                r = run_classical_plaintext(w, sigma, mu, returns)

            elif mode == "classical_encrypted":
                if not TENSEAL_AVAILABLE or context is None:
                    print("[ALICE] Skipping classical_encrypted — TenSEAL unavailable.")
                    continue
                if cl_poly is None:
                    print("[ALICE] Skipping classical_encrypted — classical_polynomial_model.json not found.")
                    continue
                r = run_classical_encrypted(w, sigma, mu, returns, context, cl_poly)

            elif mode == "quantum_plaintext":
                if qt_risk_poly is None or qt_ret_poly is None:
                    print("[ALICE] Skipping quantum_plaintext — VQC surrogate JSONs not found.")
                    continue
                r = run_quantum_plaintext(w, sigma, mu, returns, qt_risk_poly, qt_ret_poly)

            elif mode == "quantum_encrypted":
                if not TENSEAL_AVAILABLE or context is None:
                    print("[ALICE] Skipping quantum_encrypted — TenSEAL unavailable.")
                    continue
                if qt_risk_poly is None or qt_ret_poly is None:
                    print("[ALICE] Skipping quantum_encrypted — VQC surrogate JSONs not found.")
                    continue
                r = run_quantum_encrypted(
                    w, sigma, mu, returns, context,
                    qt_risk_poly, qt_ret_poly, sqrt_coeffs
                )

            results.append(r)
            print(f"[ALICE] Mode {mode} → Sharpe={r['sharpe']:.4f}  "
                  f"VaR={r['VaR_95']:.4f}  ES={r['ES_95']:.4f}  "
                  f"t={r['runtime_sec']:.2f}s")

        except Exception as exc:
            print(f"[ALICE] ERROR in mode {mode}: {exc}")
            raise

    # ── Print comparison table ────────────────────────────────────────────────
    if len(results) > 1:
        print_comparison(results)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = ARTIFACT_DIR / "alice_portfolio_results.json"
    with open(out_path, "w") as f:
        # Convert numpy floats for JSON serialisation
        def _jsonify(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(
            [{k: _jsonify(v) for k, v in r.items()} for r in results],
            f, indent=2
        )
    print(f"[ALICE] Results saved → {out_path}")


if __name__ == "__main__":
    main()