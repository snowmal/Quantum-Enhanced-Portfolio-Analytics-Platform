"""
build_classical_polynomial.py
==============================
Bridges the classical pipeline to the CKKS encrypted pipeline.

Takes the covariance matrix Σ produced by classical_model.py and
derives the exact degree-2 polynomial representation of w⊤Σw in
standardized weight space. Saves it as classical_polynomial_model.json
so that carol_portfolio_listener.py can evaluate it in ciphertext.

Also fits and saves the StandardScaler over the efficient frontier
weight vectors so Alice can standardize any weight vector consistently
before encryption.

Why this is a separate step
---------------------------
classical_model.py and classical_optimizer.py don't know about FHE.
This file is the explicit hand-off point: it reads classical artifacts,
derives FHE-compatible representations, and writes them to artifacts/.
That separation keeps each module's responsibility clean.

Inputs  (from artifacts/)
--------------------------
  sigma_full.npy         — full-sample covariance matrix (N×N)
  w_classical.csv        — Markowitz optimal weights (tickers × portfolios)
  mu_annual.npy          — annualised expected returns (N,)
  sqrt_approx_coeffs.json— Chebyshev sqrt coefficients from risk_metrics.py

Outputs (to artifacts/)
--------------------------
  classical_polynomial_model.json  — degree-2 poly for w⊤Σw in z-space
  scaler.pkl                       — StandardScaler fitted on frontier weights
  expected_returns.npy             — daily μ vector (N,)  [alias for FHE scripts]
  covariance.npz                   — Σ saved as .npz      [alias for FHE scripts]
  chebyshev_sqrt_coeffs.json       — copy of sqrt coeffs  [alias for FHE scripts]

Usage
-----
  python build_classical_polynomial.py

  # or called from main_classical.py as Step 6
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "portfolio_training", "artifacts")

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed — scaler will not be saved.", RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(filename, loader="npy"):
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            f"Run main_classical.py first to generate all classical artifacts."
        )
    if loader == "npy":
        return np.load(path)
    if loader == "npz":
        return np.load(path)
    if loader == "csv":
        return pd.read_csv(path, index_col=0)
    if loader == "json":
        with open(path) as f:
            return json.load(f)
    raise ValueError(f"Unknown loader: {loader}")


def _save_json(obj, filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved: {filename}")


def _save_pkl(obj, filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FIT SCALER ON FRONTIER WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_scaler(w_df: pd.DataFrame) -> object:
    """
    Fit a StandardScaler over all weight vectors in the efficient frontier.

    The frontier covers the full range of meaningful portfolio allocations,
    so the scaler captures the natural spread of each asset's weight.
    This is the same scaler Alice uses to standardize w → z before encryption,
    and that Carol's polynomial was calibrated on.

    Falls back to identity transform (mean=0, scale=1) if sklearn unavailable.
    """
    # w_df columns are portfolios (tangency, min_variance, equal_weight)
    # each row is an asset. Transpose so rows = portfolios, cols = assets.
    W = w_df.T.values.astype(np.float64)   # shape: (n_portfolios, n_assets)

    if not HAS_SKLEARN:
        print("  [WARN] sklearn not available — saving identity scaler dict.")
        n = W.shape[1]
        scaler_dict = {
            "mean_":  [0.0] * n,
            "scale_": [1.0] * n,
            "type":   "identity",
        }
        _save_json(scaler_dict, "scaler_params.json")
        return None

    # Augment with small random perturbations so scaler is robust to
    # portfolios not on the frontier (avoids zero variance for corner assets).
    # Tile W to (200, n_assets) by repeating the frontier rows, then add noise
    # so every augmented row is a perturbed version of an existing portfolio.
    rng        = np.random.default_rng(42)
    n_aug      = 200
    n_assets   = W.shape[1]
    # Repeat frontier rows to fill n_aug rows (round-robin)
    W_tiled    = np.tile(W, (n_aug // len(W) + 1, 1))[:n_aug]   # (200, n_assets)
    noise      = rng.normal(0, 0.005, (n_aug, n_assets))
    W_aug      = np.vstack([W, W_tiled + noise])                  # (3+200, n_assets)
    W_aug      = np.clip(W_aug, 0, 1)
    W_aug      = W_aug / W_aug.sum(axis=1, keepdims=True)         # re-normalise rows

    scaler = StandardScaler()
    scaler.fit(W_aug)

    print(f"  Scaler fitted on {len(W_aug)} weight vectors (frontier + augmented)")
    print(f"  Weight means  : {np.round(scaler.mean_, 4)}")
    print(f"  Weight scales : {np.round(scaler.scale_, 4)}")

    _save_pkl(scaler, "scaler.pkl")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DERIVE POLYNOMIAL FOR w⊤Σw IN STANDARDIZED SPACE
# ─────────────────────────────────────────────────────────────────────────────

def derive_classical_polynomial(sigma: np.ndarray, scaler) -> dict:
    """
    Derive the exact degree-2 polynomial for w⊤Σw after standardization.

    Standardization: z = (w - mean) / scale
    Inverse:         w = z * scale + mean

    Substituting into w⊤Σw:

        w⊤Σw = (z·s + m)⊤ Σ (z·s + m)
             = z⊤ (S·Σ·S) z  +  2·(Σ·m)⊤·(S·z)  +  m⊤·Σ·m

    where S = diag(scale), m = mean.

    So the polynomial is:
        f̂(z) = bias  +  a⊤z  +  z⊤Qz
        bias  = m⊤Σm
        a     = 2 · S · Σ · m      (absorbs scaler into linear term)
        Q     = S · Σ · S           (absorbs scaler into quadratic term)

    If no scaler is available (identity transform):
        bias = 0,  a = 0,  Q = Σ

    Validation: evaluate the polynomial at w=equal_weight and compare
    against the exact w⊤Σw to confirm the derivation is correct.
    """
    n = sigma.shape[0]

    if scaler is not None:
        s = np.array(scaler.scale_, dtype=np.float64)   # (n,)
        m = np.array(scaler.mean_,  dtype=np.float64)   # (n,)
        S = np.diag(s)

        Q    = S @ sigma @ S          # (n, n)  — quadratic coefficients
        a    = 2.0 * S @ sigma @ m    # (n,)    — linear coefficients
        bias = float(m @ sigma @ m)   # scalar  — constant term
    else:
        print("  [WARN] No scaler — polynomial uses raw weights (z = w).")
        Q    = sigma.copy()
        a    = np.zeros(n, dtype=np.float64)
        bias = 0.0

    poly = {
        "bias":      bias,
        "linear":    a.tolist(),
        "quadratic": Q.tolist(),
        "n":         n,
        "source":    "classical_w_sigma_w_standardized",
        "note":      "f(z) = bias + a'z + z'Qz  where z = StandardScaler(w)",
    }

    return poly


def validate_polynomial(poly: dict, sigma: np.ndarray,
                         scaler, n_assets: int) -> None:
    """
    Cross-check: evaluate polynomial at equal-weight vector and compare
    against exact w⊤Σw to confirm the algebra is correct.
    """
    w_eq = np.ones(n_assets) / n_assets
    exact_var = float(w_eq @ sigma @ w_eq)

    if scaler is not None:
        z = scaler.transform(w_eq.reshape(1, -1)).flatten().astype(np.float64)
    else:
        z = w_eq.copy()

    bias = poly["bias"]
    a    = np.array(poly["linear"],    dtype=np.float64)
    Q    = np.array(poly["quadratic"], dtype=np.float64)
    poly_var = bias + float(a @ z) + float(z @ Q @ z)

    abs_err = abs(poly_var - exact_var)
    rel_err = abs_err / (abs(exact_var) + 1e-14) * 100

    print(f"\n  Validation — equal-weight portfolio:")
    print(f"    Exact w⊤Σw      : {exact_var:.10f}")
    print(f"    Polynomial f̂(z) : {poly_var:.10f}")
    print(f"    Absolute error  : {abs_err:.2e}")
    print(f"    Relative error  : {rel_err:.6f}%")

    if rel_err < 1e-6:
        print(f"    Status: ✓ PASS — polynomial derivation is exact")
    elif rel_err < 0.01:
        print(f"    Status: ✓ PASS — error within floating-point tolerance")
    else:
        print(f"    Status: ✗ FAIL — relative error {rel_err:.4f}% exceeds 0.01%")
        print(f"    Check scaler mean/scale or Sigma loading.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SAVE FHE-REQUIRED ALIASES
# ─────────────────────────────────────────────────────────────────────────────

def save_fhe_aliases(sigma: np.ndarray, mu_daily: np.ndarray,
                     sqrt_coeffs: dict) -> None:
    """
    alice_portfolio.py and run_fhe_comparison.py expect these specific
    filenames. Save them as aliases so the FHE scripts work without
    needing to know internal classical pipeline naming.
    """
    # expected_returns.npy — daily μ (alice_portfolio.py reads this)
    np.save(os.path.join(ARTIFACTS_DIR, "expected_returns.npy"), mu_daily)
    print(f"  Saved: expected_returns.npy  (daily μ, shape={mu_daily.shape})")

    # covariance.npz — Σ as .npz (alice_portfolio.py reads sigma=data['sigma'])
    np.savez(os.path.join(ARTIFACTS_DIR, "covariance.npz"), sigma=sigma)
    print(f"  Saved: covariance.npz  (Σ, shape={sigma.shape})")

    # chebyshev_sqrt_coeffs.json — copy of sqrt_approx_coeffs.json
    # alice_portfolio.py reads key "coeffs"; risk_metrics.py uses "chebyshev_coeffs"
    # so we reformat here for the FHE scripts
    alias_coeffs = {
        "coeffs":   sqrt_coeffs.get("chebyshev_coeffs", []),
        "degree":   sqrt_coeffs.get("degree", 5),
        "max_var":  sqrt_coeffs.get("max_var", 1.0),
        "rmse":     sqrt_coeffs.get("rmse", None),
        "max_error":sqrt_coeffs.get("max_error", None),
    }
    _save_json(alias_coeffs, "chebyshev_sqrt_coeffs.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_build_polynomial():
    print("\n" + "=" * 60)
    print("  BUILD CLASSICAL POLYNOMIAL  (FHE bridge)")
    print("=" * 60 + "\n")

    # ── Load classical artifacts ──────────────────────────────────────────────
    print("[POLY] Loading classical artifacts …")
    sigma    = _load("sigma_full.npy",   "npy")
    mu_ann   = _load("mu_annual.npy",    "npy")
    w_df     = _load("w_classical.csv",  "csv")
    n_assets = sigma.shape[0]
    tickers  = w_df.index.tolist()
    mu_daily = mu_ann / 252

    print(f"  Covariance Σ : {sigma.shape}")
    print(f"  μ (annual)   : shape={mu_ann.shape}  "
          f"range=[{mu_ann.min():.4f}, {mu_ann.max():.4f}]")
    print(f"  Portfolios   : {w_df.columns.tolist()}")
    print(f"  Assets       : {n_assets}  tickers: {tickers}")

    # Load sqrt coefficients (produced by risk_metrics.py)
    sqrt_coeffs_path = os.path.join(ARTIFACTS_DIR, "sqrt_approx_coeffs.json")
    if os.path.exists(sqrt_coeffs_path):
        with open(sqrt_coeffs_path) as f:
            sqrt_coeffs = json.load(f)
        print(f"  Sqrt coeffs  : degree={sqrt_coeffs.get('degree')}, "
              f"rmse={sqrt_coeffs.get('rmse', 'N/A'):.2e}"
              if sqrt_coeffs.get('rmse') else "  Sqrt coeffs  : loaded")
    else:
        print("  [WARN] sqrt_approx_coeffs.json not found — run risk_metrics.py first.")
        sqrt_coeffs = {"chebyshev_coeffs": [], "degree": 5, "max_var": float(sigma.max() * 10)}

    # ── Step 1: Fit and save scaler ───────────────────────────────────────────
    print("\n[POLY] Step 1 — Fitting StandardScaler on frontier weights …")
    scaler = fit_and_save_scaler(w_df)

    # ── Step 2: Derive polynomial ─────────────────────────────────────────────
    print("\n[POLY] Step 2 — Deriving degree-2 polynomial for w⊤Σw …")
    poly = derive_classical_polynomial(sigma, scaler)
    print(f"  bias  = {poly['bias']:.8f}")
    print(f"  |a|   = {np.linalg.norm(poly['linear']):.6f}  (linear norm)")
    print(f"  |Q|_F = {np.linalg.norm(poly['quadratic']):.6f}  (quadratic Frobenius norm)")

    # ── Step 3: Validate ─────────────────────────────────────────────────────
    print("\n[POLY] Step 3 — Validating polynomial against exact w⊤Σw …")
    validate_polynomial(poly, sigma, scaler, n_assets)

    # Validate on all three frontier portfolios
    print("\n  Cross-validation on all frontier portfolios:")
    for col in w_df.columns:
        w_vec    = w_df[col].values.astype(np.float64)
        exact    = float(w_vec @ sigma @ w_vec)
        if scaler is not None:
            z = scaler.transform(w_vec.reshape(1, -1)).flatten()
        else:
            z = w_vec.copy()
        a   = np.array(poly["linear"],    dtype=np.float64)
        Q   = np.array(poly["quadratic"], dtype=np.float64)
        est = poly["bias"] + float(a @ z) + float(z @ Q @ z)
        err = abs(est - exact) / (abs(exact) + 1e-14) * 100
        status = "✓" if err < 0.01 else "✗"
        print(f"    {col:20s}  exact={exact:.8f}  poly={est:.8f}  "
              f"err={err:.2e}%  {status}")

    # ── Step 4: Save polynomial ───────────────────────────────────────────────
    print("\n[POLY] Step 4 — Saving polynomial model …")
    _save_json(poly, "classical_polynomial_model.json")

    # ── Step 5: Save FHE aliases ──────────────────────────────────────────────
    print("\n[POLY] Step 5 — Saving FHE-script aliases …")
    save_fhe_aliases(sigma, mu_daily, sqrt_coeffs)

    print("\n" + "=" * 60)
    print("  BUILD CLASSICAL POLYNOMIAL COMPLETE")
    print("=" * 60)
    print(f"\n  Artifacts written to: portfolio_training/artifacts/")
    print(f"  ├── classical_polynomial_model.json  (Carol's polynomial)")
    print(f"  ├── scaler.pkl                        (Alice's weight scaler)")
    print(f"  ├── expected_returns.npy              (daily μ alias)")
    print(f"  ├── covariance.npz                    (Σ alias)")
    print(f"  └── chebyshev_sqrt_coeffs.json        (sqrt approx alias)")
    print(f"\n  Next step: python run_fhe_comparison.py")
    print(f"             or: python run_fhe_comparison.py --demo")

    return poly, scaler


if __name__ == "__main__":
    run_build_polynomial()