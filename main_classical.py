"""
main_classical.py
=================
Sequential classical pipeline. Runs steps 1-8 in order and exits.
Called internally by run_platform.py, or directly for fine-grained control.

For the full platform (engines, dashboard, one command):
    python3 run_platform.py

Direct usage:
    python3 main_classical.py                # steps 1-8 (classical + FHE)
    python3 main_classical.py --skip-data    # skip data fetch
    python3 main_classical.py --skip-fhe     # steps 1-5 only
    python3 main_classical.py --fhe-only     # steps 6-8 only
"""

import os, sys, time, subprocess, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

HERE      = Path(__file__).parent
ARTIFACTS = HERE / "portfolio_training" / "artifacts"
PYTHON    = sys.executable


def banner(title):
    print("\n╔" + "═"*58 + "╗")
    print(f"║  {title:<56}║")
    print("╚" + "═"*58 + "╝")

def _ok(msg):   print(f"  ✓ {msg}")
def _warn(msg): print(f"  ⚠ {msg}")
def _info(msg): print(f"  → {msg}")


def _run(script: Path, label: str, *extra, required: bool = True) -> bool:
    """Run a script as a subprocess. Returns True on success."""
    cmd = [PYTHON, str(script)] + list(extra)
    _info(str(script.relative_to(HERE)))
    res = subprocess.run(cmd, cwd=str(HERE))
    if res.returncode == 0:
        return True
    fn = (_warn if not required else _warn)
    fn(f"{label} failed (exit {res.returncode})")
    return res.returncode == 0


def run_pipeline(skip_data=False, skip_fhe=False):
    t0 = time.time()

    # ── Steps 1-5: Classical ──────────────────────────────────────────────────
    if not skip_data:
        banner("STEP 1 — Data Pipeline")
        from portfolio_training.data_pipeline import run_pipeline as _dp
        _dp()
    else:
        _info("Step 1 skipped — using cached data")

    banner("STEP 2 — Classical Factor Model")
    from portfolio_training.classical_model import run_factor_model
    run_factor_model()

    banner("STEP 3 — Risk Metrics (VaR / ES / Sqrt)")
    from portfolio_training.risk_metrics import run_risk_metrics
    run_risk_metrics()

    banner("STEP 4 — Markowitz Optimizer")
    from portfolio_training.classical_optimizer import run_optimizer
    run_optimizer()

    banner("STEP 5 — Classical Evaluation & Backtest")
    from portfolio_training.evaluate_classical import run_evaluation
    metrics = run_evaluation()

    if skip_fhe:
        _ok(f"Classical pipeline done in {time.time()-t0:.1f}s")
        return metrics

    # ── Steps 6-8: FHE ───────────────────────────────────────────────────────
    banner("STEP 6 — FHE Polynomial Bridge")
    ok6 = _run(HERE / "build_classical_polynomial.py", "FHE polynomial", required=True)
    if not ok6:
        _warn("FHE polynomial failed — skipping steps 7 and 8")
        return metrics

    banner("STEP 7 — FHE Quick Accuracy Check")
    _run(
        HERE / "run_fhe_comparison.py",
        "FHE comparison",
        "--weights", str(ARTIFACTS / "w_classical.csv"),
        "--returns", str(ARTIFACTS / "returns.csv"),
        "--cov",     str(ARTIFACTS / "covariance.npz"),
        "--mu",      str(ARTIFACTS / "expected_returns.npy"),
        "--scaler",  str(ARTIFACTS / "scaler.pkl"),
        "--output",  str(ARTIFACTS / "fhe_comparison.json"),
        "--portfolio", "tangency",
        required=False,
    )

    banner("STEP 8 — Full Encrypted Classical Pipeline")
    _run(
        HERE / "encrypted_classical.py",
        "Encrypted classical",
        "--all-portfolios",
        required=False,
    )

    _ok(f"Full pipeline done in {time.time()-t0:.1f}s")
    return metrics


def run_fhe_only():
    """Steps 6-8 only — classical artifacts must already exist."""
    banner("FHE BRIDGE — Steps 6-8")
    ok6 = _run(HERE / "build_classical_polynomial.py", "FHE polynomial", required=True)
    if not ok6:
        _warn("Polynomial build failed — cannot run steps 7 or 8")
        return
    _run(
        HERE / "run_fhe_comparison.py", "FHE comparison",
        "--weights", str(ARTIFACTS / "w_classical.csv"),
        "--returns", str(ARTIFACTS / "returns.csv"),
        "--cov",     str(ARTIFACTS / "covariance.npz"),
        "--mu",      str(ARTIFACTS / "expected_returns.npy"),
        "--scaler",  str(ARTIFACTS / "scaler.pkl"),
        "--output",  str(ARTIFACTS / "fhe_comparison.json"),
        "--portfolio", "tangency",
        required=False,
    )
    _run(HERE / "encrypted_classical.py", "Encrypted classical",
         "--all-portfolios", required=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classical portfolio pipeline (steps 1-8)")
    p.add_argument("--skip-data",  action="store_true", help="Skip data fetch")
    p.add_argument("--skip-fhe",   action="store_true", help="Steps 1-5 only")
    p.add_argument("--fhe-only",   action="store_true", help="Steps 6-8 only")
    p.add_argument("--no-dashboard", action="store_true", help="(ignored, kept for compat)")
    args = p.parse_args()

    if args.fhe_only:
        run_fhe_only()
    else:
        run_pipeline(skip_data=args.skip_data, skip_fhe=args.skip_fhe)