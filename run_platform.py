"""
run_platform.py
===============
Single entry point for the entire platform.

Fresh start (zero artifacts):
    python3 run_platform.py

That one command runs, in order:
    Step 1   data_pipeline.py              fetch market data
    Step 2   classical_model.py            factor model + covariance
    Step 3   risk_metrics.py               VaR / ES / Chebyshev sqrt
    Step 4   classical_optimizer.py        Markowitz efficient frontier
    Step 5   evaluate_classical.py         rolling backtest
    Step 6   build_classical_polynomial.py FHE polynomial + scaler
    Step 7   run_fhe_comparison.py         quick plaintext vs CKKS check
    Step 8   encrypted_classical.py        full encrypted pipeline (all portfolios)
    Step 9   live_optimizer.py --once      live weights snapshot
    Step 10  projection_engine.py          Monte Carlo projections
    Step 11  advisory_engine.py            rule-based advice
    Step 12  dashboard_server.py           Dash at http://localhost:8050

Common usage
------------
    # Fresh start — everything from zero:
    python3 run_platform.py

    # Data already downloaded — skip yfinance fetch:
    python3 run_platform.py --skip-data

    # Classical only (no FHE, no engines, no dashboard):
    python3 run_platform.py --skip-fhe --no-engines --no-dashboard

    # Pipeline + dashboard, no background engines:
    python3 run_platform.py --no-engines

    # Dashboard only (pipeline already ran):
    python3 run_platform.py --dashboard-only

    # Restart engines only (pipeline already ran):
    python3 run_platform.py --engines-only

    # HPC — parallel lambda scan, 50k MC paths:
    python3 run_platform.py --lambda-scan --mc-paths 50000 --n-jobs 32

    # Quantum mode (adds VQC once vqc_portfolio.py is built):
    python3 run_platform.py --mode quantum-fhe
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime

HERE      = Path(__file__).parent
ARTIFACTS = HERE / "portfolio_training" / "artifacts"
PYTHON    = sys.executable


# ── Colour helpers ─────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD  = "\033[1m"
    CYAN   = "\033[96m"; GREEN = "\033[92m"
    GOLD   = "\033[93m"; RED   = "\033[91m"
    VIOLET = "\033[95m"; TEAL  = "\033[36m"
    MUTED  = "\033[90m"

def banner(msg, col=C.CYAN):
    w = 60
    print(f"\n{col}{C.BOLD}╔{'═'*w}╗\n║  {msg:<{w-2}}║\n╚{'═'*w}╝{C.RESET}")

def ok(msg):    print(f"{C.GREEN}  ✓ {msg}{C.RESET}")
def warn(msg):  print(f"{C.GOLD}  ⚠ {msg}{C.RESET}")
def info(msg):  print(f"{C.MUTED}  → {msg}{C.RESET}")
def fail(msg):  print(f"{C.RED}  ✗ {msg}{C.RESET}")


# ── Subprocess step runner ─────────────────────────────────────────────────────
def run_step(label: str, *cmd_parts, required: bool = True) -> bool:
    """
    Run one pipeline step as a subprocess, printing a clear label.
    All scripts are referenced by absolute path so cwd never matters.
    Returns True on success, False on failure.
    If required=True and step fails, caller should abort the run.
    """
    cmd = [PYTHON] + [str(p) for p in cmd_parts]
    info(f"{label}")
    t0  = time.time()
    res = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.time() - t0
    if res.returncode == 0:
        ok(f"{label} ({elapsed:.1f}s)")
        return True
    fn = fail if required else warn
    fn(f"{label} failed (exit {res.returncode}, {elapsed:.1f}s)")
    return False


def artifact(*names: str) -> bool:
    """Return True only if every named artifact file exists."""
    return all((ARTIFACTS / n).exists() for n in names)


# ── Quantum module check ───────────────────────────────────────────────────────
def quantum_modules_exist() -> bool:
    needed = [
        HERE / "portfolio_training" / "vqc_portfolio.py",
        HERE / "portfolio_training" / "surrogate_portfolio.py",
    ]
    return all(p.exists() for p in needed)


# ── ENGINE MANAGER ─────────────────────────────────────────────────────────────
class EngineManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, name: str, cmd: list, prefix: str) -> None:
        def _stream(proc, pfx):
            for line in iter(proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    print(f"{C.MUTED}[{pfx}]{C.RESET} {decoded}")

        proc = subprocess.Popen(
            cmd, cwd=str(HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        self._procs[name] = proc
        threading.Thread(target=_stream, args=(proc, prefix), daemon=True).start()
        ok(f"Engine started: {name}  (PID {proc.pid})")

    def stop_all(self) -> None:
        for name, proc in self._procs.items():
            if proc.poll() is None:
                info(f"Stopping {name} …")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        ok("All engines stopped.")

    def all_alive(self) -> dict[str, bool]:
        return {n: p.poll() is None for n, p in self._procs.items()}


# ── PIPELINE ───────────────────────────────────────────────────────────────────
def run_pipeline(args) -> bool:
    """
    Run every pipeline step in sequence.
    Steps are run via subprocess so each script handles its own paths.
    Returns True if all required steps succeeded.
    """
    t0       = time.time()
    skip_fhe = args.skip_fhe
    quantum  = "quantum" in args.mode

    banner("CLASSICAL PIPELINE  (Steps 1–5)", C.GOLD)

    # ── Steps 1–5: Classical pipeline via main_classical.py ─────────────────
    # main_classical.py uses imports (not subprocesses) for the training
    # modules, which correctly handles sys.path for cross-module imports.
    mc_args = ["--skip-fhe", "--no-dashboard"]
    if args.skip_data and artifact("returns.csv", "sigma_full.npy", "mu_annual.npy"):
        info("Step 1 skipped — cached data found")
        mc_args.append("--skip-data")

    if not run_step(
        "Steps 1–5 / Classical Pipeline (data → model → risk → optimizer → backtest)",
        HERE / "main_classical.py",
        *mc_args,
        required=True,
    ):
        fail("Classical pipeline failed — cannot continue"); return False

    ok(f"Classical pipeline complete  ({time.time()-t0:.1f}s)")

    # ── Quantum steps (only if mode includes quantum AND modules exist) ────────
    if quantum:
        banner("QUANTUM PIPELINE  (Steps 6–7)", C.VIOLET)
        if not quantum_modules_exist():
            warn("Quantum modules not yet built — skipping VQC steps")
            info("Add portfolio_training/vqc_portfolio.py and surrogate_portfolio.py")
        else:
            run_step("Step 6 / VQC Training",
                     HERE / "main_classical.py",
                     "--quantum", "--skip-fhe", "--skip-data", "--no-dashboard",
                     required=False)

    # ── FHE steps ─────────────────────────────────────────────────────────────
    if not skip_fhe:
        banner("FHE PIPELINE  (Steps 6–8)", C.TEAL)

        # Step 6: FHE polynomial bridge
        if not run_step(
            "Step 6 / FHE Polynomial Bridge",
            HERE / "build_classical_polynomial.py",
            required=True,
        ):
            warn("FHE polynomial failed — skipping encrypted steps")
            return True   # classical is still valid

        # Step 7: Quick accuracy check
        run_step(
            "Step 7 / FHE Quick Accuracy Check",
            HERE / "run_fhe_comparison.py",
            "--weights", str(ARTIFACTS / "w_classical.csv"),
            "--returns", str(ARTIFACTS / "returns.csv"),
            "--cov",     str(ARTIFACTS / "covariance.npz"),
            "--mu",      str(ARTIFACTS / "expected_returns.npy"),
            "--scaler",  str(ARTIFACTS / "scaler.pkl"),
            "--output",     str(ARTIFACTS / "fhe_comparison.json"),
            "--portfolio",  "tangency",
            required=False,
        )

        # Step 8: Full encrypted classical pipeline
        run_step(
            "Step 8 / Full Encrypted Classical Pipeline (all portfolios)",
            HERE / "encrypted_classical.py",
            "--all-portfolios",
            required=False,
        )

    elapsed = time.time() - t0
    ok(f"Full pipeline complete in {elapsed:.1f}s")
    return True


# ── ENGINES ────────────────────────────────────────────────────────────────────
def run_engines_once(args) -> None:
    """
    Run the one-shot engines synchronously so their output files exist
    before the dashboard launches and before background engines start.
    Order matters: live → projections → advisory (each needs the previous).
    """
    banner("INITIAL ENGINE OUTPUTS", C.CYAN)

    run_step(
        "Step 9  / Live Optimizer (one cycle)",
        HERE / "live_optimizer.py", "--once",
        required=False,
    )
    run_step(
        "Step 10 / Projection Engine",
        HERE / "projection_engine.py",
        "--paths",   str(args.mc_paths),
        "--horizon", str(args.mc_horizon),
        "--n-jobs",  str(args.n_jobs),
        required=False,
    )
    run_step(
        "Step 11 / Advisory Engine",
        HERE / "advisory_engine.py",
        required=False,
    )


def start_background_engines(args) -> EngineManager:
    """
    Start live_optimizer and advisory_engine as persistent background processes.
    projection_engine reruns automatically via advisory's watch loop.
    """
    banner("BACKGROUND ENGINES", C.CYAN)
    mgr = EngineManager()

    live_cmd = [
        PYTHON, str(HERE / "live_optimizer.py"),
        "--interval", str(args.live_interval),
        "--lambda",   str(args.lam),
    ]
    if args.lambda_scan:
        live_cmd.append("--lambda-scan")
    if args.n_jobs != -1:
        live_cmd += ["--n-jobs", str(args.n_jobs)]
    mgr.start("live_optimizer", live_cmd, "LIVE")

    adv_cmd = [
        PYTHON, str(HERE / "advisory_engine.py"),
        "--watch", "--interval", str(args.advisory_interval),
    ]
    mgr.start("advisory_engine", adv_cmd, "ADV")

    return mgr


# ── DASHBOARD ──────────────────────────────────────────────────────────────────
def launch_dashboard() -> subprocess.Popen:
    banner("DASHBOARD", C.CYAN)
    print(f"  {C.CYAN}Open  http://localhost:8050  in your browser{C.RESET}\n")
    return subprocess.Popen(
        [PYTHON, str(HERE / "dashboard_server.py")],
        cwd=str(HERE),
    )


# ── STATUS REPORT ──────────────────────────────────────────────────────────────
def print_status(args, engines_live: bool) -> None:
    files = [
        ("returns.csv",                     "market data"),
        ("classical_metrics.json",          "backtest results"),
        ("classical_polynomial_model.json", "FHE polynomial"),
        ("scaler.pkl",                      "weight scaler"),
        ("fhe_comparison.json",             "FHE quick check"),
        ("encrypted_classical_results.json","full encrypted pipeline"),
        ("live_state.json",                 "live optimizer"),
        ("projections.json",                "MC projections"),
        ("advice.json",                     "advisory signals"),
    ]
    print(f"\n{C.BOLD}{'═'*62}{C.RESET}")
    print(f"{C.BOLD}  PLATFORM STATUS{C.RESET}")
    print(f"{'═'*62}")
    print(f"  Mode      : {args.mode}")
    print(f"  FHE       : {'enabled' if not args.skip_fhe else 'skipped'}")
    print(f"  Engines   : {'running' if engines_live else 'not started'}")
    print(f"  Dashboard : {'http://localhost:8050' if not args.no_dashboard else 'not started'}")
    print()
    for fname, desc in files:
        exists = (ARTIFACTS / fname).exists()
        mark   = f"{C.GREEN}✓{C.RESET}" if exists else f"{C.MUTED}○{C.RESET}"
        print(f"    {mark} {fname:<42} {C.MUTED}{desc}{C.RESET}")
    print(f"{'═'*62}\n")


# ── SHUTDOWN LOOP ──────────────────────────────────────────────────────────────
def wait_for_shutdown(mgr: EngineManager | None,
                      dash: subprocess.Popen | None) -> None:
    done = threading.Event()

    def _sig(s, f):
        print(f"\n{C.GOLD}  Ctrl+C — shutting down …{C.RESET}")
        done.set()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)
    print(f"\n{C.MUTED}  Running. Press Ctrl+C to stop all processes.{C.RESET}")

    last_check = time.time()
    while not done.is_set():
        time.sleep(1)
        if mgr and time.time() - last_check > 60:
            for name, alive in mgr.all_alive().items():
                if not alive:
                    warn(f"Engine '{name}' exited unexpectedly")
            last_check = time.time()
        if dash and dash.poll() is not None:
            done.set()

    if mgr:  mgr.stop_all()
    if dash and dash.poll() is None:
        dash.terminate()
        try:    dash.wait(timeout=5)
        except: dash.kill()
    ok("Shut down cleanly.")


# ── ARGS ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Portfolio Risk Platform — single command, full stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python3 run_platform.py                          # fresh start — everything
  python3 run_platform.py --skip-data             # data cached, skip fetch
  python3 run_platform.py --skip-fhe              # classical only
  python3 run_platform.py --no-engines            # pipeline + dashboard only
  python3 run_platform.py --no-dashboard          # pipeline + engines, headless
  python3 run_platform.py --dashboard-only        # open dashboard, skip pipeline
  python3 run_platform.py --engines-only          # restart engines, skip pipeline
  python3 run_platform.py --mode quantum-fhe      # quantum mode (when VQC ready)
  python3 run_platform.py --lambda-scan --mc-paths 50000 --n-jobs 32  # HPC
        """,
    )

    p.add_argument("--mode",
                   choices=["classical", "classical-fhe", "quantum", "quantum-fhe"],
                   default="classical-fhe")
    p.add_argument("--skip-data",      action="store_true",
                   help="Skip data fetch if returns.csv already exists")
    p.add_argument("--skip-fhe",       action="store_true",
                   help="Skip FHE steps 6–8")
    p.add_argument("--no-engines",     action="store_true",
                   help="Do not start any background engines")
    p.add_argument("--no-dashboard",   action="store_true",
                   help="Do not launch the Dash dashboard")
    p.add_argument("--dashboard-only", action="store_true",
                   help="Launch dashboard only — skip pipeline entirely")
    p.add_argument("--engines-only",   action="store_true",
                   help="Run engines only — skip pipeline")

    p.add_argument("--live-interval",     type=int,   default=300)
    p.add_argument("--lam",               type=float, default=0.5)
    p.add_argument("--lambda-scan",       action="store_true")
    p.add_argument("--mc-paths",          type=int,   default=10_000)
    p.add_argument("--mc-horizon",        type=int,   default=252)
    p.add_argument("--advisory-interval", type=int,   default=60)
    p.add_argument("--n-jobs",            type=int,   default=-1)

    return p.parse_args()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Dashboard only ────────────────────────────────────────────────────────
    if args.dashboard_only:
        dash = launch_dashboard()
        try:    dash.wait()
        except KeyboardInterrupt: dash.terminate()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    banner(f"PORTFOLIO RISK PLATFORM  ·  {args.mode.upper()}", C.CYAN)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode    : {args.mode}")
    print(f"  FHE     : {'skipped' if args.skip_fhe else 'enabled'}")
    print(f"  Engines : {'none' if args.no_engines else 'live + advisory (background)'}")
    print(f"  n_jobs  : {args.n_jobs}  |  MC paths: {args.mc_paths:,}")

    # ── Engines only ──────────────────────────────────────────────────────────
    if args.engines_only:
        run_engines_once(args)
        mgr  = None if args.no_engines else start_background_engines(args)
        dash = None if args.no_dashboard else launch_dashboard()
        print_status(args, engines_live=(mgr is not None))
        wait_for_shutdown(mgr, dash)
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_ok = run_pipeline(args)
    if not pipeline_ok:
        fail("Pipeline had errors — check output above")
        warn("Starting dashboard anyway with whatever artifacts exist")

    # ── Engines (one-shot sync first, then background) ────────────────────────
    mgr = None
    if not args.no_engines:
        run_engines_once(args)
        mgr = start_background_engines(args)
        time.sleep(1)

    # ── Status ────────────────────────────────────────────────────────────────
    print_status(args, engines_live=(mgr is not None))

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dash = None
    if not args.no_dashboard:
        dash = launch_dashboard()

    # ── Wait ──────────────────────────────────────────────────────────────────
    if mgr or dash:
        wait_for_shutdown(mgr, dash)


if __name__ == "__main__":
    main()