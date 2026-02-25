"""
Orchestrator: run the full analysis pipeline in order.

Usage:
    python run_all.py          # run everything
    python run_all.py --skip-parse   # skip parsing (if CSVs already exist)
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _run_step(name: str, module_name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")
    t0 = time.time()
    mod = __import__(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  [{name}] completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parse_sessions step (use existing CSVs)")
    args = parser.parse_args()

    print("=" * 60)
    print("  EVAL DATA ANALYSIS PIPELINE")
    print("=" * 60)

    if not args.skip_parse:
        _run_step("1. Parse Sessions", "parse_sessions")
    else:
        print("\n  Skipping parse step (--skip-parse)")

    _run_step("2. Compute Metrics", "compute_metrics")
    _run_step("3. Statistical Tests", "statistical_tests")
    _run_step("4. Generate Figures", "generate_figures")
    _run_step("5. GP vs Custom Deep-Dive", "gp_vs_custom_analysis")
    _run_step("6. Implicit vs Explicit Preferences", "implicit_vs_explicit")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print(f"\n  Outputs: {SCRIPT_DIR / 'outputs'}")
    print(f"  Figures: {SCRIPT_DIR / 'figures'}")


if __name__ == "__main__":
    main()
