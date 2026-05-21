"""Clean generated files from local repository.

Removes logs, MLflow runs, predictions, and Python cache files.
Does NOT delete data, models, or reports unless --all is passed.

Usage:
    python scripts/clean.py          # safe clean (logs, cache, predictions)
    python scripts/clean.py --all    # full clean (also data/processed, models, reports)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAFE_TARGETS = [
    ROOT / "logs",
    ROOT / "mlruns",
    ROOT / "predictions",
    ROOT / ".pytest_cache",
    ROOT / ".mypy_cache",
    ROOT / ".ruff_cache",
]

ALL_TARGETS = SAFE_TARGETS + [
    ROOT / "data" / "processed",
    ROOT / "models",
    ROOT / "reports" / "figures",
    ROOT / "reports" / "trade_performance.txt",
    ROOT / "reports" / "next_day_ensemble_prediction.txt",
    ROOT / "reports" / "ensemble_regimes_explanation.txt",
    ROOT / "reports" / "baseline_results.md",
]


def remove(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  removed  {path.relative_to(ROOT)}")
    else:
        print(f"  skipped  {path.relative_to(ROOT)} (not found)")


def clean_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        shutil.rmtree(p)
    print("  removed  **/__pycache__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated repository files")
    parser.add_argument("--all", action="store_true", help="Also remove data/processed, models, and reports")
    args = parser.parse_args()

    targets = ALL_TARGETS if args.all else SAFE_TARGETS
    label = "full" if args.all else "safe"

    print(f"\nRunning {label} clean...\n")
    for t in targets:
        remove(t)
    clean_pycache(ROOT)
    print("\nDone.")
    if not args.all:
        print("Tip: run with --all to also remove data/processed, models, and reports")


if __name__ == "__main__":
    main()
