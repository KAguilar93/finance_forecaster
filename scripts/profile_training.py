"""Profile the model training entrypoint with cProfile.

This script supports Phase 2 Section 3: Profiling Python and Machine Learning Code.

It runs the default training pipeline through cProfile, saves the raw profiler
output, and writes a readable cumulative-time report for documentation.
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from finance_forecaster.train_model import main


PROFILE_DIR = Path("reports/profiling")
PROFILE_OUTPUT = PROFILE_DIR / "training_profile.out"
PROFILE_REPORT = PROFILE_DIR / "cprofile_results.txt"


def run_profile() -> None:
    """Run training under cProfile and save readable profiling results."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats(PROFILE_OUTPUT)

    with PROFILE_REPORT.open("w", encoding="utf-8") as report_file:
        stats = pstats.Stats(profiler, stream=report_file)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(50)

    print(f"Raw cProfile output saved to: {PROFILE_OUTPUT}")
    print(f"Readable profiling report saved to: {PROFILE_REPORT}")


if __name__ == "__main__":
    run_profile()