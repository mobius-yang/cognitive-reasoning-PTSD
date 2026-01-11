from __future__ import annotations

"""Command-line entrypoint.

The implementation lives in `src/modeling/` to keep the pipeline modular.
This file provides a small CLI so running the project is straightforward.
"""

import argparse
import warnings

import os
import tempfile

from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt

if __package__ is None:
    # Allows `python src/model.py` from the project root.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.modeling.pipeline import main  # type: ignore
else:
    from .modeling.pipeline import main


# Keep plotting consistent on machines with Chinese fonts installed.
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modeling pipeline")
    parser.add_argument(
        "--task1",
        default="data/task1_scored.xlsx",
        help="Path to task1 scored Excel (default: data/task1_scored.xlsx)",
    )
    parser.add_argument(
        "--task2",
        default="data/task2_processing_data.xlsx",
        help="Path to task2 processed Excel (default: data/task2_processing_data.xlsx)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write metrics/plots (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        task1_path=args.task1,
        task2_path=args.task2,
        results_dir=Path(args.results_dir),
        random_state=args.seed,
    )
