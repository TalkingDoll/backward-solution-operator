"""Redraw the Lorenz-96 data-length figures from the stored sweep results.

``exp_lorenz96_data_length`` trains 40 (N, seed) operators before it plots. When
only the presentation changes, that work is already done and saved: this entry
point loads ``l96_data_length_results.npz`` and calls the same plotting
functions, so the curves are provably the recorded ones.

Usage:
    python -m experiments.lorenz96.replot_lorenz96_data_length
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from experiments._data_length import load_result, plot_two_versions
from project_paths import FIGURES_DIR, RESULTS_DIR

TITLE_SUFFIX = "(K=40, F=8)"


def main() -> None:
    results = RESULTS_DIR / "l96_data_length" / "l96_data_length_results.npz"
    if not results.exists():
        raise SystemExit(
            f"{results} not found; run experiments.lorenz96."
            "exp_lorenz96_data_length first")
    result = load_result(results)
    out_figures = FIGURES_DIR / "lorenz96"
    written = []
    # Each figure twice: with the 25-75 percentile bands, and medians only.
    # Valid-prediction-time is deliberately not plotted for Lorenz-96: at
    # tau=0.02 every method's horizon collapses into {0.00 ... 0.12}, six steps
    # of resolution, which cannot separate them.
    for bands in (True, False):
        written.extend(plot_two_versions(result, out_figures, "Lorenz-96",
                                         title_suffix=TITLE_SUFFIX,
                                         bands=bands))
    for path in written:
        print(f"[lorenz96:data-length] replotted: {path}")


if __name__ == "__main__":
    main()
