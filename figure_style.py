"""Publication-figure specification: one central style, explicit layouts.

Every visual element inside a subplot has the same *physical* size regardless of
the subplot grid, so a 1x2 and a 1x3 figure look like they belong to the same
paper. Overall figure dimensions may differ; fonts, lines, markers, ticks and
legend spacing may not.

Three rules this module exists to enforce, all of which Matplotlib gets wrong by
default:

* **No automatic layout.** ``tight_layout``, ``constrained_layout`` and
  ``bbox_inches="tight"`` all resize or crop the figure after the fact, which
  makes the exported size differ from the requested one and makes physical
  margins irreproducible. Margins here are given in *inches* and converted to
  ``subplots_adjust`` fractions by :func:`make_figure`.
* **``wspace``/``hspace`` are not portable between grids.** Matplotlib defines
  them as fractions of the *average axes size*, so the same value produces a
  different physical gap in a 1x2 than in a 1x3. :func:`make_figure` takes the
  gap in inches and solves for the fraction per grid.
* **Defaults are not inherited.** Font sizes, tick geometry, grid styling and
  export geometry are all written out explicitly in :data:`PUBLICATION_RC`.

Usage::

    with publication_style():
        fig, axes = make_figure(LAYOUT)
        ...
        save_publication_pair(fig, path)

This module is deliberately independent of ``plotting_utils.save_figure_pair``,
which the non-ENSO experiments still use and whose output must not change.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

# Widths mandated by the specification.
SINGLE_COLUMN_WIDTH = 3.4      # inches; include with width=\columnwidth
DOUBLE_COLUMN_WIDTH = 7.0      # inches; include with width=\textwidth
DPI = 300

TEXT_COLOR = "#111111"
GRID_COLOR = "#9CA3AF"
WHITE = "#FFFFFF"

# The three line classes the specification distinguishes. Encoding "which curve
# is the method" here keeps it out of per-figure judgement.
LINEWIDTH = {"main": 1.7, "data": 1.4, "reference": 1.0}
MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = 0.8


# ---------------------------------------------------------------------------
# 1. The central style dictionary
# ---------------------------------------------------------------------------

PUBLICATION_RC = {
    # Typography
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "font.weight": "normal",
    "mathtext.fontset": "dejavusans",
    "figure.titlesize": 10,
    "figure.titleweight": "normal",
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.titlepad": 5.0,
    "axes.labelsize": 9,
    "axes.labelpad": 4.0,
    "axes.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Lines and markers
    "lines.linewidth": LINEWIDTH["data"],
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": MARKER_EDGE_WIDTH,

    # Axes and ticks: outward, bottom and left only, all four spines kept.
    "axes.linewidth": 0.8,
    "axes.edgecolor": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.labelcolor": TEXT_COLOR,
    "ytick.labelcolor": TEXT_COLOR,
    "xtick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "xtick.major.pad": 2.0,
    "xtick.minor.size": 1.8,
    "xtick.minor.width": 0.6,
    "ytick.major.size": 3.0,
    "ytick.major.width": 0.8,
    "ytick.major.pad": 2.0,
    "ytick.minor.size": 1.8,
    "ytick.minor.width": 0.6,
    "xtick.top": False,
    "xtick.bottom": True,
    "ytick.right": False,
    "ytick.left": True,

    # Grid, always drawn below the data.
    "grid.linewidth": 0.6,
    "grid.alpha": 0.25,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "-",
    "axes.axisbelow": True,

    # Colours and background
    "text.color": TEXT_COLOR,
    "axes.facecolor": WHITE,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "savefig.edgecolor": WHITE,
    "savefig.transparent": False,

    # Export geometry: no automatic cropping, embedded TrueType fonts.
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Every legend spacing parameter the specification enumerates. Location, bbox
# anchor and column count are per figure and supplied by the caller.
LEGEND_KW = {
    "fontsize": 8,
    "frameon": False,
    "fancybox": False,
    "framealpha": 1.0,
    "borderaxespad": 0.3,
    "borderpad": 0.2,
    "labelspacing": 0.35,
    "handlelength": 2.2,
    "handleheight": 0.7,
    "handletextpad": 0.5,
    "columnspacing": 1.2,
}


@contextmanager
def publication_style():
    """Apply :data:`PUBLICATION_RC` for the duration of one figure."""
    with plt.rc_context(PUBLICATION_RC):
        yield


# ---------------------------------------------------------------------------
# 2. Inch-accurate layout
# ---------------------------------------------------------------------------

def _subplots_adjust(layout: dict) -> dict:
    """Convert inch margins and gaps into ``subplots_adjust`` fractions.

    ``wspace`` and ``hspace`` are fractions of the *average axes* width/height,
    not of the figure, so they are solved per grid from the requested physical
    gap. Reusing one value across different grids -- which the specification
    forbids -- would silently change the physical spacing.
    """
    width, height = layout["figsize"]
    nrows, ncols = int(layout["nrows"]), int(layout["ncols"])
    left, right = float(layout["left"]), float(layout["right"])
    bottom, top = float(layout["bottom"]), float(layout["top"])
    wgap, hgap = float(layout.get("wgap", 0.0)), float(layout.get("hgap", 0.0))

    panel_w = (width - left - right - (ncols - 1) * wgap) / ncols
    panel_h = (height - bottom - top - (nrows - 1) * hgap) / nrows
    if panel_w <= 0 or panel_h <= 0:
        raise ValueError(f"margins exceed the figure size: {layout['figsize']}")

    return {
        "left": left / width,
        "right": 1.0 - right / width,
        "bottom": bottom / height,
        "top": 1.0 - top / height,
        "wspace": (wgap / panel_w) if ncols > 1 else 0.0,
        "hspace": (hgap / panel_h) if nrows > 1 else 0.0,
    }


def make_figure(layout: dict):
    """``(fig, axes)`` sized and positioned entirely from ``layout``.

    ``layout`` must carry ``figsize``, ``nrows``, ``ncols``, the four margins and
    the inter-panel gaps, all in inches. No automatic layout engine is used, so
    the exported figure is exactly ``figsize``.
    """
    fig, axes = plt.subplots(
        int(layout["nrows"]), int(layout["ncols"]),
        figsize=tuple(layout["figsize"]), dpi=DPI, squeeze=False)
    fig.subplots_adjust(**_subplots_adjust(layout))
    return fig, axes


def style_axis(ax, *, xlabel=None, ylabel=None, title=None, grid=True):
    """Apply the label/title/grid conventions explicitly, never by default.

    The specification requires the grid to be enabled or disabled deliberately
    on every subplot, so ``grid`` has no implicit case.
    """
    if title is not None:
        ax.set_title(title, fontsize=9, fontweight="normal", color=TEXT_COLOR,
                     pad=5.0)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9, fontweight="normal",
                      color=TEXT_COLOR, labelpad=4.0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9, fontweight="normal",
                      color=TEXT_COLOR, labelpad=4.0)
    ax.grid(bool(grid), which="major", color=GRID_COLOR, linewidth=0.6,
            alpha=0.25, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", direction="out", length=3.0,
                   width=0.8, pad=2.0, labelsize=8, colors=TEXT_COLOR,
                   top=False, right=False, bottom=True, left=True)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.8,
                   width=0.6, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor(TEXT_COLOR)
    return ax


def add_figure_title(fig, text, *, y):
    """Figure-level title at 10 pt, positioned in figure fraction."""
    return fig.suptitle(text, fontsize=10, fontweight="normal",
                        color=TEXT_COLOR, y=float(y))


def add_legend(fig, handles, labels, *, loc, bbox_to_anchor, ncol):
    """Figure legend with every spacing parameter stated explicitly."""
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor,
                      ncol=int(ncol), **LEGEND_KW)


# ---------------------------------------------------------------------------
# 3. Export
# ---------------------------------------------------------------------------

def save_publication_pair(fig, pdf_path: str | Path) -> tuple[Path, Path]:
    """Vector PDF plus 300 dpi PNG at exactly the requested figure size.

    ``bbox_inches=None`` and ``pad_inches=0`` are passed explicitly: any
    automatic cropping would make the exported width differ from the layout's
    ``figsize`` and break cross-figure consistency.
    """
    pdf = Path(pdf_path)
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"expected a .pdf output path, received {pdf}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png = pdf.with_suffix(".png")
    for target in (pdf, png):
        fig.savefig(target, dpi=DPI, bbox_inches=None, pad_inches=0.0,
                    transparent=False, facecolor=WHITE, edgecolor=WHITE)
    return pdf, png
