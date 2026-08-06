"""Explicit, shared publication-figure styling and export helpers.

The values below are the single source of truth for every experiment figure.
They are expressed in final paper units (inches and points), so a 3.4-inch
single-column figure and a 7.0-inch double-column figure have identical text,
line, marker, and spacing sizes when included at those physical widths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


# ---------------------------------------------------------------------------
# Complete publication style: no typography, line, marker, tick, grid, legend,
# background, or export size is left to a Matplotlib environment default.
# ---------------------------------------------------------------------------
PUBLICATION_STYLE = {
    # Paper widths and output resolution.
    "single_column_width_in": 3.4,
    "double_column_width_in": 7.0,
    "dpi": 300,
    # Font family and sizes (points at final paper size).
    "font_family": "sans-serif",
    "font_sans_serif": ("DejaVu Sans",),
    "mathtext_fontset": "dejavusans",
    "base_font_size": 8.0,
    "suptitle_size": 10.0,
    "panel_title_size": 9.0,
    "axis_label_size": 9.0,
    "tick_label_size": 8.0,
    "legend_size": 8.0,
    "annotation_size": 8.0,
    "colorbar_tick_size": 8.0,
    "font_weight": "normal",
    "title_weight": "normal",
    # Text spacing (points).
    "panel_title_pad": 5.0,
    "axis_label_pad": 4.0,
    "tick_label_pad": 2.0,
    # Data, reference, marker, axes, and grid sizes (points).
    "line_width": 1.4,
    "highlight_line_width": 1.7,
    "reference_line_width": 1.0,
    "marker_size": 4.0,
    "marker_edge_width": 0.8,
    "axes_line_width": 0.8,
    "major_tick_length": 3.0,
    "major_tick_width": 0.8,
    "minor_tick_length": 1.8,
    "minor_tick_width": 0.6,
    "grid_line_width": 0.6,
    "grid_alpha": 0.25,
    # Legend geometry (points unless otherwise noted).
    "legend_frameon": False,
    "legend_fancybox": False,
    "legend_framealpha": 1.0,
    "legend_borderaxespad": 0.3,
    "legend_borderpad": 0.2,
    "legend_labelspacing": 0.35,
    "legend_handlelength": 2.2,
    "legend_handleheight": 0.7,
    "legend_handletextpad": 0.5,
    "legend_columnspacing": 1.2,
    # Colors and layer ordering.
    "text_color": "#111111",
    "axes_edge_color": "#111111",
    "axes_face_color": "#FFFFFF",
    "figure_face_color": "#FFFFFF",
    "grid_color": "#9CA3AF",
    "axes_axisbelow": True,
    # Export settings.
    "save_face_color": "#FFFFFF",
    "save_edge_color": "#FFFFFF",
    "save_transparent": False,
    "save_bbox_inches": None,
    "save_pad_inches": 0.0,
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
}


def apply_publication_rcparams() -> None:
    """Apply every environment-level parameter used by publication figures."""

    import matplotlib as mpl

    style = PUBLICATION_STYLE
    mpl.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.sans-serif": list(style["font_sans_serif"]),
            "font.size": style["base_font_size"],
            "font.weight": style["font_weight"],
            "text.color": style["text_color"],
            "mathtext.fontset": style["mathtext_fontset"],
            "axes.titlesize": style["panel_title_size"],
            "axes.titleweight": style["title_weight"],
            "axes.titlepad": style["panel_title_pad"],
            "axes.labelsize": style["axis_label_size"],
            "axes.labelweight": style["font_weight"],
            "axes.labelpad": style["axis_label_pad"],
            "axes.linewidth": style["axes_line_width"],
            "axes.edgecolor": style["axes_edge_color"],
            "axes.labelcolor": style["text_color"],
            "axes.facecolor": style["axes_face_color"],
            "axes.axisbelow": style["axes_axisbelow"],
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.grid.which": "major",
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "grid.color": style["grid_color"],
            "grid.alpha": style["grid_alpha"],
            "grid.linewidth": style["grid_line_width"],
            "grid.linestyle": "-",
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "xtick.color": style["text_color"],
            "ytick.color": style["text_color"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": style["major_tick_length"],
            "ytick.major.size": style["major_tick_length"],
            "xtick.major.width": style["major_tick_width"],
            "ytick.major.width": style["major_tick_width"],
            "xtick.major.pad": style["tick_label_pad"],
            "ytick.major.pad": style["tick_label_pad"],
            "xtick.minor.size": style["minor_tick_length"],
            "ytick.minor.size": style["minor_tick_length"],
            "xtick.minor.width": style["minor_tick_width"],
            "ytick.minor.width": style["minor_tick_width"],
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": style["legend_size"],
            "legend.frameon": style["legend_frameon"],
            "legend.fancybox": style["legend_fancybox"],
            "legend.framealpha": style["legend_framealpha"],
            "legend.borderaxespad": style["legend_borderaxespad"],
            "legend.borderpad": style["legend_borderpad"],
            "legend.labelspacing": style["legend_labelspacing"],
            "legend.handlelength": style["legend_handlelength"],
            "legend.handleheight": style["legend_handleheight"],
            "legend.handletextpad": style["legend_handletextpad"],
            "legend.columnspacing": style["legend_columnspacing"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "lines.markeredgewidth": style["marker_edge_width"],
            "figure.titlesize": style["suptitle_size"],
            "figure.titleweight": style["title_weight"],
            "figure.facecolor": style["figure_face_color"],
            "figure.edgecolor": style["figure_face_color"],
            "figure.dpi": style["dpi"],
            "figure.autolayout": False,
            "figure.constrained_layout.use": False,
            "savefig.dpi": style["dpi"],
            "savefig.facecolor": style["save_face_color"],
            "savefig.edgecolor": style["save_edge_color"],
            "savefig.transparent": style["save_transparent"],
            "pdf.fonttype": style["pdf_fonttype"],
            "ps.fonttype": style["ps_fonttype"],
        }
    )


def style_axis(
    ax,
    *,
    grid_axis: str = "both",
    grid_visible: bool = True,
    minor_ticks: bool = False,
) -> None:
    """Apply explicit, identical axis geometry to one subplot."""

    style = PUBLICATION_STYLE
    ax.set_facecolor(style["axes_face_color"])
    ax.set_axisbelow(bool(style["axes_axisbelow"]))
    for side in ("left", "right", "bottom", "top"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(float(style["axes_line_width"]))
        spine.set_edgecolor(style["axes_edge_color"])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=float(style["major_tick_length"]),
        width=float(style["major_tick_width"]),
        pad=float(style["tick_label_pad"]),
        labelsize=float(style["tick_label_size"]),
        colors=style["text_color"],
        bottom=True,
        top=False,
        left=True,
        right=False,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=float(style["minor_tick_length"]),
        width=float(style["minor_tick_width"]),
        colors=style["text_color"],
        bottom=bool(minor_ticks),
        top=False,
        left=bool(minor_ticks),
        right=False,
    )
    if grid_visible:
        ax.grid(
            visible=True,
            which="major",
            axis=grid_axis,
            color=style["grid_color"],
            linestyle="-",
            linewidth=float(style["grid_line_width"]),
            alpha=float(style["grid_alpha"]),
        )
    else:
        # Supplying line properties while visible=False makes Matplotlib
        # re-enable the grid, so keep the disabled call property-free.
        ax.grid(visible=False, which="major", axis=grid_axis)
    ax.grid(visible=False, which="minor", axis=grid_axis)


def apply_physical_layout(fig, layout: Mapping[str, float | int]) -> None:
    """Apply margins and subplot gaps specified in physical inches.

    Required keys are ``nrows``, ``ncols``, ``left_in``, ``right_in``,
    ``bottom_in``, ``top_in``, ``column_gap_in``, and ``row_gap_in``.
    Matplotlib's relative ``wspace``/``hspace`` values are calculated from the
    requested physical gaps, preventing layouts with different grids from
    silently acquiring different gap sizes.
    """

    width_in, height_in = (float(value) for value in fig.get_size_inches())
    nrows = int(layout["nrows"])
    ncols = int(layout["ncols"])
    left_in = float(layout["left_in"])
    right_in = float(layout["right_in"])
    bottom_in = float(layout["bottom_in"])
    top_in = float(layout["top_in"])
    column_gap_in = float(layout["column_gap_in"])
    row_gap_in = float(layout["row_gap_in"])

    inner_width = width_in - left_in - right_in
    inner_height = height_in - bottom_in - top_in
    axes_width = (inner_width - max(ncols - 1, 0) * column_gap_in) / ncols
    axes_height = (inner_height - max(nrows - 1, 0) * row_gap_in) / nrows
    if axes_width <= 0.0 or axes_height <= 0.0:
        raise ValueError(
            "Explicit margins and gaps leave no positive subplot area: "
            f"figure={width_in:g}x{height_in:g} in, layout={dict(layout)}"
        )

    wspace = 0.0 if ncols == 1 else column_gap_in / axes_width
    hspace = 0.0 if nrows == 1 else row_gap_in / axes_height
    fig.subplots_adjust(
        left=left_in / width_in,
        right=1.0 - right_in / width_in,
        bottom=bottom_in / height_in,
        top=1.0 - top_in / height_in,
        wspace=wspace,
        hspace=hspace,
    )


def y_from_top(fig, offset_in: float) -> float:
    """Return normalized figure y for an explicit offset from the top edge."""

    height_in = float(fig.get_size_inches()[1])
    return 1.0 - float(offset_in) / height_in


def save_figure_pair(
    fig,
    pdf_path: str | Path,
    *,
    dpi: int | None = None,
) -> tuple[Path, Path]:
    """Save an exact-size vector PDF and matching explicitly configured PNG."""

    style = PUBLICATION_STYLE
    pdf = Path(pdf_path)
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf output path, received {pdf}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png = pdf.with_suffix(".png")
    output_dpi = int(style["dpi"] if dpi is None else dpi)
    common = {
        "facecolor": style["save_face_color"],
        "edgecolor": style["save_edge_color"],
        "transparent": bool(style["save_transparent"]),
        "bbox_inches": style["save_bbox_inches"],
        "pad_inches": float(style["save_pad_inches"]),
    }
    fig.savefig(pdf, format="pdf", dpi=output_dpi, **common)
    fig.savefig(png, format="png", dpi=output_dpi, **common)
    return pdf, png


def print_figure_caption(text: str) -> None:
    """Emit a copy-ready manuscript caption immediately after a plotting cell."""

    print("\nFigure caption:\n" + text)
