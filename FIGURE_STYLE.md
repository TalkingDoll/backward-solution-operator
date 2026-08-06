# Publication figure specification

This file is the copyable specification for every experiment figure in this
repository. The executable source of truth is
[`plotting_utils.py`](plotting_utils.py); each plotting script or notebook also
states its complete layout dictionary next to the plotting code.

The standard is based on the effective Lorenz-63/Lorenz-96 paper size:

- single-column figure width: **3.4 in**
- double-column or wide figure width: **7.0 in**
- final raster check: **300 dpi**
- font sizes: **10/9/9/8/8 pt** for figure title / panel title / axis label /
  tick label / legend

The key rule is that text, line, marker, tick, and spacing values are specified
in final physical units. The number of subplots changes the figure dimensions,
not the size of the contents of an individual subplot.

## Shared style values

```python
PUBLICATION_STYLE = {
    # Paper widths and resolution.
    "single_column_width_in": 3.4,
    "double_column_width_in": 7.0,
    "dpi": 300,

    # Typography.
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

    # Text spacing, in points.
    "panel_title_pad": 5.0,
    "axis_label_pad": 4.0,
    "tick_label_pad": 2.0,

    # Lines, markers, axes, ticks, and grids, in points.
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

    # Legend geometry, in points except for Boolean values.
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

    # Colors and draw order.
    "text_color": "#111111",
    "axes_edge_color": "#111111",
    "axes_face_color": "#FFFFFF",
    "figure_face_color": "#FFFFFF",
    "grid_color": "#9CA3AF",
    "axes_axisbelow": True,

    # Exact-size export.
    "save_face_color": "#FFFFFF",
    "save_edge_color": "#FFFFFF",
    "save_transparent": False,
    "save_bbox_inches": None,
    "save_pad_inches": 0.0,
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
}
```

All lines explicitly select `line_width`, `highlight_line_width`, or
`reference_line_width`. Marker-bearing curves explicitly select
`marker_size` and `marker_edge_width`. Every axis calls `style_axis`, and every
title, label, legend, annotation, and colorbar supplies its corresponding size
from this dictionary.

## Explicit figure layouts

All margins and inter-panel gaps below are physical inches. `top_in` and
`bottom_in` reserve space for the title, legend, and axis labels; they are not
Matplotlib's relative fractions.

| Output(s) | Grid | Figure (in) | L / R / B / T (in) | Column / row gap (in) | Figure-level extras |
|---|---:|---:|---:|---:|---|
| `eeg/eeg_rollout_comparison` | 1x1 | 7.0x3.0 | 0.48 / 0.10 / 0.50 / 0.80 | 0 / 0 | title 0.05 in from top; legend 0.38 in from top, 5 columns |
| `lorenz63/l63_prediction` | 1x3 | 7.0x3.0 | 0.50 / 0.10 / 0.46 / 0.86 | 0.30 / 0 | title 0.05 in from top; legend 0.35 in from top, 2 columns |
| `lorenz96/l96_prediction` | 1x3 | 7.0x3.0 | 0.50 / 0.10 / 0.46 / 0.86 | 0.30 / 0 | title 0.05 in from top; legend 0.35 in from top, 2 columns |
| `ou/*_eigenvalues` | 1x1 real-axis strip | 6.0x1.65 | 0.23 / 0.10 / 0.46 / 0.33 | 0 / 0 | in-axis legend, 2 columns; y axis hidden explicitly |
| `ou/*_eigenfunctions` | 2x3 | 7.0x4.6 | 0.48 / 0.10 / 0.46 / 0.65 | 0.34 / 0.46 | title 0.05 in from top; legend 0.23 in from top, 2 columns, explicit DejaVu Sans semibold 9 pt |
| `sst/sst_forecast_skill` | 2x2 | 7.0x5.25 | 0.55 / 0.10 / 0.48 / 0.90 | 0.62 / 0.62 | title 0.05 in from top; legend 0.34 in from top, 3 columns |
| `sst/sst_rollout_stability` | 1x2 | 7.0x3.0 | 0.53 / 0.10 / 0.48 / 0.82 | 0.54 / 0 | title 0.05 in from top; legend 0.36 in from top, 3 columns |
| `triple_well/tw_eigenvalues` | 1x1 real-axis strip | 6.0x1.65 | 0.23 / 0.10 / 0.46 / 0.33 | 0 / 0 | in-axis legend, 2 columns; y axis hidden explicitly |
| `triple_well/tw_eigenfunctions` | 1x3 | 7.0x2.2 | 0.55 / 0.32 / 0.43 / 0.48 | 0.65 / 0 | title 0.05 in from top; colorbar size 4.5%, pad 0.04 in |

The older two-panel OU evaluation function uses 7.0x3.0 in, margins
0.52 / 0.10 / 0.48 / 0.30 in, and a 0.55 in column gap.

## Required plotting and export sequence

```python
from plotting_utils import (
    PUBLICATION_STYLE,
    apply_physical_layout,
    apply_publication_rcparams,
    save_figure_pair,
    style_axis,
    y_from_top,
)

apply_publication_rcparams()
STYLE = PUBLICATION_STYLE

fig, axes = plt.subplots(
    nrows=LAYOUT["nrows"],
    ncols=LAYOUT["ncols"],
    figsize=LAYOUT["figsize"],
    dpi=STYLE["dpi"],
    squeeze=False,              # state explicitly for stable axes shape
    sharex=False,
    sharey=False,
    constrained_layout=False,
    facecolor=STYLE["figure_face_color"],
)

# Each plot call states its color, linestyle, linewidth, marker,
# markersize, markeredgewidth, markerfacecolor, markeredgecolor, alpha,
# zorder, and label as applicable.
# Each set_title/set_xlabel/set_ylabel/legend/annotate/colorbar call states
# its size and spacing using STYLE.
# Each subplot calls:
style_axis(ax, grid_axis="both", grid_visible=True, minor_ticks=False)

apply_physical_layout(fig, LAYOUT)
save_figure_pair(fig, output_pdf, dpi=STYLE["dpi"])
```

Do not add `tight_layout`, `constrained_layout=True`, or
`bbox_inches="tight"`: each can change the final canvas or the physical
subplot spacing. PDF and PNG are both written with an exact white,
non-transparent canvas; the PDF embeds TrueType fonts (`pdf.fonttype = 42`).

## Overleaf inclusion

Use the PDF outputs. Include a 3.4-inch figure at `\columnwidth` and a 7.0-inch
figure at `\textwidth`:

```latex
\includegraphics[width=\columnwidth]{figures/ou/ou_exp1_eigenvalues.pdf}
\includegraphics[width=\textwidth]{figures/lorenz63/l63_prediction.pdf}
```

Use only one width setting and do not add a second `scale` option. Consistency
assumes the single- and double-column figures are displayed at the matching
paper widths; arbitrary independent rescaling would also rescale the fonts,
lines, markers, margins, and gaps.
