"""Shared figure export, caption, and layout helpers for experiment notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


# Every notebook copies and overrides this dictionary near its configuration
# cell, keeping all visual-spacing controls visible in one place.
DEFAULT_LAYOUT = {
    "left": 0.065,
    "right": 0.985,
    "bottom": 0.16,
    "top": 0.80,
    "wspace": 0.22,
    "hspace": 0.30,
    "suptitle_y": 0.985,
    "legend_y": 0.945,
    "bottom_legend_y": 0.02,
    "dpi": 220,
}


def apply_layout(fig, layout: Mapping[str, float]) -> None:
    """Apply the exposed subplot-spacing controls to a Matplotlib figure."""

    fig.subplots_adjust(
        left=float(layout["left"]),
        right=float(layout["right"]),
        bottom=float(layout["bottom"]),
        top=float(layout["top"]),
        wspace=float(layout["wspace"]),
        hspace=float(layout["hspace"]),
    )


def save_figure_pair(fig, pdf_path: str | Path, *, dpi: int = 220) -> tuple[Path, Path]:
    """Save a publication PDF and a matching high-resolution PNG side by side."""

    pdf = Path(pdf_path)
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf output path, received {pdf}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png = pdf.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=int(dpi))
    return pdf, png


def print_figure_caption(text: str) -> None:
    """Emit a copy-ready manuscript caption immediately after a plotting cell."""

    print("\nFigure caption:\n" + text)
