"""One coherent visual system for every chart in the project.

Colour is assigned by the *job* it does, not by taste:
  * categorical -> identity (lane classes, a handful of named series)
  * sequential  -> magnitude (one hue, light to dark; never a rainbow)
  * diverging   -> polarity (two hues either side of a neutral grey midpoint)
  * status      -> state (reserved for severity; never reused as "series 4")

Two rules are enforced structurally rather than by discipline:
  * **No dual-axis charts.** Two measures on different scales become two
    stacked panels sharing an x-axis. A second y-scale lets the author imply
    any correlation they like by rescaling, which is exactly the claim these
    charts exist to test.
  * **Direct labels on low-contrast marks.** Several palette slots sit below
    3:1 against the surface, so values are labelled in text ink rather than
    left to be read from colour alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# Categorical slots, in fixed order. Never cycled: a 9th series folds into
# "Other" or becomes a facet instead of getting an invented hue.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Reserved status ramp - severity only.
STATUS = {"good": "#1baf7a", "warning": "#eda100",
          "serious": "#eb6834", "critical": "#c1272d"}

# Sequential: a single hue, light to dark. Magnitude only.
SEQ = LinearSegmentedColormap.from_list("spx_seq", ["#f2f7fd", "#2a78d6", "#10365f"])
# Diverging: two hues with a neutral grey midpoint - never a hue at the middle.
DIV = LinearSegmentedColormap.from_list("spx_div", ["#1baf7a", "#eeeeec", "#c1272d"])

# Stable colour per lane class, so a filtered chart never repaints survivors.
LANE_CLASS_COLORS = {
    "Intra-region": CAT[2], "Inter-region (land)": CAT[0],
    "Island-crossing": CAT[1], "Unmapped": INK_MUTED,
}
LANE_CLASS_ORDER = ["Intra-region", "Inter-region (land)", "Island-crossing", "Unmapped"]


def apply_theme() -> None:
    """Recessive grid and axes; the data carries the ink."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 160,
        "savefig.bbox": "tight", "figure.dpi": 110,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.labelcolor": INK_2, "axes.labelsize": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 8,
        "lines.solid_capstyle": "round",
    })


def titled(ax, title: str, subtitle: str | None = None) -> None:
    """Title plus an optional one-line reading of what the chart shows.

    Both are drawn as axes-relative text rather than via `set_title`, so the
    subtitle can be stacked underneath without the two colliding.
    """
    if subtitle:
        ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, va="bottom", ha="left")
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="bottom", ha="left")
    else:
        ax.text(0, 1.015, title, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, va="bottom", ha="left")


def caption(fig, text: str) -> None:
    """The observation the chart supports, rendered under the figure."""
    fig.text(0.0, -0.02, text, fontsize=9, color=INK_2, va="top", ha="left", wrap=True)


def save(fig, path, name: str):
    """Write the figure and return its path."""
    out = path / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def label_bars(ax, bars, values, fmt="{:.2f}", dx=0.0, dy=0.0, size=9, color=None):
    """Direct value labels - the relief required where marks sit under 3:1."""
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2 + dx, b.get_height() + dy,
                fmt.format(v), ha="center", va="bottom",
                fontsize=size, color=color or INK, fontweight="normal")


def rounded_bars(ax, x, heights, width=0.7, color=CAT[0], **kw):
    """Thin bars with softened data-ends, anchored to the baseline."""
    return ax.bar(x, heights, width=width, color=color,
                  edgecolor=SURFACE, linewidth=1.2, zorder=3, **kw)
