"""The brief's speed metrics — implemented once, used everywhere.

The assignment defines two weighted-average metrics:

    avg_BWT = SUM(sum_bwt) / SUM(parcel_qty)
    avg_APT = SUM(sum_apt) / SUM(parcel_qty)

The critical property is that these are *parcel-weighted*, not row averages.
`AVG(sum_bwt / parcel_qty)` gives a different (and wrong) answer because it lets
a 10-parcel row count as much as a 40,000-parcel row. Every aggregate in this
project routes through `weighted_metrics()` so that mistake cannot happen in
one place and not another.
"""
from __future__ import annotations

import pandas as pd

# Column names, kept as constants so the SQL guardrail can import the same list.
QTY, APT, BWT = "parcel_qty", "sum_apt", "sum_bwt"


def weighted_metrics(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Apply the brief's two formulas at any grain.

    `by=None` computes the global grain (brief image2, example 4).
    Returns parcel_qty / sum_apt / sum_bwt totals alongside the derived metrics
    so downstream code can re-aggregate without going back to the row data.
    """
    agg = {QTY: "sum", APT: "sum", BWT: "sum"}

    if by is None:
        # Global grain: sum everything into a single row.
        out = df[[QTY, APT, BWT]].sum().to_frame().T
    else:
        out = df.groupby(by, dropna=False, observed=True).agg(agg).reset_index()

    # The brief's formulas. Guard against divide-by-zero: a group with no
    # parcels has no defined average speed, so NaN is the honest answer.
    qty = out[QTY].replace(0, pd.NA)
    out["avg_BWT"] = out[BWT] / qty
    out["avg_APT"] = out[APT] / qty

    # Derived: the transit leg. Per brief image1, APT spans [paid -> handover]
    # and BWT spans [paid -> delivered], so the difference is everything that
    # happens after the seller hands over — i.e. the 3PL's portion.
    out["avg_transit"] = out["avg_BWT"] - out["avg_APT"]

    # Parcel-days of buyer waiting: converts a rate into a business quantity,
    # so priorities can be ranked by total impact rather than by worst rate.
    out["parcel_days_waiting"] = out[BWT]
    return out


def add_week_block(df: pd.DataFrame, date_col: str = "dt") -> pd.DataFrame:
    """Add fixed 7-day blocks for the brief's weekly grain (image2, example 3).

    Deliberately NOT ISO weeks. The data starts Thu 1 Jan 2026 and ends Fri 30
    Jan, so ISO weeks would be ragged at both ends and week-over-week
    comparisons would silently compare 7 days against 3. Fixed blocks anchored
    to the first date keep every complete block exactly 7 days wide; the
    remainder is flagged so it can be excluded from trend claims.
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    day_index = (d - d.min()).dt.days
    block = day_index // 7
    out["week_block"] = "W" + (block + 1).astype(str)
    # A block is complete only if all 7 of its days are present in the data.
    days_per_block = day_index.groupby(block).transform("nunique")
    out["week_is_complete"] = days_per_block == 7
    return out


def spec_grain(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Compute one of the four grains mandated by the brief's image2."""
    from spx.config import SPEC_GRAINS

    if grain not in SPEC_GRAINS:
        raise KeyError(f"{grain!r} is not one of the brief's grains: {list(SPEC_GRAINS)}")
    keys = SPEC_GRAINS[grain]["keys"]
    if grain == "weekly" and "week_block" not in df.columns:
        df = add_week_block(df)
    return weighted_metrics(df, by=keys)
