"""Engineered features shared by EDA, the ML models, the fact pack and the personas.

The important one is `detect_campaign_days`: the volume spikes in this dataset
are *found* from the signal, not hardcoded as dates. Hardcoding would produce a
notebook that silently breaks the first time it sees a different month.

Naming note: the `campaign` prefix is a convention retained across the feature
names, the fact pack and the SQL schema. It is not a claim about cause -- see
`detect_campaign_days` below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spx.config import ISLAND_REGIONS, UNKNOWN_SENTINEL
from spx.metrics import add_week_block, weighted_metrics


def detect_campaign_days(df: pd.DataFrame, z_threshold: float = 1.0) -> pd.DatetimeIndex:
    """Find recurring volume-spike days from the volume signal itself.

    Any day whose total volume sits `z_threshold` robust standard deviations
    above the median is flagged. A robust (median/MAD) statistic is used
    because the spikes themselves would inflate a mean/std and mask the very
    days we are hunting for.

    What this returns is a set of *volume spikes*, not confirmed sale events.
    In this dataset the spikes are unambiguous (z = 3.0-4.4 against a highest
    unflagged day of z = 0.76, stable for any threshold in 1.0-3.0) but their
    cause is not established: they recur on an exact nine-day cycle matching no
    commercial calendar, rotate across days of the week, and 1 January -- the
    largest scheduled sale date in the region -- is below median. The function
    name is kept for continuity with the fact-pack and SQL column names; treat
    the output as "recurring demand pulse", not "campaign".
    """
    daily = df.groupby("dt")["parcel_qty"].sum()
    med = daily.median()
    mad = (daily - med).abs().median()
    # 1.4826 rescales MAD to be a consistent estimator of sigma for normal data.
    robust_sigma = mad * 1.4826
    z = (daily - med) / robust_sigma
    return pd.DatetimeIndex(daily.index[z > z_threshold])


def add_time_features(df: pd.DataFrame, campaign_days=None) -> pd.DataFrame:
    """Calendar features plus distance-to-campaign, the key operational driver."""
    out = add_week_block(df)
    out["dt"] = pd.to_datetime(out["dt"])
    out["day_of_week"] = out["dt"].dt.day_name()
    out["dow_num"] = out["dt"].dt.dayofweek
    out["is_weekend"] = out["dow_num"] >= 5

    if campaign_days is None:
        campaign_days = detect_campaign_days(out)
    out["is_campaign_day"] = out["dt"].isin(campaign_days)

    # Signed distance in days to the nearest campaign. Negative = before a
    # campaign, positive = after. The "after" side is where backlog lives.
    if len(campaign_days):
        # Subtract as datetime64, not as raw int64. `dt` arrives as
        # datetime64[us] when parsed from the CSV but datetime64[ns] when read
        # back from the parquet, while `Timestamp.value` is always ns. Mixing
        # the two is off by 1000x and silently collapses every row to the same
        # constant (~-20440, i.e. the date as days since epoch) instead of
        # raising. numpy handles the unit promotion correctly.
        dt64 = pd.DatetimeIndex(out["dt"]).values
        camp = pd.DatetimeIndex(campaign_days).values
        deltas = (dt64[:, None] - camp[None, :]) / np.timedelta64(1, "D")
        nearest = np.argmin(np.abs(deltas), axis=1)
        out["days_from_campaign"] = deltas[np.arange(len(out)), nearest].astype(int)
    else:
        out["days_from_campaign"] = 0
    return out


def add_lane_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lane geography: the dominant physical driver of transit time."""
    out = df.copy()
    b, s = out["buyer_region"].astype(str), out["seller_region"].astype(str)

    out["lane"] = s + " > " + b
    out["has_unknown_endpoint"] = (b == UNKNOWN_SENTINEL) | (s == UNKNOWN_SENTINEL)
    out["is_intra_region"] = (b == s) & ~out["has_unknown_endpoint"]

    # An island crossing means a ferry or air leg, not just a longer drive --
    # a step change in transit time, not a gradual one.
    b_isl, s_isl = b.isin(ISLAND_REGIONS), s.isin(ISLAND_REGIONS)
    out["is_island_crossing"] = (b_isl != s_isl) | (b_isl & s_isl & (b != s))
    out["is_island_crossing"] &= ~out["has_unknown_endpoint"]

    # A single ordered class for charts and for the persona vocabulary.
    out["lane_class"] = np.select(
        [out["has_unknown_endpoint"], out["is_intra_region"], out["is_island_crossing"]],
        ["Unmapped", "Intra-region", "Island-crossing"],
        default="Inter-region (land)",
    )
    return out


def add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level rates. Safe because zero-qty rows are quarantined upstream."""
    out = df.copy()
    q = out["parcel_qty"].where(out["parcel_qty"] > 0)
    out["avg_BWT"] = out["sum_bwt"] / q
    out["avg_APT"] = out["sum_apt"] / q
    out["avg_transit"] = out["avg_BWT"] - out["avg_APT"]
    return out


def add_lane_baseline(df: pd.DataFrame, min_obs: int = 8,
                      mad_floor: float = 0.05) -> pd.DataFrame:
    """Each lane's own baseline speed and dispersion, with hierarchical shrinkage.

    This is what makes anomaly detection meaningful: 8 days is catastrophic on a
    Jakarta-Jakarta lane and completely normal on a Penang-Sabah one, so a row
    must be judged against its own lane, never a global average.

    Two problems make the naive version unusable, both fixed here:

    1. **Sparsity.** The lane x provider grain has a median of only 5
       observations, and 686 of 1,578 groups have fewer than 5. A MAD computed
       from 3 points is not a dispersion estimate. Groups below `min_obs` are
       therefore *shrunk* onto a coarser but well-populated fallback --
       country x lane_class -- rather than trusted on their own.
    2. **Near-zero dispersion.** 271 groups have a MAD of exactly 0 and 3,113
       sit below 0.05 days. Dividing by these produces z-scores in the millions.
       A floor of `mad_floor` days (~72 minutes) is applied: below that,
       variation in a multi-day delivery metric is measurement noise, not signal.
    """
    out = df.copy()
    fine = ["lane", "logistics_provider"]
    coarse = ["buyer_country", "lane_class"]

    def _mad(s):
        return (s - s.median()).abs().median()

    g_fine = out.groupby(fine, observed=True)["avg_BWT"]
    n_obs = g_fine.transform("size")
    fine_median, fine_mad = g_fine.transform("median"), g_fine.transform(_mad)

    g_coarse = out.groupby(coarse, observed=True)["avg_BWT"]
    coarse_median, coarse_mad = g_coarse.transform("median"), g_coarse.transform(_mad)

    # Trust the fine-grained estimate only where there is enough data for it.
    enough = n_obs >= min_obs
    out["lane_obs_count"] = n_obs
    out["baseline_source"] = np.where(enough, "lane x provider", "country x lane_class")
    out["lane_baseline_bwt"] = np.where(enough, fine_median, coarse_median)
    out["lane_mad_bwt"] = np.maximum(np.where(enough, fine_mad, coarse_mad), mad_floor)
    out["bwt_residual"] = out["avg_BWT"] - out["lane_baseline_bwt"]
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline, in dependency order."""
    out = add_row_metrics(df)
    out = add_lane_features(out)
    out = add_time_features(out)
    out = add_lane_baseline(out)
    return out
