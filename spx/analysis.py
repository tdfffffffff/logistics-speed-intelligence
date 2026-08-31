"""Statistical analysis: anomalies, confounding, lag structure, business impact.

Deliberate separation of concerns: everything here is *deterministic and
auditable*. The LLM layer never decides what is anomalous or which provider is
worst -- it only explains findings this module has already established. That
keeps the analytical claims reproducible and the LLM's role bounded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spx.metrics import weighted_metrics


# --------------------------------------------------------------- anomalies
def detect_anomalies(df: pd.DataFrame, z_threshold: float = 6.0,
                     min_parcels: int = 5_000) -> pd.DataFrame:
    """Flag lane x provider x day cells that deviate from their own baseline.

    Uses a modified z-score built on the median and MAD rather than mean and
    standard deviation. With outliers present, a mean/std z-score is dragged
    toward the very anomalies it is meant to find; the median/MAD version is
    resistant to that. 0.6745 is the constant that puts the modified z-score on
    the same scale as a standard one.

    Judged against each lane's *own* baseline: 8 days is a crisis on a
    Jakarta-Jakarta lane and unremarkable on Penang-Sabah.

    On threshold choice: the residual distribution is heavy-tailed, as
    operational data almost always is, so the textbook |z| > 3.5 (which assumes
    normality) flags 4.1% of cells -- far more than any team could action. The
    default of 6.0 is set by **alert budget** instead: it yields ~103 alerts
    over 30 days, roughly 3 per day, which a duty operations manager can
    realistically triage. This is a business calibration, not a statistical
    one, and is exposed as a parameter so it can be retuned per team.
    """
    out = df[df["parcel_qty"] >= min_parcels].copy()
    # lane_mad_bwt is already floored and shrunk in features.add_lane_baseline,
    # so this division is safe and the z-scores are on a comparable scale.
    out["mad_z"] = 0.6745 * out["bwt_residual"] / out["lane_mad_bwt"]
    out["is_anomaly"] = out["mad_z"].abs() > z_threshold

    anomalies = out[out["is_anomaly"]].copy()
    anomalies["direction"] = np.where(anomalies["mad_z"] > 0, "slower", "faster")
    # Severity blends how abnormal it is with how many buyers it touched --
    # a 5-sigma blip on 6,000 parcels matters less than 4-sigma on 40,000.
    anomalies["severity"] = anomalies["mad_z"].abs() * np.log10(anomalies["parcel_qty"])
    anomalies["excess_parcel_days"] = anomalies["bwt_residual"] * anomalies["parcel_qty"]
    return anomalies.sort_values("severity", ascending=False)


# ------------------------------------------------- confounding / Simpson's
def provider_naive_vs_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Compare providers naively, then controlling for which lanes they serve.

    The naive ranking is confounded: providers do not serve the same lanes, and
    lane difficulty spans 1.1 to 10.4 days. A provider that only operates in
    Indonesia's island network will look terrible even if it is executing well.

    The matched figure uses **lane fixed effects**: within each lane, take each
    provider's gap to that lane's own volume-weighted mean, then average those
    gaps across lanes (weighting by parcels). Only lanes served by 2+ providers
    contribute, since a lane with one provider carries no comparative signal.
    """
    naive = weighted_metrics(df, ["logistics_provider"])[
        ["logistics_provider", "parcel_qty", "avg_BWT"]
    ].rename(columns={"avg_BWT": "naive_bwt"})

    lane_prov = weighted_metrics(df, ["lane", "logistics_provider"])
    # Restrict to lanes where a comparison is actually possible.
    n_prov = lane_prov.groupby("lane")["logistics_provider"].transform("nunique")
    comp = lane_prov[n_prov >= 2].copy()

    # Each lane's own benchmark = parcel-weighted mean BWT across its providers.
    lane_tot = comp.groupby("lane").apply(
        lambda g: np.average(g["avg_BWT"], weights=g["parcel_qty"]), include_groups=False
    ).rename("lane_benchmark")
    comp = comp.join(lane_tot, on="lane")
    comp["gap"] = comp["avg_BWT"] - comp["lane_benchmark"]

    matched = comp.groupby("logistics_provider").apply(
        lambda g: pd.Series({
            "matched_gap_vs_lane": np.average(g["gap"], weights=g["parcel_qty"]),
            "lanes_compared": g["lane"].nunique(),
            "parcels_compared": g["parcel_qty"].sum(),
        }), include_groups=False
    ).reset_index()

    out = naive.merge(matched, on="logistics_provider")
    out["naive_rank"] = out["naive_bwt"].rank().astype(int)
    out["matched_rank"] = out["matched_gap_vs_lane"].rank().astype(int)
    out["rank_shift"] = out["naive_rank"] - out["matched_rank"]
    return out.sort_values("matched_gap_vs_lane")


# ------------------------------------------------------------ lag structure
def lag_correlation(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Correlate daily volume against avg_BWT at increasing lags."""
    daily = df.groupby("dt").apply(
        lambda g: pd.Series({"qty": g["parcel_qty"].sum(),
                             "bwt": g["sum_bwt"].sum() / g["parcel_qty"].sum()}),
        include_groups=False,
    )
    rows = []
    for lag in range(max_lag + 1):
        paired = pd.concat([daily["qty"], daily["bwt"].shift(-lag)], axis=1).dropna()
        rows.append({"lag_days": lag, "n": len(paired),
                     "correlation": paired["qty"].corr(paired["bwt"])})
    return pd.DataFrame(rows)


def block_bootstrap_ci(df: pd.DataFrame, lag: int = 2, block: int = 5,
                       n_boot: int = 5_000, seed: int = 42) -> dict:
    """Confidence interval for a lagged correlation on an autocorrelated series.

    A plain correlation CI assumes independent observations. Daily logistics
    volume is strongly autocorrelated (spikes last two days, weekends
    repeat), so that assumption overstates precision. A moving-block bootstrap
    resamples contiguous blocks instead of individual days, preserving the
    short-range dependence and giving an honest interval.
    """
    daily = df.groupby("dt").apply(
        lambda g: pd.Series({"qty": g["parcel_qty"].sum(),
                             "bwt": g["sum_bwt"].sum() / g["parcel_qty"].sum()}),
        include_groups=False,
    )
    x = daily["qty"].values[:len(daily) - lag]
    y = daily["bwt"].values[lag:]
    n = len(x)
    observed = np.corrcoef(x, y)[0, 1]

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        xb, yb = x[idx], y[idx]
        if xb.std() > 0 and yb.std() > 0:
            stats.append(np.corrcoef(xb, yb)[0, 1])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"lag_days": lag, "observed_r": observed, "ci_low": lo, "ci_high": hi,
            "n_effective": n, "block_size": block, "n_bootstrap": len(stats),
            "ci_method": f"moving-block bootstrap, {block}-day blocks, "
                         f"{len(stats)} resamples"}


# --------------------------------------------------------- business impact
def impact_sizing(df: pd.DataFrame, group: list[str], benchmark_quantile: float = 0.5
                  ) -> pd.DataFrame:
    """Rank opportunities by parcel-days recoverable, not by worst rate.

    A lane averaging 10 days on 600k parcels matters far more than one
    averaging 12 days on 20k. Converting a rate into parcel-days of buyer
    waiting is what makes the two comparable -- and it is the number an
    operations manager can actually act on.
    """
    agg = weighted_metrics(df, group)
    benchmark = agg["avg_BWT"].quantile(benchmark_quantile)
    agg["benchmark_bwt"] = benchmark
    agg["excess_bwt"] = (agg["avg_BWT"] - benchmark).clip(lower=0)
    agg["recoverable_parcel_days"] = agg["excess_bwt"] * agg["parcel_qty"]
    total = agg["recoverable_parcel_days"].sum()
    agg["pct_of_total_opportunity"] = 100 * agg["recoverable_parcel_days"] / total
    return agg.sort_values("recoverable_parcel_days", ascending=False)
