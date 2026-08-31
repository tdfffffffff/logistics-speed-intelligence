"""The hero charts. Each answers one question and carries one observation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spx import viz
from spx.metrics import weighted_metrics
from spx.viz import CAT, DIV, GRID, INK, INK_2, INK_MUTED, SEQ, STATUS, SURFACE


def _daily(df):
    return df.groupby("dt").apply(
        lambda g: pd.Series({"qty": g["parcel_qty"].sum(),
                             "bwt": g["sum_bwt"].sum() / g["parcel_qty"].sum()}),
        include_groups=False).reset_index()


def fig_campaign_backlog(df, campaign_days, lag_tbl, ci):
    """Volume and speed on stacked panels sharing one x-axis.

    Deliberately NOT a dual-axis chart. Plotting parcels and days against two
    y-scales would let the axis limits manufacture or hide the very lag this
    figure exists to demonstrate. Separate panels force the reader to compare
    positions along a shared time axis, which is the honest comparison.
    """
    d = _daily(df)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), height_ratios=[2, 2, 1.6],
                             constrained_layout=True)
    ax1, ax2, ax3 = axes
    camp = set(pd.to_datetime(campaign_days))

    # --- panel 1: volume -------------------------------------------------
    colors = [CAT[1] if t in camp else CAT[0] for t in d["dt"]]
    ax1.bar(d["dt"], d["qty"] / 1e6, width=0.72, color=colors,
            edgecolor=SURFACE, linewidth=1.2, zorder=3)
    viz.titled(ax1, "Parcel volume peaks on detected spike days",
               "Orange = detected volume spike (> 1 robust SD above median)")
    ax1.set_ylabel("Parcels (millions)")
    for t in sorted(camp):
        v = d.loc[d["dt"] == t, "qty"].iloc[0] / 1e6
        ax1.text(t, v + 0.12, f"{v:.1f}M", ha="center", fontsize=8,
                 color=CAT[1], fontweight="bold")

    # --- panel 2: speed, with the +2 day backlog window marked ------------
    ax2.plot(d["dt"], d["bwt"], color=CAT[0], zorder=3)
    ax2.scatter(d["dt"], d["bwt"], s=26, color=CAT[0], zorder=4,
                edgecolor=SURFACE, linewidth=1.2)
    for t in sorted(camp):
        # Shade the two days AFTER each campaign - where the backlog lands.
        ax2.axvspan(t + pd.Timedelta(days=0.5), t + pd.Timedelta(days=2.5),
                    color=CAT[1], alpha=0.10, zorder=1)
    worst = d.nlargest(4, "bwt")
    for _, r in worst.iterrows():
        ax2.annotate(f"{r['bwt']:.2f}d", (r["dt"], r["bwt"]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8, color=INK, fontweight="bold")
    viz.titled(ax2, "Buyer waiting time degrades 1-2 days AFTER each volume spike",
               "Shaded band = the 48 hours following a volume spike")
    ax2.set_ylabel("avg_BWT (days)")

    # --- panel 3: the lag structure --------------------------------------
    bars = ax3.bar(lag_tbl["lag_days"], lag_tbl["correlation"], width=0.55,
                   color=[STATUS["critical"] if l == 2 else INK_MUTED
                          for l in lag_tbl["lag_days"]],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax3.axhline(0, color=GRID, linewidth=1.2, zorder=2)
    for b, v in zip(bars, lag_tbl["correlation"]):
        ax3.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.11),
                 f"{v:+.2f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    viz.titled(ax3, "Correlation of today's volume with BWT n days later",
               f"Peak at lag 2: r = {ci['observed_r']:.3f}  "
               f"(95% block-bootstrap CI {ci['ci_low']:.2f} to {ci['ci_high']:.2f}, n = {ci['n_effective']})")
    ax3.set_xlabel("Lag (days after the volume spike)")
    ax3.set_ylabel("Correlation")
    ax3.set_ylim(-0.75, 1.12)
    ax3.grid(axis="y")
    return fig


def fig_lane_class(df):
    """Where the delay physically lives: distance class, split APT vs transit."""
    agg = weighted_metrics(df, ["lane_class"]).set_index("lane_class")
    agg = agg.reindex([c for c in viz.LANE_CLASS_ORDER if c in agg.index])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1.35, 1],
                                   constrained_layout=True)
    y = np.arange(len(agg))
    # Stacked: preparation (seller-controlled) vs transit (3PL-controlled).
    ax1.barh(y, agg["avg_APT"], height=0.6, color=CAT[3], zorder=3,
             edgecolor=SURFACE, linewidth=1.6, label="Preparation (APT) - seller")
    ax1.barh(y, agg["avg_transit"], left=agg["avg_APT"], height=0.6, color=CAT[0],
             zorder=3, edgecolor=SURFACE, linewidth=1.6, label="Transit - 3PL network")
    for i, (a, t) in enumerate(zip(agg["avg_APT"], agg["avg_transit"])):
        ax1.text(a / 2, i, f"{a:.2f}", ha="center", va="center", fontsize=9,
                 color="#3d2c00", fontweight="bold")
        ax1.text(a + t / 2, i, f"{t:.2f}", ha="center", va="center", fontsize=9,
                 color="white", fontweight="bold")
        ax1.text(a + t + 0.09, i, f"{a+t:.2f}d", va="center", fontsize=9.5,
                 color=INK, fontweight="bold")
    ax1.set_yticks(y); ax1.set_yticklabels(agg.index)
    ax1.invert_yaxis(); ax1.grid(axis="x"); ax1.set_axisbelow(True)
    ax1.set_xlabel("Days")
    # Legend sits below the plot: inside the axes it collided with the total
    # labels at the end of the longest bar.
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    # Headroom on the right so the bold total labels are never clipped.
    ax1.set_xlim(0, (agg["avg_APT"] + agg["avg_transit"]).max() * 1.16)
    viz.titled(ax1, "Preparation time is flat; transit time is not",
               "APT barely moves across lane types - the delay is downstream of the seller")

    # Volume, so the reader can weigh the rates above by how many buyers they touch.
    bars = ax2.bar(range(len(agg)), agg["parcel_qty"] / 1e6, width=0.6,
                   color=[viz.LANE_CLASS_COLORS[c] for c in agg.index],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
    viz.label_bars(ax2, bars, agg["parcel_qty"] / 1e6, fmt="{:.1f}M", dy=0.6)
    ax2.set_xticks(range(len(agg)))
    ax2.set_xticklabels([c.replace(" (land)", "\n(land)").replace("-", "-\n", 1)
                         if len(c) > 13 else c for c in agg.index], fontsize=8.5)
    ax2.set_ylabel("Parcels (millions)")
    ax2.set_ylim(0, agg["parcel_qty"].max() / 1e6 * 1.18)
    viz.titled(ax2, "...and volume is concentrated in the slow classes",
               "Island-crossing carries 68M parcels, not a rounding error")
    return fig


def fig_simpsons(cmp_tbl):
    """Naive provider ranking vs the same providers compared within lanes."""
    t = cmp_tbl.copy()
    # Express both on one comparable scale: gap to the all-provider mean.
    overall = np.average(t["naive_bwt"], weights=t["parcel_qty"])
    t["naive_gap"] = t["naive_bwt"] - overall
    t = t.sort_values("naive_gap")

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    for i, r in enumerate(t.itertuples()):
        moved = abs(r.naive_gap - r.matched_gap_vs_lane)
        # Colour the connector by whether controlling for lanes helped or hurt.
        col = STATUS["good"] if r.matched_gap_vs_lane < r.naive_gap else STATUS["serious"]
        ax.plot([r.naive_gap, r.matched_gap_vs_lane], [i, i], color=col,
                linewidth=2.4 if moved > 0.4 else 1.2,
                alpha=0.85 if moved > 0.4 else 0.4, zorder=2,
                solid_capstyle="round")
        ax.scatter(r.naive_gap, i, s=70, color=INK_MUTED, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(r.matched_gap_vs_lane, i, s=95, color=col, zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.text(r.matched_gap_vs_lane + (0.06 if r.matched_gap_vs_lane >= r.naive_gap else -0.06),
                i, f"{r.matched_gap_vs_lane:+.2f}",
                va="center", ha="left" if r.matched_gap_vs_lane >= r.naive_gap else "right",
                fontsize=9, color=INK, fontweight="bold")

    ax.axvline(0, color=GRID, linewidth=1.4, zorder=1)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t["logistics_provider"])
    ax.set_xlabel("Gap to benchmark, in days (negative = faster)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    # Limits derived from the data, with headroom for the direct labels -- a
    # hardcoded range clipped SiCepat's naive marker off the canvas.
    lo = min(t["naive_gap"].min(), t["matched_gap_vs_lane"].min())
    hi = max(t["naive_gap"].max(), t["matched_gap_vs_lane"].max())
    pad = (hi - lo) * 0.18
    ax.set_xlim(lo - pad, hi + pad)

    # Legend by proxy marks, since identity here is shape+colour, not colour alone.
    ax.scatter([], [], s=70, color=INK_MUTED, label="Naive: vs national average")
    ax.scatter([], [], s=95, color=STATUS["good"], label="Lane-matched: looks better once controlled")
    ax.scatter([], [], s=95, color=STATUS["serious"], label="Lane-matched: looks worse once controlled")
    ax.legend(loc="lower right")
    viz.titled(ax, "Most of the provider gap is geography - but not all of it",
               "Comparing each provider only against rivals on the SAME lanes")
    return fig


def fig_failure_modes(df, anomalies):
    """Speed and reliability are different problems with different fixes.

    A provider can be slow-but-predictable or fast-but-erratic. Averaging hides
    the distinction, yet it decides the intervention: renegotiate a lane vs.
    fix incident response. Plotted as position on two axes with direct labels,
    so identity never depends on colour.
    """
    base = df[df["parcel_qty"] >= 5_000]
    n_elig = base.groupby("logistics_provider").size()
    n_anom = anomalies.groupby("logistics_provider").size().reindex(n_elig.index).fillna(0)
    prov = weighted_metrics(df, ["logistics_provider"]).set_index("logistics_provider")
    prov["anomaly_rate"] = 100 * n_anom / n_elig

    med_speed = np.average(prov["avg_BWT"], weights=prov["parcel_qty"])
    med_rate = prov["anomaly_rate"].median()

    fig, ax = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)
    ax.axvline(med_speed, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(med_rate, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)

    # Quadrant labels make the read explicit rather than implied.
    # x increases to the RIGHT = slower; y increases UPWARD = more erratic.
    #   left  = faster        right = slower
    #   top   = erratic       bottom = consistent
    for xf, ha, yf, va, txt, col in [
        (0.015, "left",  0.97, "top",    "ERRATIC\nfast on average, spikes hard", STATUS["warning"]),
        (0.985, "right", 0.97, "top",    "URGENT\nslow AND unreliable", STATUS["critical"]),
        (0.985, "right", 0.03, "bottom", "STRUCTURALLY SLOW\nconsistent, but consistently late", STATUS["serious"]),
        (0.015, "left",  0.03, "bottom", "HEALTHY\nfast and predictable", STATUS["good"])]:
        ax.text(xf, yf, txt, transform=ax.transAxes, ha=ha, va=va,
                fontsize=8.5, color=col, fontweight="bold", alpha=0.8, linespacing=1.35)

    sizes = 90 + 900 * (prov["parcel_qty"] / prov["parcel_qty"].max())
    ax.scatter(prov["avg_BWT"], prov["anomaly_rate"], s=sizes, color=CAT[0],
               alpha=0.55, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    for name, r in prov.iterrows():
        ax.annotate(name, (r["avg_BWT"], r["anomaly_rate"]),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=9, color=INK, fontweight="bold", zorder=4)
    ax.set_xlabel("avg_BWT (days) - slower to the right")
    ax.set_ylabel("Severe anomaly rate (% of high-volume cells flagged)")
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.margins(0.16)
    viz.titled(ax, "Two different failure modes need two different fixes",
               "Bubble size = parcel volume. Dashed lines = volume-weighted mean and median")
    return fig


def fig_anomaly_timeline(anomalies, campaign_days):
    """When incidents happen, relative to the detected volume spikes."""
    a = anomalies[anomalies["direction"] == "slower"].copy()
    camp = set(pd.to_datetime(campaign_days))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), height_ratios=[2, 1.1],
                                   constrained_layout=True)
    # Severity ramp is a reserved status scale, not a categorical one.
    bins = [0, 20, 50, 100, np.inf]
    names = ["warning", "serious", "critical", "critical"]
    a["sev_color"] = [STATUS[names[int(np.digitize(z, bins)) - 1]] for z in a["mad_z"].abs()]

    for t in sorted(camp):
        ax1.axvspan(t + pd.Timedelta(days=0.5), t + pd.Timedelta(days=2.5),
                    color=CAT[1], alpha=0.10, zorder=1)
    # Small horizontal jitter: several incidents share a date, and without it
    # the low-severity cluster becomes an unreadable blob.
    rng = np.random.default_rng(42)
    jitter = pd.to_timedelta(rng.uniform(-0.32, 0.32, len(a)), unit="D")
    ax1.scatter(a["dt"] + jitter, a["mad_z"].abs(),
                s=28 + 340 * (a["parcel_qty"] / a["parcel_qty"].max()),
                color=a["sev_color"], alpha=0.62, edgecolor=SURFACE,
                linewidth=1.0, zorder=3)
    ax1.set_ylabel("Deviation from lane baseline (modified z)")
    ax1.set_yscale("log")
    viz.titled(ax1, "Severe delay incidents cluster in the post-spike window",
               "Each dot = one lane x provider x day. Shaded = 48h after a spike")

    # The same claim as a distribution, so it does not rest on eyeballing dots.
    counts = a["days_from_campaign"].value_counts().sort_index()
    cols = [STATUS["critical"] if i in (1, 2) else INK_MUTED for i in counts.index]
    bars = ax2.bar(counts.index, counts.values, width=0.62, color=cols,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
    viz.label_bars(ax2, bars, counts.values, fmt="{:.0f}", dy=0.6)
    ax2.set_xlabel("Days from nearest volume spike (negative = before)")
    ax2.set_ylabel("Incidents")
    ax2.set_ylim(0, counts.max() * 1.2)
    share = 100 * counts.reindex([1, 2]).fillna(0).sum() / counts.sum()
    viz.titled(ax2, f"{share:.0f}% of severe incidents land 1-2 days after a spike",
               "Independent confirmation of the lag found in the correlation analysis")
    return fig


def fig_lane_matrix(df, country: str, ax=None):
    """Region-to-region speed for one country. Sequential = magnitude."""
    sub = df[(df["buyer_country"] == country) & (~df["has_unknown_endpoint"])]
    m = weighted_metrics(sub, ["seller_region", "buyer_region"])
    piv = m.pivot(index="seller_region", columns="buyer_region", values="avg_BWT")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.reindex(index=order, columns=order)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    # Lanes with no traffic are masked and painted a neutral grey. Left as NaN
    # they render near-white, which is visually almost identical to the fastest
    # lanes -- so "no data" would read as "excellent", the wrong conclusion.
    cmap = SEQ.copy()
    cmap.set_bad("#dedcd6")
    im = ax.imshow(np.ma.masked_invalid(piv.values), cmap=cmap,
                   aspect="auto", vmin=1, vmax=10)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("Buyer region (destination)"); ax.set_ylabel("Seller region (origin)")
    ax.grid(False)
    # Label only the extremes: a number in every cell is noise.
    flat = piv.stack().sort_values()
    for lbl in list(flat.index[:2]) + list(flat.index[-3:]):
        i, j = piv.index.get_loc(lbl[0]), piv.columns.get_loc(lbl[1])
        v = piv.loc[lbl[0], lbl[1]]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                color="white" if v > 5.5 else INK, fontweight="bold")
    # Regions on both axes are ordered slowest-last, so the visual block in the
    # lower-right corner is the island group.
    viz.titled(ax, country)
    return (fig, im) if created else im


def fig_lane_matrix_grid(df):
    """All four countries on one shared colour scale, so they are comparable."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), constrained_layout=True)
    for ax, c in zip(axes.ravel(), ["ID", "MY", "PH", "TH"]):
        im = fig_lane_matrix(df, c, ax=ax)
    cbar = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02)
    cbar.set_label("avg_BWT (days)", fontsize=9)
    cbar.outline.set_visible(False)
    fig.suptitle("Geography of delay: island crossings are a step change, not a gradient",
                 fontsize=13, fontweight="bold", x=0.01, ha="left", y=1.045)
    # Said once for the whole grid rather than repeated in every panel.
    fig.text(0.01, 1.012,
             "Regions ordered fastest to slowest on both axes, so the hot block in the "
             "lower-right of each panel is that country's island group.  "
             "Diagonal = same-region.  Grey = no traffic on that lane.",
             fontsize=9, color=INK_2, ha="left", va="bottom")
    return fig


def fig_impact(df):
    """Prioritise by parcel-days recoverable, not by worst rate.

    A 12-day lane carrying 20k parcels is a smaller problem than a 5-day lane
    carrying 600k. Ranking by rate alone sends the team to the wrong place;
    this converts rate into a quantity of buyer waiting that can be recovered.
    """
    from spx.analysis import impact_sizing
    imp = impact_sizing(df[~df["has_unknown_endpoint"]], ["buyer_country", "lane_class"])
    imp = imp[imp["recoverable_parcel_days"] > 0].head(8).iloc[::-1]
    labels = imp["buyer_country"] + "  " + imp["lane_class"]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    y = np.arange(len(imp))
    bars = ax.barh(y, imp["recoverable_parcel_days"] / 1e6, height=0.62,
                   color=[viz.LANE_CLASS_COLORS[c] for c in imp["lane_class"]],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for i, r in enumerate(imp.itertuples()):
        ax.text(r.recoverable_parcel_days / 1e6 + 1.2, i,
                f"{r.recoverable_parcel_days/1e6:.0f}M  ({r.pct_of_total_opportunity:.0f}%)",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Recoverable parcel-days per month (millions)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_xlim(0, imp["recoverable_parcel_days"].max() / 1e6 * 1.28)
    total = imp["recoverable_parcel_days"].sum() / 1e6
    viz.titled(ax, "Where the recoverable buyer-waiting actually sits",
               f"If every segment hit the median lane speed: ~{total:.0f}M parcel-days/month. "
               "Top 2 segments = 68% of the prize")
    return fig


def fig_informative_missingness(df):
    """Unmapped rows are not missing at random - they are the slowest lanes.

    This matters twice over: it points Data Engineering at remote-address
    geocoding rather than a random pipeline glitch, and it means any analyst
    who quietly drops these rows will *understate* national BWT.
    """
    rows = []
    u, m = df[df["has_unknown_endpoint"]], df[~df["has_unknown_endpoint"]]
    for (c, p), g in u.groupby(["buyer_country", "logistics_provider"], observed=True):
        peer = m[(m["buyer_country"] == c) & (m["logistics_provider"] == p)]
        if len(peer) > 20 and len(g) > 3:
            rows.append({"label": f"{c}  {p}",
                         "unmapped": g["sum_bwt"].sum() / g["parcel_qty"].sum(),
                         "mapped": peer["sum_bwt"].sum() / peer["parcel_qty"].sum()})
    t = pd.DataFrame(rows)
    t["delta"] = t["unmapped"] - t["mapped"]
    t = t.sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 6.6), constrained_layout=True)
    for i, r in enumerate(t.itertuples()):
        ax.plot([r.mapped, r.unmapped], [i, i], color=STATUS["serious"],
                linewidth=2.2, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter(r.mapped, i, s=64, color=CAT[0], zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.scatter(r.unmapped, i, s=64, color=STATUS["critical"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.text(r.unmapped + 0.1, i, f"+{r.delta:.1f}d", va="center",
                fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t["label"], fontsize=8.5)
    ax.set_xlabel("avg_BWT (days)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_xlim(right=t["unmapped"].max() * 1.14)
    ax.scatter([], [], s=64, color=CAT[0], label="Rows with a mapped region")
    ax.scatter([], [], s=64, color=STATUS["critical"], label="Rows with an UNKNOWN region")
    ax.legend(loc="lower right")
    viz.titled(ax, "The missing region data is not missing at random",
               f"Unmapped rows are slower in {(t['delta']>0).sum()} of {len(t)} "
               f"country x provider pairs - median +{t['delta'].median():.2f} days")
    return fig


def fig_provider_trend(df):
    """Direction of travel per provider, first 10 days vs last 10."""
    d = df.copy()
    # Period cuts derived from the observed date range, not hardcoded dates --
    # the same reason campaign days are detected rather than named.
    day = (d["dt"] - d["dt"].min()).dt.days
    span = day.max() + 1
    d["period"] = np.where(day < span / 3, "W1",
                           np.where(day < 2 * span / 3, "W2", "W3"))
    t = (d.groupby(["logistics_provider", "period"], observed=True)
           .apply(lambda g: g["sum_bwt"].sum() / g["parcel_qty"].sum(), include_groups=False)
           .unstack())
    t["delta"] = t["W3"] - t["W1"]
    t = t.sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        worse = r.delta > 0
        col = STATUS["critical"] if worse else STATUS["good"]
        ax.annotate("", xy=(r.W3, i), xytext=(r.W1, i),
                    arrowprops=dict(arrowstyle="-|>", color=col, linewidth=2.2,
                                    shrinkA=0, shrinkB=0, mutation_scale=14))
        ax.scatter(r.W1, i, s=52, color=INK_MUTED, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.text(max(r.W1, r.W3) + 0.06, i, f"{r.delta:+.2f}d", va="center",
                fontsize=9, color=col, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=9.5)
    ax.set_xlabel("avg_BWT (days)  -  arrow runs from the first third of the period to the last")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_xlim(t[["W1", "W3"]].min().min() - 0.2, t[["W1", "W3"]].max().max() + 0.55)
    ax.scatter([], [], s=52, color=INK_MUTED, label="Start of month")
    ax.plot([], [], color=STATUS["critical"], linewidth=2.2, label="Got slower")
    ax.plot([], [], color=STATUS["good"], linewidth=2.2, label="Got faster")
    ax.legend(loc="lower right")
    viz.titled(ax, "Direction of travel: who is improving and who is sliding",
               "SiCepat is both the slowest provider and the fastest-deteriorating")
    return fig


def fig_bakeoff(scores: pd.DataFrame, judge: pd.DataFrame):
    """Model comparison on the axes that decide a production choice.

    Two panels rather than one composite bar, because the interesting result is
    the *disagreement* between the axes: the model the LLM judge likes best is
    the one that fabricates most.
    """
    ok = scores[scores["ok"] & scores["words"].notna()]
    a = ok.groupby("model").agg(hallu=("hallucination_rate", "mean"),
                                cov=("coverage", "mean"),
                                words=("words", "mean"),
                                lat=("latency_s", "median"),
                                usd=("usd_cost", "sum")).join(
        judge.groupby("model")["judge_mean"].mean().rename("judge"))
    a = a.sort_values("hallu")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    # --- panel 1: the two quality axes side by side ----------------------
    y = np.arange(len(a))
    h = 0.38
    b1 = ax1.barh(y + h/2, 100 * a["hallu"], height=h, color=STATUS["critical"],
                  edgecolor=SURFACE, linewidth=1.3, zorder=3,
                  label="Ungrounded figures (%) - lower is better")
    b2 = ax1.barh(y - h/2, 100 * (a["judge"] - 1) / 4, height=h, color=CAT[0],
                  edgecolor=SURFACE, linewidth=1.3, zorder=3,
                  label="LLM judge score (normalised %) - higher is better")
    for i, (hv, jv) in enumerate(zip(a["hallu"], a["judge"])):
        ax1.text(100 * hv + 1.2, i + h/2, f"{100*hv:.1f}%", va="center",
                 fontsize=8.5, color=INK, fontweight="bold")
        ax1.text(100 * (jv - 1) / 4 + 1.2, i - h/2, f"{jv:.2f}/5", va="center",
                 fontsize=8.5, color=INK, fontweight="bold")
    ax1.set_yticks(y); ax1.set_yticklabels(a.index, fontsize=9)
    ax1.set_xlabel("Percent")
    ax1.grid(axis="x"); ax1.set_axisbelow(True); ax1.set_xlim(0, 105)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    viz.titled(ax1, "The judge does not penalise fabrication",
               "Worst-grounded model also scores highest with the LLM judge")

    # --- panel 2: what you actually pay for grounding --------------------
    ax2.scatter(a["usd"], 100 * a["hallu"], s=150, color=CAT[0], alpha=0.65,
                edgecolor=SURFACE, linewidth=1.6, zorder=3)
    for name, r in a.iterrows():
        ax2.annotate(name, (r["usd"], 100 * r["hallu"]), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8.5, color=INK,
                     fontweight="bold")
    # Pareto frontier: cheapest model at each level of grounding quality.
    front = a.sort_values("usd")
    best = np.inf; px, py = [], []
    for _, r in front.iterrows():
        if r["hallu"] < best:
            best = r["hallu"]; px.append(r["usd"]); py.append(100 * r["hallu"])
    ax2.step(px, py, where="post", color=STATUS["good"], linewidth=2.0,
             alpha=0.8, zorder=2, label="Pareto frontier")
    ax2.set_xlabel("List-price cost for all 6 briefs (USD)")
    ax2.set_ylabel("Ungrounded figures (%)")
    ax2.legend(loc="upper right")
    ax2.margins(0.18)
    viz.titled(ax2, "Cost buys very little grounding",
               "Cheapest model is mid-pack on accuracy; the dearest is the worst")
    return fig


def fig_routing(routing: pd.DataFrame):
    """Which model is assigned to which audience, and why."""
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    y = np.arange(len(routing))
    bars = ax.barh(y, routing["composite"], height=0.6, color=CAT[0],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for i, r in enumerate(routing.itertuples()):
        ax.text(r.composite + 0.008, i, f"{r.model}   ({r.composite:.3f})",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(routing["persona_name"], fontsize=9.5)
    ax.set_xlabel("Composite score of the winning model")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_xlim(0, routing["composite"].max() * 1.55)
    viz.titled(ax, "Per-audience model routing",
               "Different audiences are won by different models - directional at n=6, not conclusive")
    return fig
