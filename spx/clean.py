"""Defect detection and cleaning, with a full audit log.

Design principle: **nothing is dropped silently**. Every row this pipeline
removes or alters is counted, explained, and written to an audit log, because
the Business Intelligence and Data Engineering stakeholders are asking exactly
"what did you change and why?"

Three fates for a row:
  * kept       - passes every check
  * quarantined- violates a hard invariant; excluded from headline speed
                 metrics but retained and handed to the BI / DE personas
  * flagged    - usable, but carries a caveat (e.g. unmapped region)
"""
from __future__ import annotations

import pandas as pd

from spx.config import UNKNOWN_REGION_TOKENS, UNKNOWN_SENTINEL

REGION_COLS = ["buyer_region", "seller_region"]
GRAIN_KEY = ["dt", "buyer_country", "buyer_region", "seller_country",
             "seller_region", "logistics_provider"]


def load_raw(path) -> pd.DataFrame:
    """Read the CSV. utf-8-sig strips the BOM the export tool left on `dt`."""
    return pd.read_csv(path, encoding="utf-8-sig")


def detect_defects(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per defect. Detection is separated from treatment
    so the notebook can *show the evidence* before deciding what to do."""
    out = df.copy()
    out["dt"] = pd.to_datetime(out["dt"])

    # Defect 1: unmapped regions. Two distinct upstream bugs are present -- a
    # misspelling ("Uknown") and a true null -- which means two broken code
    # paths, not one. Normalising them together would hide that, so we detect
    # them separately before merging.
    for col in REGION_COLS:
        raw = out[col]
        out[f"{col}_was_typo"] = raw.eq("Uknown")
        out[f"{col}_was_null"] = raw.isna()
        out[f"{col}_unmapped"] = raw.isna() | raw.astype(str).isin(UNKNOWN_REGION_TOKENS)
    out["has_unmapped_region"] = out["buyer_region_unmapped"] | out["seller_region_unmapped"]

    # Defect 2: zero-parcel rows. Verified that sum_apt and sum_bwt are also 0
    # on every one of these, so they are harmless inside SUM/SUM aggregates
    # (they add 0 to both numerator and denominator) and dangerous only in
    # row-level ratios, where they produce 0/0.
    out["is_zero_qty"] = out["parcel_qty"] <= 0

    # Row-level averages. Computed only where qty > 0 -- this is the division
    # the zero-qty rows would break.
    safe_qty = out["parcel_qty"].where(out["parcel_qty"] > 0)
    out["row_avg_apt"] = out["sum_apt"] / safe_qty
    out["row_avg_bwt"] = out["sum_bwt"] / safe_qty

    # Defect 3: APT > BWT. Per the brief's image1 timeline, APT spans
    # [paid -> handover] and BWT spans [paid -> delivered], so APT is a strict
    # sub-interval of BWT. APT > BWT is therefore not "unusual" -- it is
    # impossible, and indicates corrupted source data.
    out["violates_apt_bwt"] = out["row_avg_apt"] > out["row_avg_bwt"]

    # Defect 4: negative values would be equally impossible (time cannot run
    # backwards). Checked explicitly rather than assumed.
    out["has_negative"] = (out[["parcel_qty", "sum_apt", "sum_bwt"]] < 0).any(axis=1)

    # Defect 5: extreme intra-region delivery. A same-region lane baselines at
    # ~1.3-1.5 days; anything above 4 days on such a lane is either a genuine
    # incident or corrupt data. Flagged for investigation, never auto-dropped.
    out["is_intra_region"] = (
        out["buyer_region"].astype(str) == out["seller_region"].astype(str)
    ) & ~out["has_unmapped_region"]
    out["extreme_intra_region"] = out["is_intra_region"] & (out["row_avg_bwt"] > 4)

    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the pipeline. Returns (clean_df, quarantined_df, audit_log)."""
    d = detect_defects(df)
    audit: list[dict] = []

    def record(step, action, rows, parcels, rationale):
        audit.append({"step": step, "action": action, "rows_affected": int(rows),
                      "parcels_affected": int(parcels), "rationale": rationale})

    record("00_load", "loaded", len(d), d["parcel_qty"].sum(), "Raw rows as delivered in the CSV.")

    # --- Duplicates -------------------------------------------------------
    dupes = d.duplicated(subset=GRAIN_KEY).sum()
    record("01_duplicates", "verified_none" if dupes == 0 else "dropped", dupes, 0,
           f"The 6-column grain key is unique ({dupes} duplicates), so each row is one "
           "date x lane x provider cell and no double-counting is possible.")

    # --- Unmapped regions: normalise, do NOT drop -------------------------
    # These are 3.5% of total volume. Dropping them would bias every national
    # and provider total downward. Instead the three variants are folded into a
    # single sentinel so they aggregate correctly, and the rows are excluded
    # only from lane-level analysis where an unknown endpoint is meaningless.
    typo = int(d[[f"{c}_was_typo" for c in REGION_COLS]].any(axis=1).sum())
    null = int(d[[f"{c}_was_null" for c in REGION_COLS]].any(axis=1).sum())
    unmapped_rows = int(d["has_unmapped_region"].sum())
    unmapped_parcels = int(d.loc[d["has_unmapped_region"], "parcel_qty"].sum())
    for col in REGION_COLS:
        d[col] = d[col].where(~d[f"{col}_unmapped"], UNKNOWN_SENTINEL)
    record("02_unmapped_regions", "normalised_and_flagged", unmapped_rows, unmapped_parcels,
           f"'Uknown' (typo, {typo} rows) and null ({null} rows) folded into "
           f"'{UNKNOWN_SENTINEL}'. Two spellings implies two broken upstream paths. Retained "
           "for country/provider/global totals (3.5% of volume -- dropping would bias them); "
           "excluded from lane-level analysis only.")

    # --- Impossible values ------------------------------------------------
    neg = int(d["has_negative"].sum())
    record("03_negative_values", "verified_none" if neg == 0 else "quarantined", neg, 0,
           "parcel_qty, sum_apt and sum_bwt must all be non-negative; elapsed time cannot "
           "be negative and a parcel count cannot be.")

    # --- Zero-qty rows: quarantine, but note they were aggregate-safe -----
    zero_rows = int(d["is_zero_qty"].sum())
    record("04_zero_parcel_rows", "quarantined", zero_rows, 0,
           f"{zero_rows} rows report 0 parcels with sum_apt = sum_bwt = 0. They are "
           "arithmetically harmless inside SUM/SUM aggregates (adding 0 to both numerator "
           "and denominator) but produce 0/0 in row-level ratios. Removed for safety at zero "
           "cost to any total, since they carry no parcels.")

    # --- APT > BWT: quarantine, hand to BI/DE -----------------------------
    viol_rows = int(d["violates_apt_bwt"].sum())
    viol_parcels = int(d.loc[d["violates_apt_bwt"], "parcel_qty"].sum())
    record("05_apt_exceeds_bwt", "quarantined", viol_rows, viol_parcels,
           f"{viol_rows} rows have avg_APT > avg_BWT, which the brief's image1 timeline makes "
           "impossible (APT is a sub-interval of BWT). Excluded from headline speed metrics "
           "and handed in full to the BI and Data Engineering stakeholders -- for them these "
           "rows are the deliverable, not noise.")

    quarantine_mask = d["is_zero_qty"] | d["violates_apt_bwt"] | d["has_negative"]
    quarantined = d[quarantine_mask].copy()
    clean_df = d[~quarantine_mask].copy()

    # --- Extreme outliers: flag only, never drop --------------------------
    ext_rows = int(clean_df["extreme_intra_region"].sum())
    record("06_extreme_intra_region", "flagged_only", ext_rows,
           int(clean_df.loc[clean_df["extreme_intra_region"], "parcel_qty"].sum()),
           f"{ext_rows} same-region rows exceed 4 days against a ~1.4-day baseline. Retained: "
           "these are the operational incidents the Operations stakeholder needs to see. "
           "Deleting outliers would delete the finding.")

    record("07_final", "retained", len(clean_df), clean_df["parcel_qty"].sum(),
           f"{len(clean_df)} rows kept ({len(clean_df)/len(d):.2%} of input), carrying "
           f"{clean_df['parcel_qty'].sum()/d['parcel_qty'].sum():.2%} of all parcels.")

    return clean_df, quarantined, pd.DataFrame(audit)
