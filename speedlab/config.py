"""Central configuration: paths, and the analysis specification encoded as data.

Every requirement lives here as a constant so that the notebook can *assert*
compliance rather than claim it.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "parcel_deliveries_202601.csv"
CLEAN_PARQUET = ROOT / "data" / "clean" / "deliveries_clean.parquet"
QUARANTINE_PARQUET = ROOT / "data" / "clean" / "deliveries_quarantine.parquet"
AUDIT_LOG = ROOT / "data" / "clean" / "cleaning_audit_log.csv"
CACHE_DIR = ROOT / "data" / "cache" / "llm_responses"
FIGURES = ROOT / "outputs" / "figures"
SCREENSHOTS = ROOT / "outputs" / "screenshots"
REPORTS = ROOT / "outputs" / "reports"
EVAL = ROOT / "outputs" / "eval"
ASSETS = ROOT / "assets"

# ------------------------------------------------- the brief's data dictionary
# The source data dictionary. Used to assert we consume every column.
DATA_DICTIONARY = {
    "dt": "date of report",
    "buyer_country": "buyer country",
    "buyer_region": "buyer country (more granular)",
    "seller_country": "seller country",
    "seller_region": "seller country (more granular)",
    "logistics_provider": "name of third party logistics",
    "parcel_qty": "number of parcels delivered",
    "sum_apt": "total actual preparation time (APT) in days: buyer paid -> seller hands parcel to 3PL",
    "sum_bwt": "total buyer waiting time (BWT) in days: buyer paid -> buyer receives order",
}

# The four aggregation grains mandated by image2 of the brief.
# `keys=None` means the global (ungrouped) grain.
SPEC_GRAINS = {
    "provider":  {"keys": ["logistics_provider"],                "source": "brief image2 example 1"},
    "route":     {"keys": ["seller_country", "buyer_country"],   "source": "brief image2 example 2"},
    "weekly":    {"keys": ["week_block"],                        "source": "brief image2 example 3"},
    "global":    {"keys": None,                                  "source": "brief image2 example 4"},
}

# APT spans [order paid -> handover to 3PL]; BWT spans [order paid -> delivered].
# image1 of the brief therefore makes APT a strict sub-interval of BWT.
INVARIANT_APT_SUBSET_OF_BWT = "avg_APT <= avg_BWT"

# Region values that mean "we could not map this region" (two upstream bugs:
# a typo variant and a null variant).
UNKNOWN_REGION_TOKENS = {"Unknown", "Uknown", "unknown", "UNKNOWN", "", "nan", "NaN", "None"}
UNKNOWN_SENTINEL = "UNKNOWN"

# Island / sea-crossing regions: lanes touching these require ferry or air legs,
# which is the dominant physical driver of transit time in this dataset.
ISLAND_REGIONS = {
    # Malaysia (East Malaysia, across the South China Sea from the peninsula)
    "Sabah", "Sarawak",
    # Indonesia (each a separate major island from Java)
    "Bali", "North Sumatra", "Riau", "South Sulawesi",
    # Philippines (Visayas / Mindanao island groups, separate from Luzon)
    "Central Visayas", "West Visayas", "Northern Mindanao", "SOCCSKSARGEN",
}

RANDOM_SEED = 42
