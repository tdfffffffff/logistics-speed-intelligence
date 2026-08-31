"""Pseudonymisation before anything leaves the machine.

Provider-level performance data is commercially sensitive: it feeds contract
negotiations, and several of these carriers compete with each other. Sending
"SiCepat is 0.91 days slower than its rivals on the same lanes" to a third-party
API is a disclosure decision, not a technical detail.

The approach here is to send the *shape* of the problem and keep the *names*
local. The model reasons over LP_A and R_03; the mapping back to real names
never leaves this process. That preserves analytical quality -- the model needs
the relationships, not the trade names -- while removing the commercially
identifying part of the payload.

Worth stating plainly: this is pseudonymisation, not anonymisation. Anyone with
the underlying data could re-identify a carrier from its volume profile. It
lowers exposure; it does not eliminate it.
"""
from __future__ import annotations

import re


class Pseudonymiser:
    """Deterministic, reversible name mapping with a stable ordering.

    Codes are assigned by descending parcel volume rather than alphabetically,
    so the same carrier gets the same code across runs and the mapping does not
    churn when a new provider appears mid-month.
    """

    def __init__(self, providers: list[str], regions: list[str]):
        self.provider_map = {p: f"LP_{chr(65 + i)}" for i, p in enumerate(providers)}
        self.region_map = {r: f"R_{i:02d}" for i, r in enumerate(regions)}
        self._reverse = {v: k for k, v in {**self.provider_map, **self.region_map}.items()}

    def scrub(self, text: str) -> str:
        """Replace every real name with its code. Longest names first, so a
        provider like 'Pos Malaysia' is not half-matched by a shorter one."""
        out = text
        for real, code in sorted({**self.provider_map, **self.region_map}.items(),
                                 key=lambda kv: -len(kv[0])):
            out = re.sub(rf"\b{re.escape(real)}\b", code, out)
        return out

    def restore(self, text: str) -> str:
        """Map codes back to real names once the response is home."""
        out = text
        for code, real in sorted(self._reverse.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"\b{re.escape(code)}\b", real, out)
        return out

    def scrub_obj(self, obj):
        """Recursively scrub a nested dict/list structure (the fact pack)."""
        if isinstance(obj, dict):
            return {self.scrub(str(k)): self.scrub_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.scrub_obj(v) for v in obj]
        if isinstance(obj, str):
            return self.scrub(obj)
        return obj

    @classmethod
    def from_frame(cls, df):
        """Build from data, ordering codes by volume so they are stable."""
        prov = (df.groupby("logistics_provider")["parcel_qty"].sum()
                  .sort_values(ascending=False).index.tolist())
        regions = sorted(set(df["buyer_region"].astype(str)) |
                         set(df["seller_region"].astype(str)))
        return cls(prov, regions)


# Patterns that must never reach a third-party API, regardless of pseudonyms.
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL_REDACTED]"),
    (re.compile(r"\+?\d[\d\s-]{8,}\d"), "[PHONE_REDACTED]"),
    (re.compile(r"\bTRK\d{6,}\b", re.I), "[TRACKING_REDACTED]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Strip anything that looks like personal data. Returns (clean, hits).

    This dataset is pre-aggregated and contains no PII, so in normal operation
    this finds nothing. It is here because the same pipeline pointed at a
    row-level parcel table would find plenty, and a guardrail added after the
    first leak is a guardrail added too late.
    """
    hits = []
    out = text
    for pattern, replacement in _PII_PATTERNS:
        found = pattern.findall(out)
        if found:
            hits.extend(str(f) for f in found)
            out = pattern.sub(replacement, out)
    return out, hits
