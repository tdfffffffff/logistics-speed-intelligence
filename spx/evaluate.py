"""Scoring the models on five axes, three of them fully automated.

The point of this module is to replace "this output reads nicely" with numbers.
Choosing a model for operational reporting is a production decision, and the
axis that matters most -- whether the figures are real -- happens to be the one
that can be measured without any human judgement at all.

  1. grounding    automated   share of cited figures traceable to the fact pack
  2. coverage     semi-auto   recall against a hand-labelled golden set
  3. reliability  automated   do repeated runs agree on the findings?
  4. persona fit  LLM judge   tone, structure, actionability (bias-checked)
  5. cost/latency automated   list-price USD per report, p50 seconds

The hand-labelling in (2) is deliberate: deciding which findings a good brief
*must* contain is the analyst's job, not the model's, and writing that list down
is the part of the reasoning worth showing.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spx.guardrails import (check_abstention, check_grounding,
                            check_unsupported_metrics)


# --------------------------------------------------------------- golden set
# Ten findings established by the deterministic analysis. Each carries keyword
# alternatives so a brief can be credited for expressing the finding in its own
# words rather than matching a fixed phrase. `weight` marks how central the
# finding is; `personas` marks who genuinely needs it -- a Data Engineering
# brief is not penalised for omitting the campaign staffing rule.
@dataclass
class GoldenFinding:
    key: str
    description: str
    keywords: list[list[str]]      # outer = AND, inner = OR
    weight: float
    personas: list[str]


GOLDEN_SET: list[GoldenFinding] = [
    GoldenFinding("campaign_lag",
        "BWT degrades 1-2 days after a campaign volume spike (peak lag r=0.886)",
        [["campaign", "sale", "spike", "peak", "surge"],
         ["after", "lag", "following", "subsequent", "later", "day 1", "day 2"]],
        1.0, ["management", "capacity", "ops"]),
    GoldenFinding("island_effect",
        "Island-crossing lanes are the dominant delay driver (4.76d vs 1.35d intra-region)",
        [["island", "sea", "crossing", "sabah", "sulawesi", "sumatra", "ferry"]],
        1.0, ["management", "ops", "vendor", "capacity"]),
    GoldenFinding("apt_flat",
        "APT is flat (~0.76d) everywhere, so delay is transit-side not seller-side",
        [["apt", "preparation"], ["flat", "stable", "consistent", "unchanged",
                                  "transit", "not the seller", "downstream"]],
        1.0, ["management", "ops", "vendor"]),
    GoldenFinding("simpsons",
        "Naive provider ranking is confounded by lane mix; lane-matched ranking differs",
        [["lane-matched", "matched", "like-for-like", "controlling", "confound",
          "mix", "adjusted", "same lanes"]],
        1.0, ["vendor", "management"]),
    GoldenFinding("sicepat_real",
        "SiCepat is genuinely slowest even after matching (+0.913d) and is degrading",
        [["sicepat"], ["slow", "worst", "degrad", "deteriorat", "0.91", "5.16"]],
        0.8, ["vendor", "ops", "management"]),
    GoldenFinding("ninjavan_volatile",
        "Ninja Van is average on speed but produces ~half of severe anomalies",
        [["ninja"], ["anomal", "volatil", "incident", "spike", "erratic",
                     "inconsistent", "reliab"]],
        0.8, ["ops", "vendor"]),
    GoldenFinding("unmapped_mnar",
        "Unmapped-region rows are systematically slower - missingness is informative",
        [["unknown", "unmapped", "missing", "null"],
         ["slower", "not at random", "informative", "bias", "understate", "5.1"]],
        0.9, ["bi", "dataeng"]),
    GoldenFinding("apt_bwt_violation",
        "32 rows violate APT <= BWT, which is logically impossible",
        [["apt"], ["exceed", "greater", "impossible", "violat", "32"]],
        0.9, ["bi", "dataeng"]),
    GoldenFinding("zero_qty",
        "29 rows have zero parcels and must be excluded from row-level ratios",
        [["zero", "0 parcel", "29"], ["divide", "division", "exclude", "quarantin",
                                      "ratio", "null"]],
        0.6, ["bi", "dataeng"]),
    GoldenFinding("domestic_only",
        "All lanes are domestic - the route grain from the brief is degenerate",
        [["domestic", "same country", "no cross-border", "buyer_country equals",
          "within country", "not cross-border"]],
        0.6, ["bi", "management", "dataeng"]),
]


def score_coverage(text: str, persona_key: str) -> dict:
    """Recall against the golden findings that this persona actually needs."""
    low = text.lower()
    relevant = [g for g in GOLDEN_SET if persona_key in g.personas]
    hits, missed = [], []
    for g in relevant:
        # Every AND-group must be satisfied by at least one of its OR-terms.
        if all(any(k in low for k in group) for group in g.keywords):
            hits.append(g.key)
        else:
            missed.append(g.key)
    total_w = sum(g.weight for g in relevant) or 1.0
    got_w = sum(g.weight for g in relevant if g.key in hits)
    return {"coverage": got_w / total_w, "found": hits, "missed": missed,
            "n_relevant": len(relevant)}


# ------------------------------------------------------------- reliability
_SENT = re.compile(r"[.!?]\s+")


def _finding_tokens(text: str) -> set[str]:
    """Content words that carry a finding: entities and numbers."""
    low = text.lower()
    ents = set(re.findall(
        r"\b(sicepat|ninja van|j&t|pos indonesia|pos malaysia|kerry|lbc|2go|dhl|fedex"
        r"|sabah|sarawak|sulawesi|sumatra|bali|riau|visayas|mindanao|luzon"
        r"|island|intra-region|inter-region|campaign|unmapped|apt|bwt|transit)\b", low))
    nums = set(re.findall(r"\d+\.\d+", low))
    return ents | nums


def score_self_consistency(texts: list[str]) -> dict:
    """Do repeated runs of the same prompt agree?

    A model that names different providers or different figures each run cannot
    be put in front of an operations team, however good any single run looks.
    Measured as mean pairwise Jaccard overlap of the entities and figures each
    run actually mentions.
    """
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return {"consistency": np.nan, "n_runs": len(texts)}
    sets = [_finding_tokens(t) for t in texts]
    scores = [len(a & b) / len(a | b) if (a | b) else 1.0
              for a, b in itertools.combinations(sets, 2)]
    return {"consistency": float(np.mean(scores)), "n_runs": len(texts),
            "pairwise": [round(s, 3) for s in scores]}


# ------------------------------------------------------------- LLM judge
JUDGE_SYSTEM = """You are grading operations briefs for fitness for a named reader.

Score each dimension 1-5 (5 best). Judge ONLY what is asked; ignore your own
view of the underlying logistics.

  persona_fit    Does the register, length and level of detail suit the reader?
  actionability  Could the reader act on Monday morning without asking a question?
  structure      Are the requested sections present and in order?
  restraint      Is it dense and curated, or padded and dumped?

Reply with ONLY a JSON object:
{"persona_fit": n, "actionability": n, "structure": n, "restraint": n, "note": "one sentence"}"""


def build_judge_prompt(persona, text: str) -> str:
    return (f"READER: {persona.audience}\n"
            f"THEIR QUESTION: {persona.question}\n"
            f"REQUIRED SECTIONS: {' | '.join(persona.sections)}\n"
            f"WORD LIMIT: {persona.max_words}\n"
            f"THINGS TO AVOID: {'; '.join(persona.avoid)}\n\n"
            f"BRIEF TO GRADE:\n---\n{text}\n---")


def parse_judge(raw: str) -> dict:
    """Pull the JSON out of a judge reply, tolerating fences and stray prose."""
    import json
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except Exception:                                        # noqa: BLE001
        return {}
    keys = ["persona_fit", "actionability", "structure", "restraint"]
    out = {k: float(d[k]) for k in keys if isinstance(d.get(k), (int, float))}
    if out:
        out["judge_mean"] = sum(out.values()) / len(out)
    out["note"] = str(d.get("note", ""))[:200]
    return out


# ------------------------------------------------------------- scorecard
def score_report(text: str, persona, allowed_numbers: set[float],
                 response=None) -> dict:
    """All the automated axes for a single generated brief."""
    g = check_grounding(text, allowed_numbers)
    cov = score_coverage(text, persona.key)
    words = len(text.split())
    row = {
        "grounded_figures": g.grounded,
        "total_figures": g.total_numbers,
        "hallucination_rate": g.hallucination_rate,
        "ungrounded_examples": ", ".join(g.ungrounded[:5]),
        "invented_metrics": ", ".join(check_unsupported_metrics(text)),
        "coverage": cov["coverage"],
        "findings_found": len(cov["found"]),
        "findings_missed": ", ".join(cov["missed"]),
        "words": words,
        # Respecting a length cap is a real requirement, not a nicety: an ops
        # brief nobody finishes reading has failed regardless of accuracy.
        "within_word_cap": words <= persona.max_words,
        "sections_present": sum(
            1 for s in persona.sections
            if re.search(re.escape(s.split("(")[0].strip()[:18]), text, re.I)),
        "sections_required": len(persona.sections),
    }
    if response is not None:
        row.update({"latency_s": response.latency_s, "usd_cost": response.usd_cost,
                    "tokens_out": response.tokens_out, "cached": response.cached})
    return row


def composite(df: pd.DataFrame, w=None) -> pd.DataFrame:
    """Blend the axes into one comparable score, per model.

    Weights are a judgement call and are stated rather than hidden: grounding
    dominates because a confident wrong number is worse than a missing insight.
    """
    w = w or {"grounding": 0.40, "coverage": 0.25, "judge": 0.20,
              "consistency": 0.15}
    d = df.copy()
    d["grounding_score"] = 1 - d["hallucination_rate"].fillna(1)
    # An invented KPI is a hard penalty: no number check can catch it and it
    # reads as authoritative.
    d.loc[d["invented_metrics"].fillna("") != "", "grounding_score"] *= 0.5
    d["judge_score"] = (d.get("judge_mean", pd.Series(np.nan, index=d.index)) - 1) / 4
    d["consistency_score"] = d.get("consistency", pd.Series(np.nan, index=d.index))

    parts, weights = [], []
    for col, key in [("grounding_score", "grounding"), ("coverage", "coverage"),
                     ("judge_score", "judge"), ("consistency_score", "consistency")]:
        if col in d:
            parts.append(d[col].astype(float))
            weights.append(w[key])
    stacked = pd.concat(parts, axis=1)
    wt = np.array(weights)
    # Renormalise over the axes actually available, so a missing judge score
    # does not silently drag every model down.
    mask = stacked.notna().values
    d["composite"] = np.where(
        mask.any(axis=1),
        np.nansum(stacked.values * wt, axis=1) / np.where(mask.any(axis=1), (mask * wt).sum(axis=1), 1),
        np.nan)
    return d
