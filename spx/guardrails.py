"""Deterministic checks around every LLM call.

The guiding assumption is that the model is a useful but unreliable component,
so correctness is enforced *outside* it. Nothing here asks a model to check
another model: these are regexes, parsers and set membership tests, which is
why they can be trusted to police the thing that cannot be.

Four layers:
  * grounding  - is every number in the output traceable to the fact pack?
  * SQL        - is the generated query read-only, in-schema, and correctly weighted?
  * abstention - did the model refuse when the data could not answer?
  * injection  - did untrusted field text try to issue instructions?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers, with optional thousands separators, decimals, %, and M/K/bn suffixes.
_NUM = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*(%|M\b|K\b|bn\b|million|billion)?",
                  re.IGNORECASE)

# Figures a report may state without them existing in the fact pack: small
# integers used as ordinals ("top 3", "the 2 days after"), years, and 100 for
# percentages. Without this, ordinary prose reads as hallucination.
_ALLOWED_BARE = set(range(0, 32)) | {100, 1000, 2025, 2026, 2027}


@dataclass
class GroundingResult:
    total_numbers: int
    grounded: int
    ungrounded: list[str] = field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        """Share of cited figures with no support in the fact pack."""
        return 0.0 if not self.total_numbers else 1 - self.grounded / self.total_numbers

    @property
    def summary(self) -> str:
        return (f"{self.grounded}/{self.total_numbers} figures grounded "
                f"({100 * self.hallucination_rate:.1f}% unsupported)")


def _precision_of(literal: str) -> int:
    """Decimal places the model actually wrote, e.g. '3.30' -> 2, '12,400' -> 0."""
    return len(literal.split(".")[1]) if "." in literal else 0


def check_grounding(text: str, allowed: set[float], strict: bool = True
                    ) -> GroundingResult:
    """Verify every number in the output against the fact pack.

    **Matching is precision-aware, not relative-tolerance.** A cited figure is
    grounded if some value in the pack *rounds to it at the precision the model
    wrote*. So "3.3" is accepted for 3.33 (legitimate rounding) while "4.7" is
    rejected even though the pack contains 4.761 (which rounds to 4.8).

    This replaced a relative-tolerance version, which was measured and found
    unusable: at +/-2% the 306 pack values matched **54.7% of the entire 0-10
    number line**, so a fabricated "4.7 days" scored as grounded purely by
    collision. Precision matching cuts that false-negative surface to near zero
    without penalising normal rounding.

    Honest limitation: this verifies a figure *exists* in the pack, not that it
    was used in the right place. A model that correctly quotes Indonesia's BWT
    but attributes it to Thailand passes this check. Catching that needs the
    claim-level review in `check_unsupported_metrics` plus a human read.
    """
    total, grounded, bad = 0, 0, []
    for raw, suffix in _NUM.findall(text):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        s_ = (suffix or "").lower()
        prec = _precision_of(raw)

        # A model may quote a scaled or percentage form; accept either reading.
        # Each candidate carries its OWN precision: converting "91.2%" to the
        # proportion 0.912 shifts the decimal point two places, so comparing it
        # at 1dp would round it to 0.9 and collide with anything near 0.9.
        candidates = {(val, prec)}
        if s_ in ("m", "million"):
            candidates.add((val * 1e6, max(prec - 6, 0)))
        elif s_ == "k":
            candidates.add((val * 1e3, max(prec - 3, 0)))
        elif s_ in ("bn", "billion"):
            candidates.add((val * 1e9, max(prec - 9, 0)))
        elif s_ == "%":
            candidates.add((val / 100, prec + 2))

        total += 1
        if val in _ALLOWED_BARE and not s_:
            grounded += 1
            continue

        def matches(c: float, p_: int) -> bool:
            for a in allowed:
                # Round the *pack* value to the precision the model wrote.
                if round(a, p_) == round(c, p_):
                    return True
                # A proportion in the pack quoted as a percentage (0.228 -> 22.8%).
                if s_ == "%" and round(a * 100, prec) == round(val, prec):
                    return True
                # A large figure quoted in units (185266641 -> "185.3M").
                if abs(a) > 1000 and abs(a - c) <= 0.001 * abs(a):
                    return True
            return False

        if any(matches(c, p_) for c, p_ in candidates):
            grounded += 1
        else:
            bad.append(f"{raw}{suffix or ''}")
    return GroundingResult(total, grounded, bad)


# Metrics that do NOT exist in this dataset. A model inventing one of these is
# a more dangerous failure than a wrong digit, because the sentence reads as
# authoritative and no number check will catch it -- there is no such column to
# disagree with. This list is the vocabulary boundary of the fact pack.
UNSUPPORTED_METRICS = [
    "on-time delivery", "on time delivery", "otd", "sla compliance", "sla attainment",
    "first attempt", "first-attempt", "failed delivery", "return rate", "rto",
    "nps", "csat", "customer satisfaction", "complaint rate", "ticket volume",
    "cost per parcel", "cost per delivery", "revenue", "margin", "profit",
    "driver count", "fleet size", "hub count", "warehouse capacity", "utilisation",
    "lost parcel", "damage rate", "delivery attempt", "pickup success",
]


# A mention inside a negation or abstention is CORRECT behaviour, not invention.
# The model was told to answer INSUFFICIENT_DATA for metrics the dataset lacks,
# so saying "there is no on-time delivery data" must not be scored as a
# hallucination -- the first version of this check did exactly that and
# penalised the single best-behaved model in the bake-off for following its
# instructions. Context is checked at sentence level.
_NEGATION = re.compile(
    r"(INSUFFICIENT_DATA|no\s+data|not\s+available|not\s+present|not\s+in\s+the|"
    r"does\s+not\s+(contain|include|exist)|doesn't\s+(contain|include)|absent|"
    r"lacks?|unavailable|cannot\s+be|can't\s+be|missing\s+from|outside\s+the\s+scope|"
    r"no\s+such|would\s+require|not\s+captured|not\s+measured|excluded)", re.I)

# Deliberately NOT sentence-splitting on ":" -- "On-time delivery rate:
# INSUFFICIENT_DATA" would be torn into two fragments, orphaning the negation
# from the metric and re-introducing the false positive this check exists to
# avoid. A character window around the mention is robust to punctuation.
_CONTEXT_WINDOW = 180


def check_unsupported_metrics(text: str) -> list[str]:
    """Flag references to KPIs the dataset does not contain.

    The dataset has exactly three measures: parcel_qty, sum_apt, sum_bwt. A
    claim about on-time rate, cost, satisfaction or fleet size is fabricated by
    construction, however plausible it sounds -- and no numeric check can catch
    it, because there is no column for it to disagree with.

    Mentions that sit inside a negation or an INSUFFICIENT_DATA declaration are
    NOT counted: correctly telling the reader a metric is unavailable is the
    behaviour we asked for.
    """
    body = text or ""
    low = body.lower()
    flagged = []
    for metric in UNSUPPORTED_METRICS:
        for m in re.finditer(re.escape(metric), low):
            lo = max(0, m.start() - _CONTEXT_WINDOW // 2)
            hi = min(len(body), m.end() + _CONTEXT_WINDOW)
            # Look both ways: negations precede ("there is no X") and follow
            # ("X: INSUFFICIENT_DATA") the metric roughly equally often.
            if not _NEGATION.search(body[lo:hi]):
                flagged.append(metric)
                break
    return list(dict.fromkeys(flagged))



# --------------------------------------------------------------------- SQL
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|GRANT|REVOKE"
    r"|ATTACH|DETACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|PRAGMA|SET|CALL)\b", re.I)
# Ratio-of-averages: the specific mistake the brief's own formula rules out.
_UNWEIGHTED_AVG = re.compile(
    r"AVG\s*\(\s*(?:CAST\s*\()?\s*sum_(?:bwt|apt)\s*(?:AS\s+\w+\s*\))?\s*/", re.I)
_AVG_OF_DERIVED = re.compile(r"AVG\s*\(\s*(avg_BWT|avg_APT|row_avg_\w+)\s*\)", re.I)


@dataclass
class SQLVerdict:
    allowed: bool
    reason: str = "ok"
    sql: str = ""


def validate_sql(sql: str, allowed_columns: set[str], table: str = "spx",
                 row_limit: int = 500) -> SQLVerdict:
    """Static checks before the query is ever executed.

    Ordered cheapest-first, and every rejection names the rule it broke so the
    model can be given a single, specific repair instruction.
    """
    q = sql.strip().rstrip(";").strip()
    if not q:
        return SQLVerdict(False, "empty query")

    # Strip comments first -- otherwise a forbidden keyword can hide behind one.
    bare = re.sub(r"--[^\n]*|/\*.*?\*/", " ", q, flags=re.S)

    if ";" in bare:
        return SQLVerdict(False, "multiple statements are not allowed (possible chained query)")
    if not re.match(r"^\s*(SELECT|WITH)\b", bare, re.I):
        return SQLVerdict(False, "only SELECT or WITH queries are permitted")
    if _FORBIDDEN.search(bare):
        return SQLVerdict(False, f"forbidden keyword: {_FORBIDDEN.search(bare).group(1).upper()}")
    if _UNWEIGHTED_AVG.search(bare) or _AVG_OF_DERIVED.search(bare):
        return SQLVerdict(False,
            "unweighted average: the brief defines avg_BWT as SUM(sum_bwt)/SUM(parcel_qty). "
            "AVG() of a per-row ratio weights a 10-parcel row equally with a 40,000-parcel "
            "row and returns a different, wrong answer.")

    # Any identifier that looks like a column must actually exist. This catches
    # hallucinated fields (a favourite being 'on_time_rate') before execution.
    #
    # Two things must be admitted alongside the real columns or valid SQL gets
    # rejected: names the query *defines itself* (column aliases after AS, CTE
    # names, table aliases) and case variants, since SQL identifiers are
    # case-insensitive while the DataFrame columns are not.
    lower_cols = {c.lower() for c in allowed_columns}
    defined = set()
    #  ... AS alias        (column and CTE aliases both use AS)
    defined |= {m.lower() for m in re.findall(r"\bAS\s+\"?([a-z_][a-z0-9_]*)\"?", bare, re.I)}
    #  WITH name AS (...)  and  , name AS (...)
    defined |= {m.lower() for m in re.findall(r"(?:\bWITH|,)\s*([a-z_][a-z0-9_]*)\s+AS\s*\(", bare, re.I)}
    #  FROM/JOIN tbl alias (bare table alias without AS)
    defined |= {m.lower() for m in re.findall(r"\b(?:FROM|JOIN)\s+[a-z_][a-z0-9_]*\s+([a-z_][a-z0-9_]*)", bare, re.I)}

    referenced = set(re.findall(r"\b([a-z_][a-z0-9_]{2,})\b", bare.lower()))
    sql_words = {
        "select", "from", "where", "group", "order", "by", "having", "limit", "with",
        "as", "and", "or", "not", "null", "is", "in", "on", "join", "left", "right",
        "inner", "outer", "full", "case", "when", "then", "else", "end", "distinct",
        "sum", "count", "avg", "min", "max", "round", "cast", "date", "desc", "asc",
        "between", "like", "ilike", "abs", "coalesce", "nullif", "extract",
        "interval", "day", "month", "year", "week", "strftime", "date_trunc", "over",
        "partition", "rank", "row_number", "lag", "lead", "union", "all", "exists",
        "double", "integer", "varchar", "decimal", "float", "bigint", "current_date",
        "offset", "cross", "using", "filter", "percentile_cont", "median", "true",
        "stddev", "variance", "greatest", "least", "trunc", "floor", "ceil",
        "false", "case", "cume_dist", "ntile", "first", "last", "desc", table.lower(),
    }
    unknown = {r for r in referenced
               if r not in sql_words and r not in lower_cols and r not in defined}
    if unknown:
        return SQLVerdict(False, f"unknown column(s): {sorted(unknown)}. "
                                 f"Available columns: {sorted(allowed_columns)}")

    # Cap result size so a runaway query cannot flood the notebook or the UI.
    if not re.search(r"\bLIMIT\s+\d+", bare, re.I):
        q = f"{q}\nLIMIT {row_limit}"
    return SQLVerdict(True, "ok", q)


# -------------------------------------------------------------- abstention
ABSTAIN_TOKEN = "INSUFFICIENT_DATA"


def check_abstention(text: str) -> bool:
    """Did the model correctly decline instead of inventing an answer?"""
    return ABSTAIN_TOKEN in text.upper() or bool(re.search(
        r"\b(cannot|can't|unable to|not possible to|no data|not available|"
        r"does not contain|isn't in the data|not in the dataset)\b", text, re.I))


# ---------------------------------------------------------------- injection
_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"
    r"|disregard\s+(the\s+)?(system|previous)"
    r"|you\s+are\s+now\b|new\s+instructions?:"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|<\s*/?\s*(system|assistant|instructions?)\s*>)", re.I)


def scan_for_injection(value: str) -> tuple[bool, str]:
    """Screen untrusted field text before it is interpolated into a prompt.

    Region and provider names arrive from an upstream pipeline, so they are
    data, not instructions. If that pipeline is ever compromised, a crafted
    region name would otherwise be pasted straight into a system prompt.
    """
    m = _INJECTION.search(value or "")
    return (True, m.group(0)) if m else (False, "")


def sanitise_field(value: str, max_len: int = 64) -> str:
    """Neutralise a field value for safe prompt interpolation."""
    v = _INJECTION.sub("[BLOCKED]", str(value or ""))
    v = re.sub(r"[\r\n]+", " ", v)
    v = re.sub(r"[{}<>|`]", "", v)
    return v[:max_len].strip()
