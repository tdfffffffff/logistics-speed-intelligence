"""Natural-language querying over the cleaned data.

This is the brief's actual stated need: "communicate actionable insights to
stakeholders **without writing complex SQL for every new question**."

The loop is question -> SQL -> validate -> execute -> narrate, and the
validation step is the point. An LLM writing SQL against a production table is
only safe if something deterministic stands between the two, so every generated
query passes `guardrails.validate_sql` before it reaches the database, and the
connection itself is read-only as a second line of defence. A rejected query is
returned to the model once with the specific rule it broke; if the retry also
fails, the agent abstains rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from speedlab.guardrails import ABSTAIN_TOKEN, check_grounding, validate_sql

SCHEMA_NOTE = """TABLE deliveries  (one row per date x lane x logistics_provider)

Columns you may use:
  dt                  DATE      report date
  buyer_country       TEXT      ID | MY | PH | TH
  seller_country      TEXT      always equals buyer_country (domestic only)
  buyer_region        TEXT      destination region ('UNKNOWN' where unmapped)
  seller_region       TEXT      origin region ('UNKNOWN' where unmapped)
  logistics_provider  TEXT      third-party carrier name
  parcel_qty          BIGINT    parcels delivered  (USE AS THE WEIGHT)
  sum_apt             DOUBLE    total seller preparation days
  sum_bwt             DOUBLE    total buyer waiting days
  lane                TEXT      'seller_region > buyer_region'
  lane_class          TEXT      Intra-region | Inter-region (land) | Island-crossing | Unmapped
  is_intra_region     BOOLEAN
  is_island_crossing  BOOLEAN
  has_unknown_endpoint BOOLEAN  true where a region failed to map
  week_block          TEXT      W1..W5 (fixed 7-day blocks from the first date)
  day_of_week         TEXT
  is_weekend          BOOLEAN
  is_campaign_day     BOOLEAN   detected volume spike (recurring demand pulse; cause not established)
  days_from_campaign  INTEGER   signed days to nearest volume spike (negative = before)
  avg_BWT             DOUBLE    per-row rate - do NOT average this, see below
  avg_APT             DOUBLE    per-row rate - do NOT average this
  avg_transit         DOUBLE    avg_BWT - avg_APT

MANDATORY METRIC RULES:
  avg_BWT  =  SUM(sum_bwt) / SUM(parcel_qty)
  avg_APT  =  SUM(sum_apt) / SUM(parcel_qty)
  These are PARCEL-WEIGHTED. Never write AVG(sum_bwt/parcel_qty) or AVG(avg_BWT):
  that weights a 10-parcel row the same as a 40,000-parcel row and is wrong.

NOT IN THIS DATA (answer INSUFFICIENT_DATA if asked): on-time rate, SLA, cost,
revenue, customer satisfaction, driver or hub counts, failed-delivery rate,
return rate, any cross-border lane."""

SQL_SYSTEM = f"""You translate operations questions into a single DuckDB SELECT query.

{SCHEMA_NOTE}

Rules:
- Output ONLY the SQL. No prose, no markdown fences, no explanation.
- One statement. SELECT or WITH only. Never modify data.
- Always apply the parcel-weighted formulas above for speed metrics.
- Add a sensible ORDER BY and LIMIT.
- If the question cannot be answered from these columns, output exactly:
  {ABSTAIN_TOKEN}"""

NARRATE_SYSTEM = """You explain a query result to an operations audience.

Rules:
- Cite ONLY numbers present in the RESULT table. Never add outside figures.
- 2-4 sentences. Lead with the answer, then the most useful detail.
- Name the operational implication if there is one; otherwise stop.
- If the result is empty, say so plainly."""


@dataclass
class AgentTurn:
    question: str
    sql: str = ""
    sql_valid: bool = False
    guard_reason: str = ""
    repaired: bool = False
    abstained: bool = False
    result: pd.DataFrame | None = None
    narration: str = ""
    error: str = ""
    model_failed: bool = False      # provider unreachable, NOT a guardrail block
    attempts: list[dict] = field(default_factory=list)


class SQLAgent:
    """LLM writes the SQL; deterministic code decides whether it may run."""

    def __init__(self, df: pd.DataFrame, client, model_key: str,
                 row_limit: int = 500):
        # A read-only connection over an in-memory copy: even if every static
        # check were bypassed, there is nothing here to damage.
        self.con = duckdb.connect(":memory:")
        self.con.register("_src", df)
        self.con.execute("CREATE TABLE deliveries AS SELECT * FROM _src")
        # Compatibility view: cached model responses generated against the
        # previous table name still execute unchanged.
        self.con.execute("CREATE VIEW spx AS SELECT * FROM deliveries")
        self.con.execute("SET enable_external_access=false")
        self.columns = set(df.columns)
        self.client = client
        self.model_key = model_key
        self.row_limit = row_limit

    def _generate(self, question: str, feedback: str | None = None) -> tuple[str, str]:
        """Returns (sql, provider_error). An empty sql with an error set means
        the model was never reached -- which is NOT the same as the guardrail
        rejecting a query, and must not be reported as one."""
        user = f"QUESTION: {question}"
        if feedback:
            user += (f"\n\nYour previous query was REJECTED by the safety layer:\n"
                     f"{feedback}\n\nWrite a corrected query that satisfies that rule.")
        r = self.client.complete(self.model_key, SQL_SYSTEM, user,
                                 temperature=0.0, max_tokens=600)
        if not r.ok:
            return "", (r.error or "the model returned no content")
        if not (r.text or "").strip():
            return "", "the model returned an empty response"
        # Models often wrap SQL in fences despite instructions.
        return r.text.replace("```sql", "").replace("```", "").strip(), ""

    def ask(self, question: str) -> AgentTurn:
        turn = AgentTurn(question=question)

        for attempt in range(2):
            sql, provider_error = self._generate(
                question, turn.guard_reason if attempt else None)
            if provider_error:
                # The model could not be reached. Distinct from a guardrail
                # block: nothing was validated, because nothing was produced.
                turn.error = provider_error
                turn.model_failed = True
                return turn
            if ABSTAIN_TOKEN in sql.upper():
                turn.abstained = True
                turn.sql = ABSTAIN_TOKEN
                turn.narration = ("The agent declined: this question cannot be "
                                  "answered from the available columns.")
                turn.attempts.append({"attempt": attempt + 1, "sql": sql,
                                      "verdict": "abstained"})
                return turn

            verdict = validate_sql(sql, self.columns, row_limit=self.row_limit)
            turn.attempts.append({"attempt": attempt + 1, "sql": sql,
                                  "verdict": "allowed" if verdict.allowed else verdict.reason})
            turn.sql, turn.guard_reason = sql, verdict.reason

            if verdict.allowed:
                turn.sql_valid, turn.sql = True, verdict.sql
                turn.repaired = attempt > 0
                try:
                    turn.result = self.con.execute(verdict.sql).df()
                except Exception as e:                       # noqa: BLE001
                    # A query can be statically valid but fail at runtime.
                    turn.guard_reason = f"execution error: {type(e).__name__}: {e}"
                    turn.sql_valid = False
                    continue
                turn.narration = self._narrate(question, turn.result)
                return turn

        turn.abstained = True
        turn.narration = ("The agent could not produce a query that passed the "
                          "safety layer, so it declined rather than guess.")
        return turn

    def _narrate(self, question: str, result: pd.DataFrame) -> str:
        table = result.head(25).to_markdown(index=False)
        r = self.client.complete(
            self.model_key, NARRATE_SYSTEM,
            f"QUESTION: {question}\n\nRESULT:\n{table}",
            temperature=0.0, max_tokens=400)
        return r.text.strip() if r.ok else "(narration unavailable)"


# Demo questions of rising difficulty. The last two are traps: one asks for a
# metric that does not exist, the other invites the unweighted-average mistake.
DEMO_QUESTIONS = [
    "What is the overall average buyer waiting time?",
    "Which logistics provider is slowest, and by how much?",
    "Show the five slowest lanes with at least 500,000 parcels.",
    "How does buyer waiting time change in the two days after a campaign day?",
    "Compare island-crossing lanes against same-region lanes for each country.",
    "Which provider has degraded most between the first and last week?",
    "What is our on-time delivery rate by provider?",          # trap: no such column
    "What is the average of avg_BWT across all rows?",         # trap: unweighted average
]
