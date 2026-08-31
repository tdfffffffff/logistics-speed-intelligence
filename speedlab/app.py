"""The working prototype: a Gradio UI over the whole pipeline.

Built so it runs for someone with NO API keys. Every tab falls back to cached
responses, and the masthead says plainly which mode it is in -- a demo that
fails on the reviewer's machine demonstrates nothing.

Two design choices worth naming:

* **The Ask tab shows its working.** The generated SQL, the guardrail verdict
  and the result table are all on screen, not just the final sentence. An
  operations user who cannot see how an answer was produced has no way to know
  when to distrust it.
* **Grounding is surfaced, not buried.** Every generated brief carries a live
  count of how many of its figures were verified against the fact pack. The
  point of the whole system is that this number is visible.
"""
from __future__ import annotations

import json
import html as _html

import gradio as gr
import pandas as pd

from speedlab import config
from speedlab.analysis import detect_anomalies
from speedlab.evaluate import score_report
from speedlab.factpack import collect_numbers
from speedlab.guardrails import check_grounding
from speedlab.llm import MODELS, LLMClient, available_models
from speedlab.personas import PERSONAS, build_system_prompt, build_user_prompt
from speedlab.privacy import Pseudonymiser
from speedlab.sqlagent import DEMO_QUESTIONS, SQLAgent

# Each demo question paired with what it is testing, so a first-time user
# understands *why* the last two are supposed to be refused. Without this the
# traps look like the tool failing rather than the tool working.
QUESTION_GUIDE = [
    (DEMO_QUESTIONS[0], "warm-up", "A single global figure. Checks the parcel-weighted formula is used."),
    (DEMO_QUESTIONS[1], "ranking", "Needs a GROUP BY plus an ordering."),
    (DEMO_QUESTIONS[2], "filtering", "Adds a volume threshold and a row limit."),
    (DEMO_QUESTIONS[3], "time logic", "Requires the campaign-offset column, not a raw date."),
    (DEMO_QUESTIONS[4], "comparison", "Two lane classes side by side, per country."),
    (DEMO_QUESTIONS[5], "change over time", "Needs period-over-period reasoning."),
    (DEMO_QUESTIONS[6], "TRAP - refusing = passing",
     "UNANSWERABLE. There is no on-time / SLA / delivery-status column, so this "
     "cannot be computed - yet it is the most natural question an ops manager "
     "would ask, which is why an ungrounded model invents '94.2%' and it ends up "
     "on a slide. PASS = amber ABSTAIN. FAIL = any number at all."),
    (DEMO_QUESTIONS[7], "TRAP - the SQL is the test",
     "ANSWERABLE, but the phrasing baits AVG(avg_BWT) - the unweighted average, "
     "which lets a 10-parcel row count the same as a 40,000-parcel one and makes "
     "the network look faster than it is. PASS = green, WITH SUM(...)/SUM(...) in "
     "the SQL. FAIL = red BLOCKED, because the guardrail rejects AVG() over a "
     "speed ratio before it can run."),
]
from speedlab.theme import CSS, shopee_theme


def _cards(items) -> str:
    """Metric cards. `items` = (label, value, note, tone, tooltip)."""
    out = ["<div class='sl-cards'>"]
    for label, value, note, tone, tip in items:
        lab = (f"<span class='sl-help' data-tip=\"{_html.escape(tip)}\">{label}</span>"
               if tip else label)
        out.append(
            f"<div class='sl-card {tone}'><div class='k'>{lab}</div>"
            f"<div class='v'>{value}</div><div class='n'>{note}</div></div>")
    out.append("</div>")
    return "".join(out)


def build_demo():
    # Load .env HERE, not just in the notebook. Without this a standalone
    # `python app.py` sees no API keys, falls back to the on-device 1.5B model,
    # and every question takes ~16 minutes -- which looks like the app hanging.
    from speedlab.env import load_env
    load_env(verbose=False)

    df = pd.read_parquet(config.CLEAN_PARQUET)
    pack = json.load(open(config.ROOT / "data" / "clean" / "fact_pack.json"))
    allowed = collect_numbers(pack)
    ps = Pseudonymiser.from_frame(df)
    client = LLMClient()
    anomalies = detect_anomalies(df)

    avail = available_models()
    # A model is only genuinely usable here if it can answer interactively.
    # The local model "works" without a key but takes ~16 minutes per question,
    # so it must never be the default for an interactive tool.
    remote_live = [k for k, v in avail.items()
                   if v and MODELS[k]["provider"] != "hf_local"]
    live = [k for k, v in avail.items() if v]
    default_model = ("gemini-3.1-flash-lite" if "gemini-3.1-flash-lite" in remote_live
                     else (remote_live[0] if remote_live else list(MODELS)[0]))

    def _label(k):
        if MODELS[k]["provider"] == "hf_local":
            return f"{k}  (on-device, ~16 min per answer)"
        return k if avail.get(k) else f"{k}  (no API key)"

    model_choices = [(_label(k), k) for k in
                     sorted(MODELS, key=lambda k: (MODELS[k]["provider"] == "hf_local",
                                                   not avail.get(k), k))]
    mode_txt = (f"Live &middot; {len(remote_live)} of {len(avail)-1} API models reachable"
                if remote_live else
                "Demo mode &middot; no API keys &middot; replaying cached responses")

    try:
        routing = pd.read_csv(config.EVAL / "routing_decision.csv")
        route_map = dict(zip(routing.persona, routing.model))
    except Exception:
        route_map = {}

    agent = SQLAgent(df, client, default_model)
    h = pack["headline"]

    # ------------------------------------------------------------- handlers
    def do_ask(question, model_key):
        agent.model_key = model_key
        turn = agent.ask(question)
        if getattr(turn, "model_failed", False):
            # The provider never answered. Reporting this as a guardrail block
            # would be wrong and, in a demo, actively misleading.
            cls = "err"
            short = _html.escape(turn.error[:220])
            hint = ("This model's free-tier quota is spent. Pick another from the "
                    "Model dropdown &mdash; several are usually available."
                    if "429" in turn.error or "RESOURCE_EXHAUSTED" in turn.error
                    or "quota" in turn.error.lower() or "rate limit" in turn.error.lower()
                    else "Check the model is reachable, or pick another from the dropdown.")
            msg = (f"MODEL UNREACHABLE &mdash; the guardrail never ran, because no query "
                   f"was produced.<br><span class='sl-errdetail'>{short}</span>"
                   f"<br><b>{hint}</b>")
        elif turn.abstained:
            cls, msg = "abst", "ABSTAINED &mdash; not answerable from the available columns"
        elif turn.sql_valid:
            cls = "pass"
            msg = "PASSED the guardrail" + (" &middot; repaired after 1 rejection"
                                            if turn.repaired else "")
            # Name the specific rule that was satisfied. "Passed" on its own is
            # ambiguous -- it reads as though the query slipped past a check,
            # when in fact the model wrote the correct weighted form and there
            # was nothing to object to.
            import re as _re
            sql_l = (turn.sql or "").lower()
            if _re.search(r"sum\s*\(\s*sum_(bwt|apt)\s*\)\s*/\s*sum\s*\(\s*parcel_qty\s*\)", sql_l):
                msg += ("<br><span class='sl-rule'>Weighted-average rule <b>satisfied</b>: the model "
                        "wrote <code>SUM(&hellip;)/SUM(parcel_qty)</code>, not <code>AVG(&hellip;)</code>. "
                        "Had it written the unweighted form, this query would have been blocked "
                        "before execution.</span>")
            elif _re.search(r"\bavg\s*\(", sql_l):
                msg += ("<br><span class='sl-rule'>Note: <code>AVG()</code> is used here, but not over a "
                        "speed ratio &mdash; the weighted-average rule does not apply to this query.</span>")
        else:
            cls, msg = "block", f"BLOCKED &mdash; {_html.escape(turn.guard_reason[:150])}"
        verdict = f"<div class='sl-verdict {cls}'>{msg}</div>"

        result = turn.result if turn.result is not None else pd.DataFrame({"(no rows)": []})
        rows = len(turn.result) if turn.result is not None else 0
        cards = _cards([
            ("Rows returned", f"{rows}", "capped at 500", "", ""),
            ("Guardrail",
             "not run" if getattr(turn, "model_failed", False)
             else ("pass" if turn.sql_valid else ("abstain" if turn.abstained else "block")),
             f"{len(turn.attempts)} attempt(s)",
             "" if getattr(turn, "model_failed", False)
             else ("good" if turn.sql_valid else ("warn" if turn.abstained else "bad")),
             "Every generated query is statically checked - read-only, in-schema, "
             "and parcel-weighted - before it reaches the database."),
            ("Model", model_key, MODELS[model_key]["vendor"], "", ""),
        ])
        trace = "\n\n".join(f"# attempt {a['attempt']}: {a['verdict']}\n{a['sql']}"
                            for a in turn.attempts) or "(none)"
        sql_panel = turn.sql or ("-- the model was not reached, so no SQL exists to show"
                                 if getattr(turn, "model_failed", False)
                                 else "-- no SQL produced")
        narration = (turn.narration
                     or (f"No answer: {turn.error}" if turn.error else "-"))
        return sql_panel, verdict, cards, result, narration, trace

    def do_brief(persona_key, model_key, use_routing):
        p = PERSONAS[persona_key]
        routed = route_map.get(persona_key) if (use_routing and route_map) else None
        mk = routed or model_key
        r = client.complete(mk, build_system_prompt(p),
                            build_user_prompt(p, ps.scrub_obj(pack)),
                            temperature=0.0, max_tokens=1400)

        # If the routed model is rate-limited or keyless, fall back to the model
        # the user can actually see in the dropdown and say so -- rather than
        # reporting a failure for a model they never chose.
        fell_back = False
        if not r.ok and routed and model_key != routed:
            r = client.complete(model_key, build_system_prompt(p),
                                build_user_prompt(p, ps.scrub_obj(pack)),
                                temperature=0.0, max_tokens=1400)
            if r.ok:
                mk, fell_back = model_key, True

        if not r.ok:
            return (f"### Could not generate this brief\n\n"
                    f"`{mk}` returned:\n\n```\n{r.error}\n```\n\n"
                    f"Free-tier quotas reset periodically. Pick another model from the dropdown "
                    f"— several are usually available.", "", "")
        text = ps.restore(r.text)
        s = score_report(text, p, allowed, r)
        pct = 100 * (1 - s["hallucination_rate"])
        cards = _cards([
            ("Figures verified", f"{s['grounded_figures']}/{s['total_figures']}",
             f"{pct:.0f}% grounded",
             "good" if pct >= 98 else ("warn" if pct >= 90 else "bad"),
             "Every number in the brief is extracted and matched against the fact pack. "
             "A figure counts as grounded only if a source value rounds to it at the "
             "precision written."),
            ("Golden-set coverage", f"{100*s['coverage']:.0f}%",
             f"{s['findings_found']} findings hit", "",
             "Recall against ten findings, hand-labelled from the deterministic analysis, "
             "that this audience genuinely needs."),
            ("Length", f"{s['words']}", f"cap {p.max_words}",
             "good" if s["within_word_cap"] else "warn", ""),
            ("Latency", f"{r.latency_s:.1f}s", "cached" if r.cached else f"${r.usd_cost:.5f}", "", ""),
        ])
        flags = s["ungrounded_examples"] or "none"
        origin = ("auto-selected by the routing table" if routed and not fell_back
                  else (f"fell back from {routed}, which was unavailable" if fell_back
                        else "chosen manually"))
        note = (f"<div class='sl-note'><b>Model:</b> {mk} &middot; {MODELS[mk]['vendor']} "
                f"&middot; {MODELS[mk]['access']} &middot; {origin}"
                f"<br><b>Unverified figures:</b> {_html.escape(str(flags))}</div>")
        return f"<div class='sl-brief'>\n\n{text}\n\n</div>", cards, note

    def persona_hint(persona_key, use_routing=True):
        p = PERSONAS[persona_key]
        routed = route_map.get(persona_key) if route_map else None
        extra = (f"<br><b>Routing table picks:</b> {routed} "
                 f"&mdash; untick “Use routing table” to force the model on the right instead."
                 if (use_routing and routed) else "")
        return (f"<div class='sl-note'><b>{p.audience}</b> &mdash; “{_html.escape(p.question)}”<br>"
                f"Sections: {' &middot; '.join(p.sections)}{extra}</div>")

    def do_anoms(n, provider, direction):
        a = anomalies
        if provider != "All":
            a = a[a.logistics_provider == provider]
        if direction != "All":
            a = a[a.direction == direction.lower()]
        cols = ["dt", "lane", "logistics_provider", "parcel_qty", "avg_BWT",
                "lane_baseline_bwt", "mad_z", "excess_parcel_days"]
        out = a.head(int(n))[cols].copy()
        out["dt"] = out["dt"].dt.strftime("%Y-%m-%d")
        out = out.round(2).rename(columns={
            "dt": "Date", "lane": "Lane", "logistics_provider": "Provider",
            "parcel_qty": "Parcels", "avg_BWT": "avg_BWT",
            "lane_baseline_bwt": "Lane baseline", "mad_z": "Deviation (z)",
            "excess_parcel_days": "Excess parcel-days"})
        cards = _cards([
            ("Flagged", f"{len(a)}", f"of {len(df[df.parcel_qty>=5000]):,} eligible cells",
             "bad" if len(a) > 80 else "warn",
             "Modified z-score against each lane's own baseline. Threshold set by alert "
             "budget (~3/day), not by normal-theory significance."),
            ("Slower", f"{int((a.direction=='slower').sum())}", "incidents", "bad", ""),
            ("Faster", f"{int((a.direction=='faster').sum())}", "unexpected gains", "good", ""),
            ("Post-campaign", f"{100*a.days_from_campaign.isin([1,2]).mean():.0f}%",
             "land 1-2 days after a sale", "warn",
             "Independent confirmation of the lag found by correlation analysis."),
        ])
        return cards, out

    def do_guardrails():
        log = pd.DataFrame(client.log)
        if log.empty:
            return _cards([("No calls yet", "-", "use the other tabs first", "", "")]), pd.DataFrame()
        cards = _cards([
            ("Calls", f"{len(log)}", "this session", "", ""),
            ("From cache", f"{int(log.cached.sum())}",
             f"{100*log.cached.mean():.0f}% replayed", "good",
             "Cached responses make the notebook and this app reproducible without keys."),
            ("Failures", f"{int((~log.ok).sum())}", "degraded gracefully",
             "warn" if (~log.ok).any() else "good", ""),
            ("List-price cost", f"${log.usd_cost.sum():.5f}", "free tiers priced at list", "",
             "Actual spend on a free tier is $0. List prices are shown so models stay comparable."),
        ])
        return cards, client.cost_report().round(5)

    # ------------------------------------------------------------------ UI
    with gr.Blocks(title="Speed Intelligence", fill_width=True) as demo:
        gr.HTML(
            f"""<div class='sl-masthead'>
              <div class='sl-title'>Speed Intelligence</div>
              <div class='sl-sub'>Ask in plain English. Get audience-specific operations briefs
              backed by specific evidence, with every figure verified against the source data.</div>
              <div class='sl-pill'><span class='sl-dot'></span>{mode_txt}</div>
            </div>""")
        gr.HTML(_cards([
            ("Buyer waiting", f"{h['avg_BWT_days']}d", "avg_BWT, month", "",
             "SUM(sum_bwt) / SUM(parcel_qty) - parcel-weighted, as the brief defines it."),
            ("Preparation", f"{h['avg_APT_days']}d", f"{h['apt_share_of_bwt_pct']}% of total", "good",
             "Seller-controlled. Flat across every provider, region and date."),
            ("Network transit", f"{h['avg_transit_days']}d", "3PL-controlled", "warn",
             "avg_BWT minus avg_APT. This is where essentially all the variation sits."),
            ("Parcels", f"{h['total_parcels']/1e6:.0f}M", "30 days, 4 markets", "", ""),
        ]))

        # ---------------------------------------------------------- tab 1
        with gr.Tab("Ask"):
            gr.Markdown(
                "The agent writes SQL, a **deterministic guardrail** decides whether it may run, "
                "then it explains the result. The working is shown so you can see when to distrust it.")
            gr.HTML(
                "<div class='sl-trapnote'><b>Two of these are deliberate traps</b> &mdash; planted to "
                "test the two ways this system could produce a confidently wrong number. "
                "<b>They have different success signals:</b><br>"
                "&nbsp;&nbsp;• <b>“refusing = passing”</b> &mdash; the question is unanswerable, so a "
                "<span style='color:#B8791A;font-weight:600'>amber ABSTAIN</span> is the correct result.<br>"
                "&nbsp;&nbsp;• <b>“the SQL is the test”</b> &mdash; the question <i>is</i> answerable, so "
                "<span style='color:#0E6B4B;font-weight:600'>green PASS</span> is correct "
                "<i>provided the SQL uses</i> <code>SUM(…)/SUM(…)</code>. Watch the query, not the badge.<br>"
                "Hover any card for what it tests.</div>")
            with gr.Row():
                q = gr.Textbox(label="Question", value=DEMO_QUESTIONS[1], lines=2, scale=4,
                               placeholder="e.g. which lanes got slower after the last campaign?")
                m1 = gr.Dropdown(model_choices, value=default_model, label="Model", scale=1)
            # Real buttons, not static HTML: the previous version rendered
            # pretty cards that did nothing when clicked, with a duplicate row
            # of working chips underneath. One control, and it works.
            q_buttons = []
            with gr.Row():
                for col in range(4):
                    with gr.Column(scale=1, min_width=210):
                        for text, tag, why in QUESTION_GUIDE[col::4]:
                            is_trap = "TRAP" in tag
                            b = gr.Button(
                                f"{tag.upper()}\n{text}",
                                elem_classes=["sl-qbtn"] + (["trap"] if is_trap else []),
                                size="sm", variant="secondary")
                            q_buttons.append((b, text))
            for b, text in q_buttons:
                b.click(lambda t=text: t, None, q)

            ask_btn = gr.Button("Ask", variant="primary", size="lg")
            ask_cards = gr.HTML()
            with gr.Row():
                sql_out = gr.Code(label="Generated SQL", language="sql", scale=3)
                verdict_out = gr.HTML(label="Guardrail verdict")
            res_out = gr.Dataframe(label="Result", wrap=True, max_height=340)
            narr_out = gr.Textbox(label="Narration", lines=3)
            with gr.Accordion("Full attempt trace", open=False):
                trace_out = gr.Code(language="sql")
            ask_btn.click(do_ask, [q, m1],
                          [sql_out, verdict_out, ask_cards, res_out, narr_out, trace_out])
            q.submit(do_ask, [q, m1],
                     [sql_out, verdict_out, ask_cards, res_out, narr_out, trace_out])

        # ---------------------------------------------------------- tab 2
        with gr.Tab("Stakeholder brief"):
            gr.Markdown("The same verified facts, written for six different readers. "
                        "Hover any metric label for what it measures.")
            with gr.Row():
                pk = gr.Dropdown([(p.name, k) for k, p in PERSONAS.items()],
                                 value="ops", label="Audience", scale=2)
                m2 = gr.Dropdown(model_choices, value=default_model, label="Model", scale=2)
                use_r = gr.Checkbox(value=bool(route_map), label="Use routing table", scale=1)
            hint = gr.HTML(persona_hint("ops"))
            pk.change(persona_hint, [pk, use_r], hint)
            use_r.change(persona_hint, [pk, use_r], hint)
            brief_btn = gr.Button("Generate brief", variant="primary", size="lg")
            brief_cards = gr.HTML()
            brief_out = gr.Markdown()
            note_out = gr.HTML()
            brief_btn.click(do_brief, [pk, m2, use_r], [brief_out, brief_cards, note_out])

        # ---------------------------------------------------------- tab 3
        with gr.Tab("Anomaly console"):
            gr.Markdown("Incidents are found **statistically**, not by the LLM — a modified z-score "
                        "against each lane's own baseline. Deterministic and auditable; the model is "
                        "only ever asked to explain what this has already flagged.")
            with gr.Row():
                n = gr.Slider(5, 80, value=15, step=5, label="Rows")
                prov = gr.Dropdown(["All"] + sorted(df.logistics_provider.unique()),
                                   value="All", label="Provider")
                dirn = gr.Dropdown(["All", "Slower", "Faster"], value="All", label="Direction")
            anom_cards = gr.HTML()
            anom_out = gr.Dataframe(wrap=True, max_height=460)
            for ctl in (n, prov, dirn):          # live filtering, no button needed
                ctl.change(do_anoms, [n, prov, dirn], [anom_cards, anom_out])
            demo.load(do_anoms, [n, prov, dirn], [anom_cards, anom_out])

        # ---------------------------------------------------------- tab 4
        with gr.Tab("Guardrails & cost"):
            gr.Markdown("Every model call this session: tokens, latency and list-price cost.")
            g_cards = gr.HTML()
            g_tbl = gr.Dataframe(wrap=True)
            gr.Button("Refresh", variant="primary").click(do_guardrails, None, [g_cards, g_tbl])
            demo.load(do_guardrails, None, [g_cards, g_tbl])

    return demo


LAUNCH_KW = dict(css=CSS, theme=shopee_theme())


def launch(share: bool = False, **kw):
    """Start the UI. Used from the notebook and from app.py alike."""
    return build_demo().launch(share=share, **{**LAUNCH_KW, **kw})


if __name__ == "__main__":
    launch()
