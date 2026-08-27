"""Visual theme for the prototype: Shopee-family palette, typography, interactions.

Colours follow Shopee's brand family (the SPX Express orange `#EE4D2D` as the
primary action colour) so the tool reads as an internal Shopee product rather
than a generic dashboard.

One deliberate exception: **the charts do not use this palette.** Brand orange
against the reserved status ramp would make "warning" and "primary action"
indistinguishable, and chart colour has to encode meaning rather than identity.
The charts keep their own validated, colourblind-checked palette (`spx/viz.py`);
the brand appears in the chrome around them.
"""
from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------- palette
SHOPEE_ORANGE = "#EE4D2D"     # primary action
SHOPEE_DARK   = "#D0011B"     # pressed / emphasis
SHOPEE_TINT   = "#FFF3F0"     # subtle fill behind orange elements
INK           = "#1F1F1F"
INK_2         = "#5C5C5C"
INK_MUTED     = "#8C8C8C"
LINE          = "#EDEDED"
SURFACE       = "#FFFFFF"
CANVAS        = "#F6F6F6"
GOOD          = "#1BAF7A"
WARN          = "#F0A020"
BAD           = "#D0011B"

# Gradio's theme system wants a full 50..950 ramp for its primary hue.
ORANGE_RAMP = gr.themes.Color(
    c50="#FFF5F2", c100="#FFE7E0", c200="#FFC9BA", c300="#FFA48D",
    c400="#F97A57", c500=SHOPEE_ORANGE, c600="#DC401F", c700="#BC3115",
    c800="#96250F", c900="#711B0B", c950="#4D1207",
)


def shopee_theme() -> gr.themes.Base:
    """Base theme; fine-grained interaction states come from CSS below."""
    return gr.themes.Soft(
        primary_hue=ORANGE_RAMP,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "SFMono-Regular", "monospace"],
    ).set(
        body_background_fill=CANVAS,
        background_fill_primary=SURFACE,
        block_background_fill=SURFACE,
        block_border_width="1px",
        block_border_color=LINE,
        block_radius="12px",
        block_shadow="0 1px 2px rgba(16,24,40,.04)",
        block_label_text_weight="600",
        block_title_text_weight="600",
        button_primary_background_fill=SHOPEE_ORANGE,
        button_primary_background_fill_hover=SHOPEE_DARK,
        button_primary_text_color="#FFFFFF",
        button_large_radius="10px",
        button_small_radius="8px",
        input_border_color=LINE,
        input_border_color_focus=SHOPEE_ORANGE,
        input_radius="10px",
        table_radius="10px",
    )


# Interaction states Gradio's theme tokens do not reach: hover lifts, focus
# rings, row highlighting, pressable cards, and the animated "working" state.
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --spx-orange:{SHOPEE_ORANGE}; --spx-dark:{SHOPEE_DARK}; --spx-tint:{SHOPEE_TINT};
  --spx-ink:{INK}; --spx-ink2:{INK_2}; --spx-muted:{INK_MUTED};
  --spx-line:{LINE}; --spx-good:{GOOD}; --spx-warn:{WARN}; --spx-bad:{BAD};
  --spx-ease: cubic-bezier(.2,.7,.3,1);
}}

* {{ font-feature-settings:"cv02","cv03","cv04","tnum"; }}
body, .gradio-container {{ font-family:'Inter',system-ui,-apple-system,sans-serif !important; }}
footer {{ display:none !important; }}

/* ---------------------------------------------------------- masthead */
.spx-masthead {{
  background:linear-gradient(103deg, {SHOPEE_ORANGE} 0%, #F86A3F 52%, #FF8B5E 100%);
  color:#fff; padding:20px 26px; border-radius:14px; margin-bottom:6px;
  box-shadow:0 3px 12px rgba(238,77,45,.16);
}}
.spx-title {{ font-size:1.5rem; font-weight:700; letter-spacing:-.02em; line-height:1.2; }}
.spx-sub   {{ font-size:.9rem; opacity:.94; margin-top:5px; max-width:64ch; }}
.spx-pill {{
  display:inline-flex; align-items:center; gap:7px; margin-top:11px;
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.34);
  padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600;
  backdrop-filter:blur(6px); transition:background .18s var(--spx-ease);
}}
.spx-pill:hover {{ background:rgba(255,255,255,.28); }}
.spx-dot {{
  width:7px; height:7px; border-radius:50%; background:#fff;
  box-shadow:0 0 0 0 rgba(255,255,255,.7); animation:spx-pulse 2.2s infinite;
}}
@keyframes spx-pulse {{
  0%   {{ box-shadow:0 0 0 0 rgba(255,255,255,.65); }}
  70%  {{ box-shadow:0 0 0 9px rgba(255,255,255,0); }}
  100% {{ box-shadow:0 0 0 0 rgba(255,255,255,0); }}
}}

/* ------------------------------------------------- hover-lift on blocks */
.block, .form {{ transition:box-shadow .2s var(--spx-ease), border-color .2s var(--spx-ease); }}
.block:hover {{ border-color:#E0E0E0; box-shadow:0 2px 8px rgba(16,24,40,.05); }}

/* ------------------------------------------------------------ buttons */
button.primary, .gr-button-primary {{
  font-weight:600 !important; letter-spacing:.01em;
  transition:transform .12s var(--spx-ease), box-shadow .18s var(--spx-ease),
             background .18s var(--spx-ease) !important;
  box-shadow:0 2px 8px rgba(238,77,45,.26);
}}
button.primary:hover {{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(238,77,45,.34); }}
button.primary:active {{ transform:translateY(0) scale(.985); box-shadow:0 1px 4px rgba(238,77,45,.3); }}
button:focus-visible {{ outline:2px solid var(--spx-orange); outline-offset:2px; }}
/* Kill the dark focus/selected ring Gradio applies after a click. Focus stays
   visible for keyboard users via :focus-visible above. */
button:focus:not(:focus-visible), .block:focus-within,
.gr-box:focus-within, .selected {{
  outline:none !important; box-shadow:none !important;
}}
button.spx-qbtn:focus {{ outline:none !important; box-shadow:none !important; }}

/* --------------------------------------------------------------- tabs */
.tab-nav button {{
  font-weight:600 !important; color:var(--spx-ink2) !important;
  border:none !important; border-bottom:2px solid transparent !important;
  transition:color .16s var(--spx-ease), border-color .16s var(--spx-ease),
             background .16s var(--spx-ease) !important;
  border-radius:8px 8px 0 0 !important;
}}
.tab-nav button:hover {{ color:var(--spx-orange) !important; background:var(--spx-tint) !important; }}
.tab-nav button.selected {{
  color:var(--spx-orange) !important; border-bottom-color:var(--spx-orange) !important;
  background:transparent !important;
}}

/* -------------------------------------------------------- table rows */
.gr-dataframe table tbody tr, table tbody tr {{ transition:background .13s var(--spx-ease); }}
.gr-dataframe table tbody tr:hover, table tbody tr:hover {{ background:var(--spx-tint) !important; }}
.gr-dataframe table thead th, table thead th {{
  background:#FAFAFA !important; font-weight:600 !important;
  color:var(--spx-ink2) !important; font-size:.8rem !important;
  text-transform:uppercase; letter-spacing:.04em;
  position:sticky; top:0; z-index:2;
}}
.gr-dataframe td, table td {{ font-variant-numeric:tabular-nums; }}

/* ---------------------------------------------------- example chips */
.gr-samples-table td, .gr-examples td {{
  cursor:pointer; border-radius:8px !important;
  transition:background .15s var(--spx-ease), color .15s var(--spx-ease),
             transform .12s var(--spx-ease);
}}
.gr-samples-table td:hover, .gr-examples td:hover {{
  background:var(--spx-tint) !important; color:var(--spx-dark) !important;
  transform:translateX(2px);
}}

/* -------------------------------------------------------- verdict card */
.spx-verdict {{
  border-radius:10px; padding:12px 15px; font-size:.87rem; font-weight:600;
  border-left:4px solid var(--spx-muted); background:#FAFAFA; line-height:1.5;
  transition:background .2s var(--spx-ease), border-color .2s var(--spx-ease);
}}
.spx-verdict.pass  {{ border-left-color:var(--spx-good); background:#F1FBF7; color:#0E6B4B; }}
.spx-verdict.block {{ border-left-color:var(--spx-bad);  background:#FEF2F3; color:#8E1220; }}
.spx-verdict.abst  {{ border-left-color:var(--spx-warn); background:#FFF9EE; color:#8A5B08; }}
/* Provider unreachable: deliberately NEUTRAL, not red. A red badge would read
   as "the guardrail rejected this", which is the opposite of what happened. */
.spx-verdict.err {{
  border-left-color:var(--spx-muted); background:#F4F4F5; color:#3F3F46; font-weight:500;
}}
.spx-errdetail {{
  display:block; margin-top:6px; font-family:'JetBrains Mono',monospace;
  font-size:.72rem; color:var(--spx-ink2); word-break:break-word; line-height:1.45;
}}

/* ------------------------------------------------------- metric cards */
.spx-cards {{ display:flex; gap:11px; flex-wrap:wrap; margin:4px 0 2px; }}
.spx-card {{
  flex:1 1 128px; background:#fff; border:1px solid var(--spx-line);
  border-radius:11px; padding:12px 14px; cursor:default;
  transition:transform .16s var(--spx-ease), box-shadow .16s var(--spx-ease),
             border-color .16s var(--spx-ease);
}}
.spx-card:hover {{
  transform:translateY(-2px); border-color:#E2E2E2;
  box-shadow:0 3px 10px rgba(16,24,40,.07);
}}
.spx-card .k {{
  font-size:.68rem; text-transform:uppercase; letter-spacing:.06em;
  color:var(--spx-muted); font-weight:600;
}}
.spx-card .v {{ font-size:1.34rem; font-weight:700; color:var(--spx-ink); margin-top:3px;
                font-variant-numeric:tabular-nums; }}
.spx-card .n {{ font-size:.72rem; color:var(--spx-ink2); margin-top:2px; }}
.spx-card.good .v {{ color:var(--spx-good); }}
.spx-card.warn .v {{ color:var(--spx-warn); }}
.spx-card.bad  .v {{ color:var(--spx-bad); }}

/* ------------------------------------------------------ tooltip on hover */
.spx-help {{ position:relative; border-bottom:1px dotted var(--spx-muted); cursor:help; }}
.spx-help::after {{
  content:attr(data-tip); position:absolute; left:0; bottom:calc(100% + 8px);
  background:var(--spx-ink); color:#fff; padding:8px 11px; border-radius:8px;
  font-size:.76rem; font-weight:500; line-height:1.45; white-space:normal;
  width:max-content; max-width:290px; opacity:0; pointer-events:none;
  transform:translateY(4px); transition:opacity .16s var(--spx-ease), transform .16s var(--spx-ease);
  z-index:60; box-shadow:0 4px 14px rgba(16,24,40,.14);
}}
.spx-help:hover::after {{ opacity:1; transform:translateY(0); }}

/* --------------------------------------------------------- brief body */
.spx-brief {{ line-height:1.62; font-size:.94rem; }}
.spx-brief h1,.spx-brief h2,.spx-brief h3 {{
  color:var(--spx-ink); font-weight:700; margin-top:1.05em; font-size:1.02rem;
}}
.spx-brief strong {{ color:var(--spx-dark); }}
.spx-brief li {{ margin:.28em 0; }}

/* ------------------------------------------- question guide + traps */
.spx-trapnote {{
  background:#FFF9EE; border:1px solid #F5E2BC; border-left:3px solid var(--spx-warn);
  border-radius:10px; padding:11px 14px; font-size:.86rem; line-height:1.55;
  color:#7A5206; margin:2px 0 8px;
}}

/* Question cards are real buttons. Two lines: the tag, then the question. */
button.spx-qbtn {{
  display:block !important; width:100%; text-align:left !important;
  white-space:pre-line !important; line-height:1.4 !important;
  font-size:.82rem !important; font-weight:400 !important;
  background:#fff !important; color:var(--spx-ink) !important;
  border:1px solid var(--spx-line) !important; border-radius:10px !important;
  padding:9px 11px !important; margin-bottom:7px !important;
  box-shadow:none !important;
  transition:border-color .16s var(--spx-ease), background .16s var(--spx-ease),
             transform .13s var(--spx-ease) !important;
}}
button.spx-qbtn::first-line {{
  font-size:.62rem; font-weight:700; letter-spacing:.06em; color:var(--spx-muted);
}}
button.spx-qbtn:hover {{
  border-color:var(--spx-orange) !important; background:var(--spx-tint) !important;
  transform:translateY(-1px);
}}
button.spx-qbtn:active {{ transform:translateY(0); }}
button.spx-qbtn.trap {{ background:#FFFBF4 !important; border-color:#F0D9A8 !important; }}
button.spx-qbtn.trap::first-line {{ color:#B8791A; }}
button.spx-qbtn.trap:hover {{
  border-color:var(--spx-warn) !important; background:#FFF6E6 !important;
}}

/* ------------------------------------------------- pending / loading */
/* Gradio darkens a primary button to near-red while a call is in flight.
   Mid-demo that reads as "something failed", so the busy state is a lighter,
   obviously-working orange instead. */
button.primary[disabled], button.primary:disabled,
button.primary.pending, .gr-button-primary[disabled] {{
  background:#F79070 !important; color:#fff !important; opacity:1 !important;
  cursor:progress !important; box-shadow:none !important; transform:none !important;
}}
.progress-text, .eta-bar {{ background:var(--spx-orange) !important; }}

/* Rule-level explanation inside a verdict card. */
.spx-rule {{
  display:block; margin-top:7px; font-weight:400; font-size:.79rem;
  line-height:1.5; color:inherit; opacity:.92;
}}
.spx-rule code {{
  background:rgba(0,0,0,.06); padding:1px 4px; border-radius:4px;
  font-family:'JetBrains Mono',monospace; font-size:.74rem;
}}

/* ------------------------------------------------------------- misc */
.spx-note {{ font-size:.8rem; color:var(--spx-ink2); line-height:1.55; }}
.prose a {{ color:var(--spx-orange); }}
::selection {{ background:rgba(238,77,45,.18); }}
"""
