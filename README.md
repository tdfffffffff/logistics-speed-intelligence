# AI LLM Test: SPX Speed Intelligence

An end-to-end analysis of logistics speed across carriers, routes and time periods for Regional SPX Express,
combined with a language model layer that communicates verified findings to six operational audiences and a
working decision-support prototype.

---

## 1. Executive Summary

**Objectives:** 
- Analyse buyer waiting time and seller preparation time across four Southeast Asian markets and
ten carriers
- Identify where delays occur and why
- Communicate the findings to operational stakeholders without requiring bespoke SQL for each question

**Approach:** 
The system is organised into three layers:

1. **Deterministic analysis:** establishes and validates the underlying operational facts using pandas and SQL.
2. **Language model layer:** converts verified facts into six audience-specific briefs and answers ad-hoc
   questions in natural language.
3. **Evaluation and verification:** measures grounding, coverage, persona fit, reliability, cost and latency,
   with deterministic guardrails preventing unsupported quantitative claims.

The language model is therefore used for **interpretation and communication, not as the source of truth**.
Every reported figure is traceable either to pre-computed analysis or to a validated query executed against
the database.

### 1.1 Principal Findings

| # | Finding | Operational Implication |
|---|---|---|
| **1** | Buyer waiting time deteriorates **one to two days after campaigns**, rather than during them (lag-2 `r = 0.886`) | Campaign dates are known in advance, allowing post-campaign capacity to be planned |
| **2** | Island-crossing lanes average **4.76 days**, versus **1.35 days** for same-region lanes | The primary constraint is structural; inter-island linehaul capacity should be prioritised |
| **3** | Seller preparation time is approximately **0.76 days** across the network | Seller-side intervention is unlikely to materially improve buyer waiting time |
| **4** | Raw carrier rankings are confounded by **lane composition** | Geography must be controlled before attributing performance differences to carriers |
| **5** | Two distinct carrier failure modes emerge: **SiCepat is consistently slow; Ninja Van is volatile** | The two require different operational and commercial responses |
| **6** | Missing region data is **not missing at random**, with unmapped records slower in 18 of 18 carrier × country comparisons | Excluding these records systematically understates national delivery times |

### 1.2 Quantified Opportunity

Approximately **213 million parcel-days of buyer waiting time per month** are recoverable if each country ×
lane-class segment reaches the median segment's speed. Two island-crossing segments account for **68%** of
this opportunity.

### 1.3 Key Evaluation Finding

The language-model judge preferred the **least well-grounded model**, with higher hallucination rates
associated with higher judged quality. This result provides the empirical basis for placing deterministic
grounding and guardrails ahead of model-based judgement in the system architecture. See **Section 13.4**.

---

## 2. Setup

The notebook is designed to execute **without API credentials**. Model responses are cached, allowing the
analysis and evaluation outputs to be reproduced on a machine with no API keys.

### 2.1 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the analysis
jupyter lab TanDanFeng_regspx_eda_llm_take_home_assignment.ipynb

# 3. Optionally, run the prototype application
python app.py
# Then open http://127.0.0.1:7860
```

To regenerate model outputs live rather than replaying cached responses, copy `.env.example` to `.env` and add
the required API keys. Models without available credentials are reported as **not evaluated**, and the
remainder of the analysis proceeds unaffected.

**Requirements:** Python 3.11 or later. The complete dependency list is provided in `requirements.txt`. The
local model evaluated in Section 13.6 additionally requires `torch` and `accelerate`, and is optional.

---

## 3. Where to Find Each Deliverable

| Requirement | Location |
|---|---|
| Data cleaning and defect classification | Section 4; complete audit log in Appendix A5 |
| Exploratory data analysis | Sections 6 to 10; supplementary views in Appendix A4 |
| Language model implementation | Part II, Sections 11 to 14 |
| Generated stakeholder briefs | Section 14.1 |
| All generated briefs across models | Appendix A2 |
| Natural-language query outputs | Section 12 |
| Model evaluation and selection | Section 13; full scores in Appendix A1 and judge detail in Appendix A3 |
| Model suitability by stakeholder | Section 14 and Appendix A1 |
| Prototype application | Section 15; screenshots embedded in the notebook and screen recordings linked below |
| Generated manager brief | `SPX_Insight_Brief.docx`, produced by Section 16 |
| Methods, rationale and references | Section 18 |

### 3.1 Suggested Reading Order

**For a five-minute review:**

1. Executive Summary
2. Section 13.4: Principal Evaluation Result
3. Section 14: Model Routing

**For a detailed review:**

1. Section 4: Cleaning decisions
2. Section 6.1: Campaign lag
3. Section 8.1: Carrier ranking after controlling for lane composition
4. Section 9: Informative missingness
5. Section 11.3: Grounding verification
6. Section 13.4: Evaluation failure of the language-model judge
7. Section 14: Model routing and deployment decision

---

## 4. Directory Structure

```
TanDanFeng_regspx_eda_llm_take_home_assignment/
│
├── TanDanFeng_regspx_eda_llm_take_home_assignment.ipynb   # Main analysis
├── README.md                                              # This document
├── SPX_Insight_Brief.docx                                 # Manager brief, generated by Section 16
├── app.py                                                 # Prototype launcher
├── requirements.txt                                       # Dependencies
├── .env.example                                           # API key template
│
├── spx/
│   ├── config.py                # Paths and the assignment specification encoded as constants
│   ├── env.py                   # API key loading and availability reporting
│   ├── clean.py                 # Defect detection, classification and audit logging
│   ├── metrics.py               # Weighted metrics and aggregation grains
│   ├── features.py              # Lane taxonomy, campaign detection and baselines
│   ├── analysis.py              # Anomaly detection, confounder control and impact sizing
│   ├── factpack.py              # Fact-pack generation
│   ├── privacy.py               # Pseudonymisation before model transmission
│   ├── guardrails.py            # Grounding, SQL validation and injection screening
│   ├── personas.py              # Stakeholder registry and prompt construction
│   ├── sqlagent.py              # Natural-language querying agent
│   ├── evaluate.py              # Five-dimension evaluation framework
│   ├── llm.py                   # Multi-vendor model client and response caching
│   ├── charts.py / viz.py       # Chart construction and visualisation
│   ├── app.py / theme.py        # Prototype interface
│   └── report.py                # Automated brief generation
│
├── data/
│   ├── raw/                     # Dataset as provided
│   ├── clean/                   # Cleaned data, quarantine, audit log and fact pack
│   └── cache/llm_responses/     # Cached model responses
│
└── outputs/
    ├── figures/                 # Notebook figures
    └── eval/                    # Evaluation outputs and routing decisions
```

### 4.1 Screen Recordings

The four prototype demonstrations can be accessed via the Google Drive link below.
**Google Drive:** <https://drive.google.com/drive/folders/1jsL8rl9x59gpd51q7gDHCBDbj3RBy9AW?usp=sharing>

| File | Tab Demonstrated |
|---|---|
| `1. "Ask" Tab.mp4` | Natural-language querying, generated SQL, guardrail verdict, result table and narration |
| `2. "Stakeholder Brief" Tab.mp4` | Audience-specific brief generation and grounding report |
| `3. "Anomaly Console" Tab.mp4` | Statistically flagged incidents and filtering |
| `4. "Guardrails and Cost" Tab.mp4` | Session token consumption, latency and list-price cost |

---

## 5. Scope and Reproducibility

### 5.1 Reproducibility

All analytical results are deterministic. Model responses are content-hashed and cached, so the notebook can
reproduce the evaluated outputs without active API credentials. The application also supports a
credential-free demonstration mode using the cached responses.

### 5.2 Model Availability

Seven models were evaluated across three vendors and three deployment modes. Free-tier quotas were exhausted
during evaluation for two models. The headline model comparison is therefore restricted to models with
complete coverage across all six stakeholder briefs, since models evaluated on different subsets are not
directly comparable. This constraint is documented in Section 13.1.

### 5.3 Limitations

1. The dataset covers 30 days, representing approximately three campaign cycles. The campaign-lag finding
   therefore rests on a small number of events.
2. The campaign relationship is correlational, not causal. A confirmatory operational experiment is proposed
   in Section 17.1.
3. No service-level agreement or promised delivery date is available; performance can therefore be described
   as slow, but not formally as late.
4. No cost data is available, so the quantified opportunity cannot currently be expressed in financial terms.
5. No first-attempt or failed-delivery data is available, limiting the explanation of carrier volatility.
6. The carrier comparison controls for lane composition, but not parcel-level differences in shipment mix.
7. The stakeholder routing result is directionally supported but statistically under-powered, as discussed in
   Section 14.

The analysis therefore distinguishes between findings that are directly established by the data and
hypotheses that require further validation.
