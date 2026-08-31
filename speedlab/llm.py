"""One interface over several LLM vendors, with caching, costing and logging.

Design goals, in priority order:

1. **The notebook must render for a reviewer with no API keys.** Every response
   is content-hashed and cached to disk, so a re-run replays the exact outputs
   instead of failing or silently producing different text. This is why the
   cache is committed with the deliverable.
2. **A dead vendor must not break the run.** A missing key, a rate limit or a
   network error marks that model "unavailable" and the comparison proceeds
   with the rest, rather than aborting nine models because the tenth 429'd.
3. **Cost must be comparable.** Several of these models are used on free tiers,
   where actual spend is $0 and a cost-quality trade-off would be meaningless.
   Every model is therefore priced at its published *list* rate, clearly
   labelled as list-price-equivalent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from speedlab.config import CACHE_DIR

# --------------------------------------------------------------- registry
# usd_in / usd_out are per 1M tokens, at published list prices. `tier` records
# how the model was actually accessed, so the notebook can be honest about the
# difference between what was paid and what is being compared.
# Model IDs were verified against each provider's live catalogue rather than
# assumed -- several plausible-looking names (gemini-2.5-pro, llama-3.3-70b,
# deepseek-r1-distill) are simply not available on these accounts. See the
# notebook's provider-availability section for what was tried and rejected.
#
# The roster deliberately spans three *access modes*, because that is the axis
# an operations team actually has to choose along:
#   managed API   - best quality, data leaves the building
#   open weights  - same models, third-party inference, still off-premise
#   local         - nothing leaves the machine, quality cost is measurable
MODELS: dict[str, dict] = {
    # --- Google, managed API (free tier) --------------------------------
    "gemini-3.6-flash": dict(
        vendor="Google", provider="gemini", model="gemini-3.6-flash",
        usd_in=0.30, usd_out=2.50, tier="free (AI Studio)",
        access="managed API", weight_class="frontier-flash"),
    "gemini-2.5-flash": dict(
        vendor="Google", provider="gemini", model="gemini-2.5-flash",
        usd_in=0.30, usd_out=2.50, tier="free (AI Studio)",
        access="managed API", weight_class="previous-gen"),
    "gemini-3.1-flash-lite": dict(
        vendor="Google", provider="gemini", model="gemini-3.1-flash-lite",
        usd_in=0.10, usd_out=0.40, tier="free (AI Studio)",
        access="managed API", weight_class="small"),

    # --- OpenAI open-weight models, served by Groq ----------------------
    "gpt-oss-120b": dict(
        vendor="OpenAI (open weights)", provider="groq", model="openai/gpt-oss-120b",
        usd_in=0.15, usd_out=0.75, tier="free (Groq)",
        access="open weights", weight_class="large",
        token_multiplier=1.6, reasoning_effort="low"),
    "gpt-oss-20b": dict(
        vendor="OpenAI (open weights)", provider="groq", model="openai/gpt-oss-20b",
        usd_in=0.10, usd_out=0.50, tier="free (Groq)",
        access="open weights", weight_class="mid",
        token_multiplier=1.6, reasoning_effort="low"),

    # --- Alibaba open weights -------------------------------------------
    "qwen3.6-27b": dict(
        vendor="Alibaba", provider="groq", model="qwen/qwen3.6-27b",
        usd_in=0.29, usd_out=0.59, tier="free (Groq)",
        access="open weights", weight_class="reasoning", token_multiplier=2.0),

    # Same model with reasoning disabled. Kept as a separate roster entry so
    # the bake-off answers a real question -- does a reasoning mode help when
    # the analysis is already done and the job is to WRITE it up? -- rather
    # than silently picking one configuration. Note the mechanism: Qwen's
    # documented "/no_think" prompt switch does NOT work on Groq's build (it
    # still emitted <think> and used more tokens); the API-level
    # reasoning_effort parameter does, cutting output ~8x.
    "qwen3.6-27b-nothink": dict(
        vendor="Alibaba", provider="groq", model="qwen/qwen3.6-27b",
        usd_in=0.29, usd_out=0.59, tier="free (Groq)",
        access="open weights", weight_class="reasoning-off",
        reasoning_effort="none"),

    # --- Fully local: the on-premise baseline ----------------------------
    "qwen2.5-1.5b-local": dict(
        vendor="Alibaba (local)", provider="hf_local",
        model="Qwen/Qwen2.5-1.5B-Instruct", usd_in=0.0, usd_out=0.0,
        tier="on-premise, no data egress",
        access="local", weight_class="tiny"),
}

# Attempted and unavailable on these accounts. Recorded rather than deleted:
# "which models could we actually get?" is part of the engineering result, and
# the OpenAI outcome is directly relevant to the privacy discussion.
UNAVAILABLE = {
    "gemini-2.5-pro": "404 - retired from the AI Studio catalogue",
    "gemini-2.5-flash-lite": "404 - retired from the AI Studio catalogue",
    "gemini-3.1-pro-preview": "429 - Pro tier not included in the free quota",
    "gemini-3.7-flash": (
        "429 RESOURCE_EXHAUSTED partway through evaluation - the newest Flash "
        "model has a much smaller free-tier daily quota, and it returned 503s "
        "intermittently before exhausting it. Dropped rather than reported on "
        "partial data, since a model evaluated on 4 of 6 personas is not "
        "comparable with one evaluated on all 6."),
    "gpt-4.1-mini (OpenAI direct)": (
        "429 insufficient_quota - the key has no billing credit. OpenAI's free "
        "tier is contingent on opting in to prompts being used for training, "
        "which is precisely the trade-off the privacy section examines. "
        "OpenAI's models are still represented in the bake-off via the "
        "open-weight gpt-oss family served by Groq."),
    "llama-3.3-70b / deepseek-r1-distill": "404 - not in this Groq catalogue",
}

ENV_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY", "hf_local": None}


_THINK_CLOSED = re.compile(r"<(think|reasoning|thinking)>.*?</\1>\s*", re.S | re.I)
# An UNCLOSED block means the model ran out of tokens mid-reasoning and never
# reached its answer. Everything from the opening tag on is then discarded, and
# the empty result is what signals that the token budget was too small.
_THINK_OPEN = re.compile(r"<(think|reasoning|thinking)>.*\Z", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks some open-weight models emit.

    Qwen3.6 and the gpt-oss family wrap their reasoning in <think> tags. That
    text is not the deliverable -- and if left in, it wrecks the evaluation:
    the grounding check would score every figure the model considered and
    discarded, and the word-count cap would fail on reasoning rather than on
    the brief itself.
    """
    out = _THINK_CLOSED.sub("", text or "")
    out = _THINK_OPEN.sub("", out)
    return out.strip()


@dataclass
class LLMResponse:
    model_key: str
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    usd_cost: float = 0.0
    cached: bool = False
    ok: bool = True
    error: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["vendor"] = MODELS.get(self.model_key, {}).get("vendor", "?")
        return d


def _cache_key(model_key: str, system: str, user: str, temperature: float) -> str:
    """Content hash over everything that affects the output."""
    blob = json.dumps([model_key, system, user, temperature], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def available_models() -> dict[str, bool]:
    """Which models can actually be called right now."""
    out = {}
    for key, spec in MODELS.items():
        env = ENV_KEYS[spec["provider"]]
        out[key] = True if env is None else bool(os.getenv(env))
    return out


class LLMClient:
    """Uniform `.complete()` across vendors, with a disk cache in front."""

    def __init__(self, cache_dir=CACHE_DIR, offline: bool = False):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # offline=True forces cache-only: used to prove the notebook renders
        # for a reviewer with no keys.
        self.offline = offline
        self.log: list[dict] = []
        self._clients: dict[str, object] = {}

    # ------------------------------------------------------------ vendors
    def _gemini(self):
        if "gemini" not in self._clients:
            from google import genai
            self._clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._clients["gemini"]

    def _groq(self):
        if "groq" not in self._clients:
            from groq import Groq
            self._clients["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])
        return self._clients["groq"]

    def _openai(self):
        if "openai" not in self._clients:
            from openai import OpenAI
            self._clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._clients["openai"]

    def _hf_local(self, model_id):
        """Load the on-premise model onto the best local device.

        Deliberately NOT device_map="auto": for a 1.5B model that path decided
        to offload weights to disk on this machine, and generation slowed to
        the point of being unusable (no persona completed in 13 minutes). An
        explicit device plus fp16 keeps the whole model resident -- ~3GB, which
        Apple unified memory handles comfortably.
        """
        if "hf_local" not in self._clients:
            import torch
            from transformers import pipeline
            if torch.backends.mps.is_available():
                device, dtype = "mps", torch.float16
            elif torch.cuda.is_available():
                device, dtype = "cuda", torch.float16
            else:
                device, dtype = "cpu", torch.float32
            self._clients["hf_local"] = pipeline(
                "text-generation", model=model_id, device=device, dtype=dtype)
        return self._clients["hf_local"]

    # ------------------------------------------------------------- call
    def _call_vendor(self, spec, system, user, temperature, max_tokens):
        prov = spec["provider"]
        if prov == "gemini":
            from google.genai import types
            # Gemini 2.5+ counts hidden "thinking" tokens against
            # max_output_tokens. Measured: with the default budget the model
            # spent 477 of 500 tokens thinking and emitted a 19-token,
            # mid-sentence brief. Thinking is switched off here because the
            # reasoning has already been done -- the fact pack IS the analysis,
            # and the model's job is to write it up for a named reader.
            cfg = dict(system_instruction=system, temperature=temperature,
                       max_output_tokens=max_tokens)
            try:
                r = self._gemini().models.generate_content(
                    model=spec["model"], contents=user,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0), **cfg))
            except Exception:
                # Some Gemini 3 models require thinking; give them room instead.
                r = self._gemini().models.generate_content(
                    model=spec["model"], contents=user,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens * 4,
                        **{k: v for k, v in cfg.items() if k != "max_output_tokens"}))
            u = r.usage_metadata
            return (r.text or ""), (u.prompt_token_count or 0), (u.candidates_token_count or 0)

        if prov in ("groq", "openai"):
            client = self._groq() if prov == "groq" else self._openai()
            extra = {}
            if spec.get("reasoning_effort"):
                extra["reasoning_effort"] = spec["reasoning_effort"]
            r = client.chat.completions.create(
                model=spec["model"], temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}], **extra)
            return (r.choices[0].message.content or ""), r.usage.prompt_tokens, r.usage.completion_tokens

        if prov == "hf_local":
            pipe = self._hf_local(spec["model"])
            out = pipe([{"role": "system", "content": system},
                        {"role": "user", "content": user}],
                       max_new_tokens=max_tokens,
                       do_sample=temperature > 0,
                       temperature=temperature if temperature > 0 else None)
            text = out[0]["generated_text"][-1]["content"]
            # No usage metadata from a local pipeline; approximate for the log.
            return text, len(system + user) // 4, len(text) // 4

        raise ValueError(f"unknown provider {prov}")

    def complete(self, model_key: str, system: str, user: str,
                 temperature: float = 0.0, max_tokens: int = 2000,
                 retries: int = 4, use_cache: bool = True) -> LLMResponse:
        """Call a model, or replay it from cache. Never raises."""
        spec = MODELS[model_key]
        # Reasoning models burn most of their output budget on hidden thinking
        # before writing a word. Measured: qwen3.6-27b returns an EMPTY brief at
        # 300 tokens and a complete one at 1500. Scaling the cap per model keeps
        # the comparison about writing quality rather than about who happened to
        # get enough room to finish.
        max_tokens = int(max_tokens * spec.get("token_multiplier", 1.0))
        ck = _cache_key(model_key, system, user, temperature)
        path = self.cache_dir / f"{model_key}__{ck}.json"

        if use_cache and path.exists():
            d = json.loads(path.read_text())
            resp = LLMResponse(**{k: v for k, v in d.items()
                                  if k in LLMResponse.__annotations__})
            resp.cached = True
            self.log.append(resp.to_row())
            return resp

        env = ENV_KEYS[spec["provider"]]
        if self.offline:
            resp = LLMResponse(model_key, "", ok=False,
                               error="offline mode: no cached response for this prompt")
            self.log.append(resp.to_row())
            return resp
        if env and not os.getenv(env):
            resp = LLMResponse(model_key, "", ok=False,
                               error=f"{env} not set - model not evaluated")
            self.log.append(resp.to_row())
            return resp

        last = ""
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                text, tin, tout = self._call_vendor(spec, system, user,
                                                    temperature, max_tokens)
                text = strip_reasoning(text)
                resp = LLMResponse(
                    model_key=model_key, text=text, tokens_in=tin, tokens_out=tout,
                    latency_s=round(time.time() - t0, 2),
                    usd_cost=round(tin / 1e6 * spec["usd_in"] +
                                   tout / 1e6 * spec["usd_out"], 6))
                path.write_text(json.dumps(asdict(resp), indent=2))
                self.log.append(resp.to_row())
                return resp
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                msg = str(e)
                # Distinguish failures that clear on their own from ones that
                # never will. A 413 here is Groq's tokens-per-minute ceiling,
                # not an oversized payload -- verified by re-sending the same
                # request a minute later and having it succeed. Those, plus 429
                # and 503, get a long backoff; anything else is not worth
                # retrying and the model is marked unavailable immediately.
                transient = any(c in msg for c in ("429", "413", "503", "500", "502",
                                                   "overloaded", "UNAVAILABLE",
                                                   "RESOURCE_EXHAUSTED", "rate limit"))
                if attempt < retries and transient:
                    time.sleep(min(20, 4 * (attempt + 1)) + 1.0)
                elif not transient:
                    break

        resp = LLMResponse(model_key, "", ok=False, error=last[:300])
        self.log.append(resp.to_row())
        return resp

    def cost_report(self):
        """Per-model tokens, latency and list-price cost for this session."""
        import pandas as pd
        if not self.log:
            return pd.DataFrame()
        df = pd.DataFrame(self.log)
        return (df.groupby(["model_key", "vendor"], as_index=False)
                  .agg(calls=("model_key", "size"),
                       cached=("cached", "sum"),
                       failures=("ok", lambda s: (~s).sum()),
                       tokens_in=("tokens_in", "sum"),
                       tokens_out=("tokens_out", "sum"),
                       p50_latency_s=("latency_s", "median"),
                       usd_list_cost=("usd_cost", "sum"))
                  .sort_values("usd_list_cost", ascending=False))
