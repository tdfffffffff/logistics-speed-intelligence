"""Load API keys from .env, and report what is usable.

Separated from llm.py so the notebook can show the key-availability table
before any model is called, and so a reviewer immediately sees which parts of
the comparison ran live versus replayed from cache.
"""
from __future__ import annotations

from spx.config import ROOT


def load_env(verbose: bool = True) -> dict[str, bool]:
    """Read ROOT/.env if present, then report which providers are reachable."""
    from spx.llm import ENV_KEYS, MODELS, available_models
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    avail = available_models()
    if verbose:
        live = sum(avail.values())
        print(f"LLM access: {live} of {len(avail)} models reachable\n")
        for key, ok in avail.items():
            spec = MODELS[key]
            env = ENV_KEYS[spec["provider"]] or "(no key needed)"
            print(f"  {'LIVE ' if ok else 'no   '} {key:26} {spec['vendor']:18} {env}")
        if live < len(avail):
            print("\n  Models without a key are reported as 'not evaluated'.")
            print("  Cached responses are replayed regardless, so the notebook still runs.")
    return avail
