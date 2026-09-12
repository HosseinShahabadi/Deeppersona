"""Configuration: API client, global rate limiting, and response parsing.

Works with any OpenAI-compatible completions endpoint: OpenRouter, a
self-hosted LiteLLM/vLLM gateway, or any other provider that speaks the
OpenAI chat-completions API. Set OPENROUTER_API_KEY / OPENROUTER_BASE_URL /
DEEPPERSONA_MODEL (env vars or --api-key/--endpoint/--model on
generate_profile.py) to point at whichever endpoint you're using.
"""

import os
import json
import time
import random
import threading
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from openai import OpenAI
from typing import List, Dict, Optional, Any, Tuple

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GPT_MODEL = os.environ.get("DEEPPERSONA_MODEL", "openai/gpt-4.1-mini")

_PLACEHOLDER_KEY = "unset-pass-via-api-key-flag-or-OPENROUTER_API_KEY-env"

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. Export it or pass --api-key "
          "before generating any profiles.")

def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code


def _wrap_client(c):
    """Wrap a client's chat.completions.create with throttling + retry.

    Extracted so set_api_key()/set_base_url() can rebuild the client and
    re-apply the same wrapping, since a fresh OpenAI(...) object has its
    original, unwrapped create() method.
    """
    raw_create = c.chat.completions.create

    def _throttled_create(*args, **kwargs):
        last_exc = None
        for attempt in range(API_MAX_RETRIES):
            _throttle()
            try:
                return raw_create(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                code = _status_code(exc)
                retryable = (code == 429) or (code is not None and 500 <= code < 600) or (code is None)
                if not retryable or attempt == API_MAX_RETRIES - 1:
                    raise
                wait = max(API_DELAY, 2.0 ** attempt) + random.uniform(0, 1.0)
                print(f"  retrying in {wait:.1f}s [attempt {attempt + 1}/{API_MAX_RETRIES}] ({exc})")
                time.sleep(wait)
        raise last_exc

    c.chat.completions.create = _throttled_create
    return c


client = _wrap_client(OpenAI(api_key=OPENROUTER_API_KEY or _PLACEHOLDER_KEY,
                             base_url=OPENROUTER_BASE_URL, max_retries=0))


def set_api_key(api_key: str) -> None:
    """Point the client at a different API key, rebuilding it in place.

    IMPORTANT: this reassigns the module-global `client`. Any module that did
    `from config import client` before this call keeps a reference to the OLD
    client object. In this codebase that's safe because select_attributes.py
    only imports config lazily, inside generate_single_profile — i.e. after
    CLI setup (and therefore after this function) has already run. If you add
    a new module that imports `client` at its own top level, prefer
    `import config` + `config.client` (looked up fresh on each use) instead.
    """
    global client, OPENROUTER_API_KEY
    if not api_key:
        return
    OPENROUTER_API_KEY = api_key
    client = _wrap_client(OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, max_retries=0))


def set_base_url(base_url: str) -> None:
    """Point the client at a different endpoint, rebuilding it in place.

    Works with any OpenAI-compatible endpoint: OpenRouter, a self-hosted
    LiteLLM/vLLM gateway, etc. See set_api_key() for the same caveat about
    module-global reassignment.
    """
    global client, OPENROUTER_BASE_URL
    if not base_url:
        return
    OPENROUTER_BASE_URL = base_url
    client = _wrap_client(OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, max_retries=0))

API_DELAY = float(os.environ.get("DEEPPERSONA_API_DELAY", "0") or 0)
API_MAX_RETRIES = int(os.environ.get("DEEPPERSONA_API_MAX_RETRIES", "5"))

_rate_lock = threading.Lock()
_last_call_at = 0.0


# OpenRouter includes token accounting and, for normal (non-BYOK) requests,
# the billed USD amount in the `usage` object of a completion response.  Keep
# the raw value it reports rather than attempting to duplicate its routing and
# pricing calculations locally.
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class CostTracker:
    """Thread-safe collector for OpenRouter usage and billed-cost metadata."""

    def __init__(self, report_path: str, run_metadata: Optional[Dict[str, Any]] = None):
        self.report_path = report_path
        self.run_metadata = dict(run_metadata or {})
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _value(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def record(self, response: Any, requested_model: str) -> None:
        usage = self._value(response, "usage")
        if usage is None:
            return

        # `cost` is OpenRouter's actual billed cost. It can be absent when a
        # compatible non-OpenRouter endpoint is used, so preserve that state
        # instead of publishing an estimate as if it were an invoice amount.
        raw_cost = self._value(usage, "cost")
        try:
            cost_usd = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            cost_usd = None

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_index": _request_context.get().get("profile_index"),
            "model_requested": requested_model,
            "model_returned": self._value(response, "model", requested_model),
            "generation_id": self._value(response, "id"),
            "prompt_tokens": self._value(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": self._value(usage, "completion_tokens", 0) or 0,
            "total_tokens": self._value(usage, "total_tokens", 0) or 0,
            "openrouter_cost_usd": cost_usd,
        }
        with self._lock:
            self.events.append(event)

    def report(self, status: str) -> Dict[str, Any]:
        with self._lock:
            events = list(self.events)

        totals = {
            "api_requests": len(events),
            "prompt_tokens": sum(int(e["prompt_tokens"]) for e in events),
            "completion_tokens": sum(int(e["completion_tokens"]) for e in events),
            "total_tokens": sum(int(e["total_tokens"]) for e in events),
        }
        known_costs = [e["openrouter_cost_usd"] for e in events if e["openrouter_cost_usd"] is not None]
        totals["openrouter_cost_usd"] = round(sum(known_costs), 10) if known_costs else None
        totals["requests_with_openrouter_cost"] = len(known_costs)

        per_profile: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                     "total_tokens": 0, "openrouter_cost_usd": 0.0,
                     "requests_with_openrouter_cost": 0}
        )
        for event in events:
            if event["profile_index"] is None:
                continue
            bucket = per_profile[str(event["profile_index"])]
            bucket["api_requests"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                bucket[key] += int(event[key])
            if event["openrouter_cost_usd"] is not None:
                bucket["openrouter_cost_usd"] += event["openrouter_cost_usd"]
                bucket["requests_with_openrouter_cost"] += 1
        for bucket in per_profile.values():
            bucket["openrouter_cost_usd"] = (
                round(bucket["openrouter_cost_usd"], 10)
                if bucket["requests_with_openrouter_cost"] else None
            )

        return {
            "metadata": {
                **self.run_metadata,
                "started_at": self.started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "cost_source": "OpenRouter completion response usage.cost",
                "note": "A null cost means the endpoint did not return OpenRouter billed cost; no local estimate is substituted.",
            },
            "totals": totals,
            "per_profile": dict(per_profile),
            "requests": events,
        }

    def write(self, status: str) -> Dict[str, Any]:
        report = self.report(status)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report


_cost_tracker: Optional[CostTracker] = None


def start_cost_tracking(report_path: str, run_metadata: Optional[Dict[str, Any]] = None) -> None:
    """Start a new run-level cost ledger. Call after startup validation."""
    global _cost_tracker
    _cost_tracker = CostTracker(report_path, run_metadata)


def set_request_context(**context: Any):
    """Associate subsequent API calls in this thread with a profile/run context."""
    return _request_context.set(context)


def reset_request_context(token: Any) -> None:
    _request_context.reset(token)


def write_cost_report(status: str) -> Optional[Dict[str, Any]]:
    """Persist the current ledger and return it, or None if tracking is off."""
    return _cost_tracker.write(status) if _cost_tracker else None


def set_api_delay(seconds: float) -> None:
    global API_DELAY
    API_DELAY = max(0.0, float(seconds or 0))


def set_model(model: str) -> None:
    global GPT_MODEL
    if model:
        GPT_MODEL = model


def _throttle() -> None:
    global _last_call_at
    if API_DELAY <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = (_last_call_at + API_DELAY) - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_call_at = now




def get_completion(messages: List[Dict[str, str]], model: str = None,
                   temperature: float = 0.2) -> Optional[str]:
    """Call the chat API and return the text response.

    Throttling and retry/backoff happen inside the wrapped client (see
    _wrap_client), governed by the module-globals API_DELAY and
    API_MAX_RETRIES — not by anything passed to this function. To change the
    retry count for a specific call, temporarily set config.API_MAX_RETRIES.
    """
    if OPENROUTER_API_KEY == "" :
        print("ERROR: no API key configured. Pass --api-key or set OPENROUTER_API_KEY "
              "before running.")
        return None
    model = model or GPT_MODEL
    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        if _cost_tracker:
            _cost_tracker.record(response, model)
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def extract_json_from_markdown(response: str) -> str:
    if response and response.strip().startswith('```') and '```' in response:
        code_content = response.split('```', 2)[1]
        if code_content.startswith('json'):
            code_content = code_content[4:].strip()
        response = code_content.strip()
    return response


def parse_json_response(response: str, default_value: Any = None) -> Any:
    if not response:
        return default_value
    response = extract_json_from_markdown(response)
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return default_value


def parse_gpt_response(response: str, expected_fields: List[str] = None,
                       field_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    if field_defaults is None:
        field_defaults = {}
    result = parse_json_response(response, {})
    if not expected_fields:
        return result
    return {f: result.get(f, field_defaults.get(f)) for f in expected_fields}


def parse_nested_json_response(response: str) -> Tuple[Dict[str, Any], bool]:
    extracted = extract_json_from_markdown(response)
    try:
        result = json.loads(extracted)
        if isinstance(result, dict) and len(result) == 1 and next(iter(result.values())).startswith('{'):
            key = next(iter(result.keys()))
            try:
                return json.loads(result[key]), True
            except json.JSONDecodeError:
                pass
        return result, True
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return {}, False
