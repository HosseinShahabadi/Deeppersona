"""Configuration: API client, global rate limiting, and response parsing.

Configured for OpenRouter (OpenAI-API-compatible). Two things matter here:

1. RATE LIMITING. Free OpenRouter models have tight request limits. The throttle
   below is *global and thread-safe*: it enforces a minimum interval between API
   calls across all worker threads, not per-thread. A per-call time.sleep()
   inside each worker would not do this - with N workers you would still burst N
   requests at once.

2. SINGLE CHOKEPOINT. client.chat.completions.create is wrapped, so throttling
   and retries apply to *every* call site automatically - including the separate
   get_completion defined in select_attributes.py, which bypasses the one in
   this module.
"""

import os
import json
import time
import random
import threading
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union, Tuple

# --------------------------------------------------------------------------
# API configuration
# --------------------------------------------------------------------------

# GeoNames (unused at runtime; geonamescache ships its own data offline)
GEONAMES_USERNAME = "demo"
GEONAMES_API_BASE = "http://api.geonames.org"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Model slug. OpenRouter free models end in ":free", e.g.
#   meta-llama/llama-3.3-70b-instruct:free
#   deepseek/deepseek-chat-v3-0324:free
#   google/gemma-3-27b-it:free
GPT_MODEL = os.environ.get("DEEPPERSONA_MODEL", "openai/gpt-4.1-mini")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. Export it before running.")

# NOTE: upstream force-cleared https_proxy/http_proxy here. That has been removed
# - it would silently override a proxy you actually need to reach the API.

# max_retries=0: we implement our own retry below so that every attempt also
# passes through the throttle. Leaving the SDK's internal retries on would let
# it re-fire immediately, ignoring our rate limit.
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    max_retries=0,
)

# --------------------------------------------------------------------------
# Global throttle
# --------------------------------------------------------------------------

# Minimum seconds between consecutive API calls, process-wide. 0 disables it.
API_DELAY = float(os.environ.get("DEEPPERSONA_API_DELAY", "0") or 0)

# How many times to retry a rate-limited / transient request.
API_MAX_RETRIES = int(os.environ.get("DEEPPERSONA_API_MAX_RETRIES", "5"))

_rate_lock = threading.Lock()
_last_call_at = 0.0


def set_api_delay(seconds: float) -> None:
    """Set the global minimum interval between API calls (seconds)."""
    global API_DELAY
    API_DELAY = max(0.0, float(seconds or 0))


def set_model(model: str) -> None:
    """Override the model slug at runtime."""
    global GPT_MODEL
    if model:
        GPT_MODEL = model


def _throttle() -> None:
    """Block until at least API_DELAY seconds have passed since the last call.

    The lock is deliberately held across the sleep. That serializes the *issuing*
    of requests, which is what produces a true global rate limit; the actual API
    responses still overlap across threads, so workers remain useful.
    """
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


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract a Retry-After / reset hint from a rate-limit error, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        items = dict(headers)
    except Exception:
        items = {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset", "X-RateLimit-Reset"):
        value = items.get(key)
        if not value:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        # OpenRouter may report the reset as a millisecond epoch timestamp.
        if seconds > 1e6:
            seconds = max(0.0, (seconds / 1000.0) - time.time())
        return min(seconds, 120.0)
    return None


def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code


_raw_create = client.chat.completions.create


def _throttled_create(*args, **kwargs):
    """Throttled, retrying wrapper around chat.completions.create.

    Retries on 429 (rate limit) and 5xx (transient upstream) with exponential
    backoff plus jitter, honoring Retry-After when the server provides it.
    Other errors (401 bad key, 404 bad model slug) are raised immediately -
    retrying those just wastes time.
    """
    last_exc = None
    for attempt in range(API_MAX_RETRIES):
        _throttle()
        try:
            return _raw_create(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            code = _status_code(exc)
            retryable = (code == 429) or (code is not None and 500 <= code < 600) or (code is None)

            if not retryable or attempt == API_MAX_RETRIES - 1:
                raise

            wait = _retry_after_seconds(exc)
            if wait is None:
                # Exponential backoff with jitter, floored at the configured delay.
                wait = max(API_DELAY, 2.0 ** attempt) + random.uniform(0, 1.0)

            label = "rate limited (429)" if code == 429 else f"transient error ({code})"
            print(f"  {label}; retrying in {wait:.1f}s "
                  f"[attempt {attempt + 1}/{API_MAX_RETRIES}]")
            time.sleep(wait)

    raise last_exc


client.chat.completions.create = _throttled_create


# --------------------------------------------------------------------------
# Completion helper
# --------------------------------------------------------------------------

def get_completion(messages: List[Dict[str, str]], model: str = None,
                   temperature: float = 0.2, max_retries: int = None) -> Optional[str]:
    """Call the chat API and return the text response.

    Throttling and retry/backoff are handled inside the wrapped client, so this
    just calls through. max_retries is accepted for backwards compatibility.
    """
    model = model or GPT_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def extract_json_from_markdown(response: str) -> str:
    """Extract JSON content from a Markdown code block."""
    if response and response.strip().startswith('```') and '```' in response:
        code_content = response.split('```', 2)[1]
        if code_content.startswith('json'):
            code_content = code_content[4:].strip()
        response = code_content.strip()
    return response


def parse_json_response(response: str, default_value: Any = None) -> Any:
    """Parse a JSON response, tolerating Markdown fences."""
    if not response:
        return default_value

    response = extract_json_from_markdown(response)

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        print(f"Response was: {response[:100]}..." if len(response) > 100 else f"Response was: {response}")
        return default_value


def parse_gpt_response(response: str, expected_fields: List[str] = None,
                       field_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """Parse a response and coerce it to the expected fields."""
    if field_defaults is None:
        field_defaults = {}

    result = parse_json_response(response, {})

    if not expected_fields:
        return result

    output = {}
    for field in expected_fields:
        output[field] = result.get(field, field_defaults.get(field))

    return output


def parse_nested_json_response(response: str) -> Tuple[Dict[str, Any], bool]:
    """Parse a possibly nested JSON response."""
    extracted = extract_json_from_markdown(response)

    try:
        result = json.loads(extracted)

        if isinstance(result, dict) and len(result) == 1 and next(iter(result.values())).startswith('{'):
            key = next(iter(result.keys()))
            try:
                nested_json = json.loads(result[key])
                return nested_json, True
            except json.JSONDecodeError:
                pass

        return result, True
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return {}, False
