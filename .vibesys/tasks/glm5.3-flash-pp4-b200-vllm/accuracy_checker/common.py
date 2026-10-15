"""Shared by make_reference.py and checker.py: the cases, the request, the reference file format."""
from __future__ import annotations

import base64
import dataclasses
import http.client
import json
import secrets
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
MODEL_REVISION = "da920bb0b9f4a06727223a349e55468e38352348"
# 2: each answer holds the tokens observed over repeats and the logprob range of its token;
# cached round-1 answers are recorded through the cached path.
REFERENCE_VERSION = 2
# Top-k logprobs stored per case (vLLM's default --max-logprobs is 20).
TOP_K = 10

K = 1024


@dataclasses.dataclass(frozen=True)
class Single:
    """One lone prompt of `tokens` fresh tokens (reference/1_contract.md §2.1)."""

    name: str
    tokens: int


@dataclasses.dataclass(frozen=True)
class Session:
    """Two rounds: round 0 is `prefix` fresh tokens; round 1 is round 0's prompt, its
    greedy output token, then `append - 1` fresh tokens (`5_prefix_cache.md` §1).

    `cached`: the checker sends round 0 then round 1, so round 1 runs on a cached
    prefix. Otherwise the checker sends only round 1, cold: a session round
    without a prefix in the cache.
    """

    name: str
    prefix: int
    append: int
    cached: bool


# The §2.1 lengths, more prompts at the short lengths where they are cheap.
SINGLES: tuple[Single, ...] = tuple(
    Single(f"single_{length // K}k_{i}", length)
    for length, count in ((1 * K, 4), (2 * K, 4), (4 * K, 4), (8 * K, 4), (16 * K, 2), (32 * K, 2),
                          (64 * K, 1), (128 * K, 1))
    for i in range(count)
)
SESSIONS: tuple[Session, ...] = (
    Session("session_p4k_a1k_cached", 4 * K, 1 * K, True),
    Session("session_p8k_a1k_cached", 8 * K, 1 * K, True),
    Session("session_p16k_a2k_cached", 16 * K, 2 * K, True),
    Session("session_p32k_a1k_cached", 32 * K, 1 * K, True),
    Session("session_p32k_a8k_cached", 32 * K, 8 * K, True),
    Session("session_p128k_a1k_cached", 128 * K, 1 * K, True),
    Session("session_p8k_a1k_cold", 8 * K, 1 * K, False),
    Session("session_p32k_a4k_cold", 32 * K, 4 * K, False),
    Session("session_p64k_a1k_cold", 64 * K, 1 * K, False),
)
# Prompts come from enwik9 past this byte offset: session_runner's token pool
# (the benchmark's prompts) is the first ~100M tokens of the file, so no checker
# prompt can share a prefix with a benchmark prompt on a server that serves both.
TEXT_START = 600_000_000
TEXT_STRIDE = 4_000_000


def pack_ids(ids: list[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(ids)}I", *ids)).decode("ascii")


def unpack_ids(data: str) -> list[int]:
    raw = base64.b64decode(data)
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


@dataclasses.dataclass(frozen=True)
class Answer:
    """The first generated token of one request and the top logprobs at that position."""

    token: int
    top: dict[int, float]
    prompt_tokens: int | None
    cached_tokens: int | None
    seconds: float


class RequestError(Exception):
    pass


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RequestError(f"POST {url}: HTTP {exc.code}: {exc.read()[:300]!r}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RequestError(f"POST {url}: {exc!r}") from exc


def get_json(url: str, timeout_s: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RequestError(f"GET {url}: {exc!r}") from exc


def _token_id(name: str) -> int:
    if not name.startswith("token_id:"):
        raise RequestError(f"logprob token {name!r} is not 'token_id:<id>' (return_tokens_as_token_ids)")
    return int(name.split(":", 1)[1])


def complete(base_url: str, model: str, prompt: list[int], timeout_s: float,
             cache_salt: str | None = None) -> Answer:
    """Greedy first token with top-k logprobs, non-streaming /v1/completions on token ids."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": TOP_K,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    start = time.perf_counter()
    body = post_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout_s)
    seconds = time.perf_counter() - start
    try:
        choice = body["choices"][0]
        logprobs = choice["logprobs"]
        token = _token_id(logprobs["tokens"][0])
        top = {_token_id(name): float(lp) for name, lp in logprobs["top_logprobs"][0].items()}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RequestError(f"malformed completion response: {exc!r}: {json.dumps(body)[:400]}") from exc
    ids = choice.get("token_ids")
    if ids is not None and ids[:1] != [token]:
        raise RequestError(f"token_ids {ids} disagree with the logprob token {token}")
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return Answer(token, top, usage.get("prompt_tokens"), details.get("cached_tokens"), seconds)


def replay_shaped_token(base_url: str, model: str, prompt: list[int],
                        timeout_s: float) -> tuple[int, int | None, int | None]:
    """The first generated token of a request shaped exactly like a benchmark replay round.

    session_runner (patched, backend `openai`) streams with `return_token_ids`,
    asks for the usage chunk, sends no logprobs, and carries a 32-hex-digit
    opaque id in the body `rid` and the `x-request-id` header. This request does
    the same, so a server cannot answer the benchmark's requests differently from
    the checker's by their shape. Returns (token, usage.prompt_tokens, cached tokens).
    """
    parsed = urllib.parse.urlparse(base_url)
    rid = secrets.token_hex(16)
    payload = {
        "model": model,
        "rid": rid,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "return_prompt_token_ids": False,
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=timeout_s)
    token: int | None = None
    prompt_tokens: int | None = None
    cached: int | None = None
    try:
        conn.request("POST", f"{parsed.path.rstrip('/')}/v1/completions", body=json.dumps(payload),
                     headers={"Content-Type": "application/json", "x-request-id": rid})
        response = conn.getresponse()
        if response.status != 200:
            raise RequestError(f"replay-shaped request: HTTP {response.status}: {response.read()[:300]!r}")
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
                continue
            event = json.loads(line[5:].strip())
            for choice in event.get("choices") or []:
                ids = choice.get("token_ids") or []
                if ids and token is None:
                    token = int(ids[0])
            if event.get("usage"):
                prompt_tokens = event["usage"].get("prompt_tokens")
                cached = (event["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise RequestError(f"replay-shaped request failed: {exc!r}") from exc
    finally:
        conn.close()
    if token is None:
        raise RequestError("replay-shaped request: the server streamed no generated token ids")
    return token, prompt_tokens, cached
