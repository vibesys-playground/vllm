#!/usr/bin/env python3
"""A CPU stand-in for the vLLM OpenAI server, for the task's smoke tests only.

It serves the endpoints the benchmark and the accuracy checker use, with the
same wire shapes as vLLM:
  GET  /v1/models
  POST /v1/completions   token-ID prompts; streaming (SSE, `return_token_ids`,
                         usage with `prompt_tokens_details.cached_tokens`) or
                         not (with `logprobs` and `return_tokens_as_token_ids`)
  POST /tokenize         text -> token ids (one id per 4 bytes; not a real tokenizer)

The "model" is a deterministic function of the prompt: logits over a small
candidate set derived from a hash of the whole prompt, so the greedy token
depends on every prompt token. A prefix cache at --block-token blocks (default 16) reports the
cached tokens the way vLLM does.

Fault flags make the failure paths testable: --slow-every/--slow-ms (TTFT
outliers), --fail-after (HTTP 500), --drop-token-ids-after (no generated ids:
the client's next prompt comes up short), --canned (one fixed token for every
prompt), --noise (perturbed logits), --no-cache-report, --fake-replay (streamed
requests, the replay's shape, get a wrong token; non-streamed ones the right one),
--truncate-over N (keep only the first N prompt tokens and count those),
--text-only-after N (stream text but no token ids), --hang-at N,M,... (those
requests never answer), --cached-path-shift (a prompt that resumes a cached prefix
gets a different answer than the same prompt computed cold, as on the GPU builds),
--alt-numerics N (another honest parallel layout: on one prompt in N the top token
is a confident other candidate, deterministic per prompt, as PP4 vs TP4+EP on the
GPUs), --shuffled (every answer is the one for another prompt: plausible, wrong).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Server(ThreadingHTTPServer):
    # The default listen backlog of 5 resets connections when session_runner
    # opens hundreds at once (the real trace at N = 500).
    request_queue_size = 1024

CANDIDATES = 8


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.blocks: set[bytes] = set()
        self.requests = 0

    def next_request(self) -> int:
        with self.lock:
            self.requests += 1
            return self.requests

    def cached_and_store(self, prompt: list[int], salt: str | None) -> int:
        """Longest cached block-aligned prefix (at most len - 1), then cache the prompt.

        A `cache_salt` starts a separate hash chain, as in vLLM.
        """
        digest = hashlib.sha256((salt or "").encode())
        hashes = []
        block = self.args.block
        for start in range(0, len(prompt) - len(prompt) % block, block):
            digest.update(json.dumps(prompt[start:start + block]).encode())
            hashes.append(digest.copy().digest())
        with self.lock:
            hit = 0
            for h in hashes:
                if h not in self.blocks:
                    break
                hit += block
            self.blocks.update(hashes)
        return min(hit, max(len(prompt) - 1, 0))

    def logits(self, prompt: list[int], cached: int = 0) -> dict[int, float]:
        if self.args.canned:
            return {4242: 0.0, 4243: -3.0}
        key = json.dumps(prompt) + ("shuffled" if self.args.shuffled else "")
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        if self.args.cached_path_shift and cached > 0:
            seed ^= 1
        rng = random.Random(seed)
        tokens = [rng.randrange(1_000, 150_000) for _ in range(CANDIDATES)]
        scores = [rng.gauss(0.0, 2.0) for _ in range(CANDIDATES)]
        if self.args.alt_numerics and seed % self.args.alt_numerics == 0:
            # Promote the second-ranked candidate 2 nats above the first: a confident, systematic flip.
            order = sorted(range(CANDIDATES), key=lambda i: -scores[i])
            scores[order[1]] = scores[order[0]] + 2.0
        if self.args.noise:
            noise = random.Random(seed ^ 0x5EED)
            scores = [s + noise.gauss(0.0, self.args.noise) for s in scores]
        log_z = math.log(sum(math.exp(s) for s in scores))
        return {t: s - log_z for t, s in zip(tokens, scores, strict=True)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    model: Model

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _json(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model.args.model, "object": "model"}]})
        elif self.path == "/health":
            self._json(200, {})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/tokenize":
            raw = body["prompt"].encode("utf-8")
            tokens = [1_000 + int.from_bytes(raw[i:i + 4].ljust(4, b"\0"), "big") % 149_000
                      for i in range(0, len(raw), 4)]
            self._json(200, {"tokens": tokens, "count": len(tokens), "max_model_len": 1_048_576})
            return
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return
        args = self.model.args
        n = self.model.next_request()
        if args.fail_after is not None and n > args.fail_after:
            self._json(500, {"error": "injected failure"})
            return
        prompt = body["prompt"]
        if args.truncate_over is not None and len(prompt) > args.truncate_over:
            prompt = prompt[:args.truncate_over]
        if n in args.hang_at:
            time.sleep(3600)
        if body.get("model") != args.model:
            self._json(404, {"error": f"model {body.get('model')!r} not served"})
            return
        cached = self.model.cached_and_store(prompt, body.get("cache_salt"))
        delay_s = (args.base_ms + args.per_1k_ms * (len(prompt) - cached) / 1000) / 1e3
        if args.slow_every and n % args.slow_every == 0:
            delay_s += args.slow_ms / 1e3
        time.sleep(delay_s)
        logits = self.model.logits(prompt, cached)
        token = max(logits, key=logits.get)
        if args.fake_replay and body.get("stream"):
            # A gaming server: skip the work on the shape the replay sends.
            token = 4242
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1, "total_tokens": len(prompt) + 1}
        if not args.no_cache_report:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        drop_ids = args.drop_token_ids_after is not None and n > args.drop_token_ids_after
        text_only = args.text_only_after is not None and n > args.text_only_after
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            choice = {"index": 0, "text": "" if drop_ids else "x", "finish_reason": None}
            if body.get("return_token_ids") and not drop_ids and not text_only:
                choice["token_ids"] = [token]
            events = [{"choices": [choice]}, {"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append({"choices": [], "usage": usage})
            for event in events:
                self._chunk(f"data: {json.dumps(event)}\n\n".encode())
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
            return
        choice = {"index": 0, "text": "x", "finish_reason": "length", "token_ids": [token]}
        k = body.get("logprobs")
        if k:
            top = sorted(logits.items(), key=lambda kv: -kv[1])[:k]
            name = (lambda t: f"token_id:{t}") if body.get("return_tokens_as_token_ids") else (lambda t: f"t{t}")
            choice["logprobs"] = {
                "tokens": [name(token)],
                "token_logprobs": [logits[token]],
                "top_logprobs": [{name(t): lp for t, lp in top}],
                "text_offset": [0],
            }
        self._json(200, {"id": f"cmpl-{n}", "object": "text_completion", "model": args.model,
                         "choices": [choice], "usage": usage})

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-NVFP4")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--base-ms", type=float, default=5.0)
    parser.add_argument("--per-1k-ms", type=float, default=0.5)
    parser.add_argument("--slow-every", type=int, default=0)
    parser.add_argument("--slow-ms", type=float, default=0.0)
    parser.add_argument("--fail-after", type=int, default=None)
    parser.add_argument("--drop-token-ids-after", type=int, default=None)
    parser.add_argument("--canned", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--no-cache-report", action="store_true")
    parser.add_argument("--fake-replay", action="store_true")
    parser.add_argument("--cached-path-shift", action="store_true")
    parser.add_argument("--alt-numerics", type=int, default=0)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--truncate-over", type=int, default=None)
    parser.add_argument("--text-only-after", type=int, default=None)
    parser.add_argument("--hang-at", type=lambda v: {int(x) for x in v.split(",")}, default=set())
    args = parser.parse_args()
    Handler.model = Model(args)
    server = Server(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"fake server on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
