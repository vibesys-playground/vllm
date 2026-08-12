# Llama 3.1 8B H100 constrained-JSON task

Run from the vLLM repository root:

```bash
vibesys --project . --task llama-3-8b-h100-constrained-json \
  --local --interface service --modal
```

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `meta-llama/Llama-3.1-8B-Instruct` through
vLLM's OpenAI-compatible API. The task measures constrained-decoding
throughput and validates every streamed response against its JSON schema.

Evaluator Python dependencies are isolated from vLLM's project environment by
the manifest's `uv run --no-project` commands.
