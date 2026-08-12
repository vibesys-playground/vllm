# Llama 3.1 8B H100 long-prompt task

Run from the vLLM repository root:

```bash
vibesys --project . --task llama-3-8b-h100-long-prompts \
  --local --interface service --modal
```

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `meta-llama/Llama-3.1-8B-Instruct` through
vLLM's OpenAI-compatible API. The task measures throughput and p99 latency for
repeated long prompts with short decode tails.

Evaluator Python dependencies are isolated from vLLM's project environment by
the manifest's `uv run --no-project` commands.
