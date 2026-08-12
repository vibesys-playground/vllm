# Llama 3.3 70B 2xH100 task

Run from the vLLM repository root:

```bash
vibesys --project . --task llama-70b-2xh100-vllm \
  --local --interface service --modal
```

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `meta-llama/Llama-3.3-70B-Instruct` through
vLLM's OpenAI-compatible API across two H100 GPUs. The task measures aggregate
throughput and p99 latency while enforcing prompt-conditioned correctness.

Evaluator Python dependencies are isolated from vLLM's project environment by
the manifest's `uv run --no-project` commands.
