# Qwen3.5-397B-A17B 4xB200 task

Run from the vLLM repository root:

    vibesys --project . --task qwen3.5-397b-a17b-4xb200-vllm \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `Qwen/Qwen3.5-397B-A17B` (a hybrid
Gated-DeltaNet + sparse-MoE model) through vLLM's OpenAI-compatible API across
4 B200 GPUs, in FP8 to fit the weights. The optimization target is TP+EP
sharding, FP8 grouped-GEMM MoE kernels, and the hybrid attention state and KV
cache. This is the vLLM counterpart to the SGLang and from-scratch variants.

Evaluator Python dependencies are isolated from vLLM's project environment by
the manifest's `uv run --no-project` commands.
