# Qwen3.6-35B-A3B 2xH100 task

Run from the vLLM repository root:

    vibesys --project . --task qwen3.6-35b-a3b-2xh100-vllm \
      --local --interface service --modal

The coding agent and evaluators run with this repository as their working
directory. The candidate serves `Qwen/Qwen3.6-35B-A3B` through vLLM's
OpenAI-compatible API across 2 H100 GPUs, and the optimization target is the
sparse-MoE execution path (expert routing, fused grouped-GEMM, bandwidth-bound
decode) plus the tensor-vs-expert parallelism strategy for the second GPU.

Evaluator Python dependencies are isolated from vLLM's project environment by
the manifest's `uv run --no-project` commands.
