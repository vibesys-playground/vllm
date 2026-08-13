# Qwen3-Coder TraceLab H100 task

TraceLab-shaped vLLM serving target for `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on Modal H100.

The baseline Modal service is
`examples/deployment/vibesys_qwen3_coder_tracelab.py`. It deploys the local
`vllm/` package over the pinned compiled wheel, so the vLLM checkout is the
candidate that agents optimize.

The benchmark replays real TraceLab public coding-agent sessions through
TraceLab's own `session_runner` instead of independent chat requests. Its
task-owned implementation lives under `benchmark/tracelab-replay` and is
read-only to coding agents.

Initialize the pinned TraceLab submodule before starting a run:

```bash
git submodule update --init \
  .vibesys/tasks/qwen3-coder-tracelab-h100/benchmark/tracelab-replay/tracelab
```

Start an optimization run with:

```bash
vibesys \
  --project . \
  --task qwen3-coder-tracelab-h100 \
  --runs-dir /work/vibesys-runs \
  --local \
  --exp-name qwen3-coder-tracelab-h100 \
  --modal \
  --modal-gpu H100 \
  --agent-backend cli \
  --cli-provider codex \
  --backend cuda \
  --interface service \
  --modality text_generation \
  --profiler torch \
  --max-rounds 4 \
  --headless
```
