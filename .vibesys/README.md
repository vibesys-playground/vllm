# VibeSys tasks for vLLM

This fork keeps long-running optimization tasks beside the vLLM code they
change. Each directory under `tasks/` contains an objective, a correctness
checker, and a benchmark manifest. VibeSys launches coding agents and evaluator
commands from the repository root, so vLLM remains the candidate project.

Available tasks:

- `llama-3-8b-h100-constrained-json`
- `llama-3-8b-h100-high-concurrency`
- `llama-3-8b-h100-long-prompts`
- `llama-70b-2xh100-vllm`
- `qwen3-coder-tracelab-h100`

Select a task explicitly when this repository contains more than one:

```bash
vibesys --project . --task llama-3-8b-h100-high-concurrency
```

Task definitions and evaluator harnesses under `.vibesys/tasks/` are trusted
inputs. Coding agents may change the vLLM repository but not those inputs.
