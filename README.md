# ollama-proto

A minimal Python project for talking to a local model served by [Ollama](https://ollama.com) through its OpenAI-compatible API. One script, one dependency. A starting point for comparing models and settings on your own machine.

Tested target: Apple M4, 16 GB unified memory, `qwen3.5:9b` at Q4.

## Prerequisites

- Ollama installed and running (`ollama serve`, or `brew services start ollama`)
- At least one model pulled, e.g. `ollama pull qwen3.5:9b`
- [uv](https://docs.astral.sh/uv/) for Python dependency management

## Run

```bash
uv run chat.py "Explain what a KV cache is in two sentences."
```

`uv run` creates a virtual environment and installs `openai` the first time. The reply streams to stdout; a one-line timing summary (time to first token, tokens generated, tokens/sec) goes to stderr so you can pipe the answer without the stats.

Options:

```bash
uv run chat.py --model gemma4:e4b "Same prompt, different model."
uv run chat.py --temperature 0 "Return only valid JSON: {\"a\": 1}"
uv run chat.py --max-tokens 100 "Give me a haiku about unified memory."
uv run chat.py --system "You are a terse senior engineer." "Review this regex: ^\\d{3}-\\d{4}$"
```

## What to look at

- **tok/s** on the stderr line. On a 16 GB M4 a 9B Q4 model should land around 12–18. If it drops well below that, the model or its context is spilling out of memory; check Activity Monitor > Memory Pressure and `ollama ps`.
- **first token** latency grows with prompt length and context size.
- Swap `--model` to compare models on identical prompts. `ollama list` shows what you have.
