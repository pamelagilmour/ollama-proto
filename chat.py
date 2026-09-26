"""Send one prompt to a local Ollama model and stream the reply.

Usage:
    uv run chat.py "Explain what a KV cache is in two sentences."
    uv run chat.py --model gemma4:e4b "Same question, different model."
"""

import argparse
import sys
import time

from openai import APIConnectionError, NotFoundError, OpenAI

OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:9b"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument("prompt", help="What to ask the model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512, help="Cap on reply length so a runaway answer stops.")
    parser.add_argument("--system", default="You are a concise assistant.", help="System prompt.")
    parser.add_argument(
        "--think",
        action="store_true",
        help="Let reasoning models (e.g. Qwen 3.5) think before answering. Slower; off by default.",
    )
    args = parser.parse_args()

    # Ollama ignores the API key but the client requires a non-empty string.
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    finish_reason: str | None = None

    try:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            # Ollama enables thinking by default on capable models when this is omitted,
            # and the hidden reasoning can consume the whole max_tokens budget.
            reasoning_effort="medium" if args.think else "none",
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            text = choice.delta.content
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(text, end="", flush=True)
    except APIConnectionError:
        print(
            f"Could not reach Ollama at {OLLAMA_URL}.\n"
            "Start it with `ollama serve` (or `brew services start ollama`) and try again.",
            file=sys.stderr,
        )
        return 1
    except NotFoundError:
        print(
            f"Model '{args.model}' is not downloaded.\n"
            f"Run `ollama pull {args.model}`, or pick one from `ollama list`.",
            file=sys.stderr,
        )
        return 1

    finished = time.perf_counter()
    print()

    if first_token_at is not None:
        ttft = first_token_at - started
        total_seconds = finished - started
        # Measured over the whole request so hidden reasoning tokens (which arrive
        # before the first visible token) do not inflate the rate.
        rate = completion_tokens / total_seconds if total_seconds > 0 else 0
        print(
            f"\n[{args.model}] first token {ttft:.2f}s | {completion_tokens} tokens in {total_seconds:.1f}s | {rate:.1f} tok/s",
            file=sys.stderr,
        )
    else:
        hint = (
            f"The model used all {completion_tokens} tokens on hidden reasoning (finish_reason={finish_reason}). "
            "Raise --max-tokens or drop --think."
            if args.think
            else f"No visible answer was returned (finish_reason={finish_reason}, {completion_tokens} tokens generated)."
        )
        print(f"\n[{args.model}] {hint}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
