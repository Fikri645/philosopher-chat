"""
Test conversation memory: send a follow-up message with history attached
and verify all three providers handle the new multi-turn format correctly.

Usage:  python test_history.py
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import LLM_OPTIONS
from rag_chain import stream_llm, retrieve_docs

Q1 = "What is Nietzsche's view on the death of God?"
Q2 = "How does this relate to the concept of nihilism?"   # follow-up — needs history

# Fake first-turn history the way Gradio passes it back
MOCK_HISTORY = [
    {"role": "user", "content": Q1},
    {
        "role": "assistant",
        "content": (
            "Nietzsche argues that the death of God signals the collapse of "
            "transcendent moral frameworks, forcing humanity to create its own values. "
            "As he writes in <em>Thus Spoke Zarathustra</em>, the Übermensch "
            "must fill the void left by divine authority through acts of self-overcoming."
            "\n\n---\n**Sources:**\n- `0.30` Nietzsche — *Thus Spoke Zarathustra*"
        ),
    },
]

# Pick one representative model per provider
PROBE_MODELS = {
    label: (provider, mid)
    for label, (provider, mid) in LLM_OPTIONS.items()
    if label in (
        "Gemma 4 MoE 26B  [Google]",
        "Llama 3.1 8B  [Groq]",
        "Llama 3.3 70B  [OpenRouter]",
    )
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

docs, scores = retrieve_docs(f"{Q1} {Q2}", "All")
context_str = "\n\n".join(d.page_content for d in docs)

print(f"\nHistory test — follow-up: \"{Q2}\"\n")
print(f"{'Model':<45} {'Provider':<12} {'Status':<6} {'Time (s)':>8}  Preview")
print("-" * 110)

for label, (provider, model_id) in PROBE_MODELS.items():
    sys.stdout.write(f"  {label:<43} {provider:<12} ... ")
    sys.stdout.flush()
    t0 = time.perf_counter()
    try:
        chunks = list(stream_llm(provider, model_id, context_str, Q2, history=MOCK_HISTORY))
        elapsed = time.perf_counter() - t0
        answer = "".join(chunks).replace("\n", " ").strip()
        preview = answer[:90] + ("…" if len(answer) > 90 else "")
        print(f"\r  {label:<43} {provider:<12} {PASS}  {elapsed:>8.2f}s  {preview}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\r  {label:<43} {provider:<12} {FAIL}  {elapsed:>8.2f}s  {str(exc)[:90]}")

print()
