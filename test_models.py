"""
Run a single philosophical question against every model in LLM_OPTIONS
and print a timed comparison table.

Usage:  python test_models.py
"""

import time
import sys
import io

# Force UTF-8 output on Windows to handle Unicode characters in LLM responses
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import LLM_OPTIONS, DEFAULT_LLM

TEST_QUESTION = "What is Nietzsche's view on nihilism and the will to power?"

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"
SKIP  = "\033[93mSKIP\033[0m"

rows = []

print(f"\nTest question: \"{TEST_QUESTION}\"\n")
print(f"{'Model':<45} {'Provider':<12} {'Status':<6} {'Time (s)':>8}  Preview")
print("-" * 110)

for label, (provider, model_id) in LLM_OPTIONS.items():
    sys.stdout.write(f"  {label:<43} {provider:<12} ... ")
    sys.stdout.flush()

    t0 = time.perf_counter()
    try:
        from rag_chain import query
        result = query(TEST_QUESTION, "All", label)
        elapsed = time.perf_counter() - t0
        answer = result["answer"].replace("\n", " ").strip()
        preview = answer[:80] + ("…" if len(answer) > 80 else "")
        status = "PASS"
        rows.append((label, provider, status, elapsed, preview))
        print(f"\r  {label:<43} {provider:<12} {PASS}  {elapsed:>8.2f}s  {preview}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err = str(exc)[:80]
        status = "SKIP" if "not set" in str(exc) else "FAIL"
        rows.append((label, provider, status, elapsed, err))
        tag = SKIP if status == "SKIP" else FAIL
        print(f"\r  {label:<43} {provider:<12} {tag}  {elapsed:>8.2f}s  {err}")

print("\n" + "=" * 110)
passed = sum(1 for r in rows if r[2] == "PASS")
failed = sum(1 for r in rows if r[2] == "FAIL")
skipped = sum(1 for r in rows if r[2] == "SKIP")
print(f"Results: {passed} passed, {failed} failed, {skipped} skipped / {len(rows)} total")

# Fastest among passing
passing = [r for r in rows if r[2] == "PASS"]
if passing:
    fastest = min(passing, key=lambda r: r[3])
    slowest = max(passing, key=lambda r: r[3])
    print(f"Fastest: {fastest[0]} ({fastest[3]:.2f}s)")
    print(f"Slowest: {slowest[0]} ({slowest[3]:.2f}s)")
print()
