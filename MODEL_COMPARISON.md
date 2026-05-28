# LLM Model Comparison

Benchmarked on: *"What is Nietzsche's view on nihilism and the will to power?"*  
Setup: RTX 3060, EmbeddingGemma-300M on CUDA, ChromaDB (~5,700 chunks), `RETRIEVAL_K=5`.

Rate limits for Google verified directly from **aistudio.google.com/rate-limit** (May 2026).  
Rate limits for Groq verified from **live API response headers** + console.groq.com/docs/rate-limits.  
Rate limits for OpenRouter verified from **openrouter.ai/docs/guides/routing/model-variants/free**.

---

## Full Comparison Table

| Model | Provider | Latency | RPM | TPM | RPD | Notes |
|---|---|---|---|---|---|---|
| **Gemma 4 MoE 26B** | Google | ~65 s | 15 | ∞ | **1,500** | Best limits of any Google model; slow but deep |
| **Gemma 4 Dense 31B** | Google | ~25 s | 15 | ∞ | **1,500** | Same limits as MoE; faster, slightly less depth |
| **Gemini 3.1 Flash Lite** | Google | ~0.6 s | 15 | 250K | **500** | Newest Gemini, highest RPD among Flash models |
| **Gemini 3.5 Flash** | Google | ~0.8 s | 5 | 250K | 20 | Latest Gemini series; crisp reasoning |
| **Gemini 3 Flash** | Google | ~9 s | 5 | 250K | 20 | Solid baseline Gemini 3 |
| **Gemini 2.5 Flash** | Google | ~7 s | 5 | 250K | 20 | Previous generation; well-rounded |
| **Gemini 2.5 Flash Lite** | Google | ~2 s | 10 | 250K | 20 | Fastest 2.5; same 20 RPD as 2.5 Flash |
| **Llama 3.1 8B** | Groq | ~2 s | **14,400** | 6K | **14,400** | Highest throughput by far; limited depth |
| **Llama 4 Scout 17B** | Groq | ~1.5 s | 1,000 | 30K | 1,000 | Fastest quality model overall |
| **Llama 3.3 70B** | Groq | ~4.5 s | 1,000 | 12K | 1,000 | Best Groq quality; lower token quota |
| **Qwen3 32B** | Groq | ~5 s | 1,000 | 6K | 1,000 | Chain-of-thought; deepest Groq reasoning |
| **Nvidia Nemotron 120B** | OpenRouter | ~75 s | 20 | — | 50* | Exceptional philosophical depth; slow |
| **OpenAI OSS 120B** | OpenRouter | ~22 s | 20 | — | 50* | Strong quality; best free OR option |
| **DeepSeek V4 Flash** | OpenRouter | ~5 s† | 20 | — | 50* | 1M context window; fast when available |
| **Llama 3.3 70B** | OpenRouter | ~4 s† | 20 | — | 50* | Same weights as Groq; use Groq instead |
| **Qwen3 Next 80B** | OpenRouter | ~8 s† | 20 | — | 50* | Strong reasoning; frequently throttled |
| **Gemma 4 MoE 26B** | OpenRouter | ~5 s† | 20 | — | 50* | Same weights as Google version |

*50 RPD without account credits; 1,000 RPD with $10+ credit purchase  
†Latency when not throttled; free-tier provider-side 429s are common during peak hours  
— OpenRouter does not enforce token-based limits on free models

---

## Rate Limit Deep-Dive

### Google AI Studio — verified from aistudio.google.com/rate-limit

| Model | API Model ID | RPM | TPM | RPD |
|---|---|---|---|---|
| Gemma 4 MoE 26B | `gemma-4-26b-a4b-it` | 15 | **Unlimited** | **1,500** |
| Gemma 4 Dense 31B | `gemma-4-31b-it` | 15 | **Unlimited** | **1,500** |
| Gemini 3.1 Flash Lite | `gemini-3.1-flash-lite` | 15 | 250,000 | **500** |
| Gemini 3.5 Flash | `gemini-3.5-flash` | 5 | 250,000 | 20 |
| Gemini 3 Flash | `gemini-3-flash-preview` | 5 | 250,000 | 20 |
| Gemini 2.5 Flash | `gemini-2.5-flash` | 5 | 250,000 | 20 |
| Gemini 2.5 Flash Lite | `gemini-2.5-flash-lite` | 10 | 250,000 | 20 |
| ~~Gemini 2.5 Pro~~ | ~~`gemini-2.5-pro`~~ | 0 | 0 | 0 |
| ~~Gemini 2.0 Flash~~ | ~~`gemini-2.0-flash`~~ | 0 | 0 | 0 |

> **Key insight:** Gemma 4 models have *significantly better* limits than Gemini models — unlimited TPM and 1,500 RPD vs just 20 RPD for most Gemini Flash variants. Gemini 2.5 Pro and 2.0 Flash are completely locked (0/0/0) on this account's free tier.

---

### Groq — verified from live API headers + docs

| Model | API Model ID | RPM | TPM | RPD | TPD |
|---|---|---|---|---|---|
| Llama 3.1 8B instant | `llama-3.1-8b-instant` | **14,400** | 6,000 | **14,400** | 500,000 |
| Llama 3.3 70B versatile | `llama-3.3-70b-versatile` | 1,000 | 12,000 | 1,000 | 100,000 |
| Llama 4 Scout 17B | `meta-llama/llama-4-scout-17b-16e-instruct` | 1,000 | 30,000 | 1,000 | 500,000 |
| Qwen3 32B | `qwen/qwen3-32b` | 1,000 | 6,000 | 1,000 | 500,000 |

> **Key insight:** Groq is the most generous free tier for RAG use. Llama 3.1 8B has 14,400 RPD — useful for high-volume scenarios. Note that TPM limits (6K–30K) can be a bottleneck when RAG context is large; Llama 4 Scout has the most generous TPM at 30K.

---

### OpenRouter — all `:free` models share identical limits

| Metric | Without credits | With $10+ credits |
|---|---|---|
| RPM | 20 | 20 |
| RPD | **50** | 1,000 |
| TPM / TPD | Unlimited | Unlimited |

> **Key insight:** 50 RPD is exhausted extremely quickly — this explains the frequent 429 errors during testing. OpenRouter free tier is best for occasional access to very large models (120B+) not available elsewhere, not for regular daily use. Provider-side throttling from upstream (NVIDIA, DeepSeek, etc.) adds additional 429s beyond OpenRouter's own quota.

---

## Provider Verdict

| Provider | Best for | Main bottleneck |
|---|---|---|
| **Google (Gemma 4)** | Best free tier overall — high RPD + unlimited tokens | Slow inference (~25–65 s) |
| **Google (Gemini 3.1 Flash Lite)** | Best speed + reasonable daily quota | 500 RPD, 250K TPM |
| **Groq** | Fastest inference, high-volume use | TPM cap (6K–30K) limits long RAG contexts |
| **OpenRouter** | Accessing 120B+ models for free | 50 RPD hard cap, frequent provider throttling |

---

## Recommendations

| Use case | Best choice |
|---|---|
| Best overall (default) | **Gemma 4 MoE 26B [Google]** — best limits + quality |
| Fastest response | **Llama 4 Scout 17B [Groq]** — ~1.5 s |
| Fastest + high daily quota | **Gemini 3.1 Flash Lite [Google]** — 500 RPD, ~0.6 s |
| Deepest philosophical reasoning | **Qwen3 32B [Groq]** or **Llama 3.3 70B [Groq]** |
| Maximum context window | **DeepSeek V4 Flash [OR]** — 1M tokens |
| Highest model quality | **Nvidia Nemotron 120B [OR]** or **OpenAI OSS 120B [OR]** |
| High-volume / many requests/day | **Llama 3.1 8B [Groq]** — 14,400 RPD |

---

## Running the benchmark

```bash
python test_models.py
```

Requires `.env` with at least one key:
```
GOOGLE_API_KEY=...      # aistudio.google.com
GROQ_API_KEY=...        # console.groq.com
OPENROUTER_API_KEY=...  # openrouter.ai
```
