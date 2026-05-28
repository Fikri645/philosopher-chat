# LLM Model Comparison

Benchmarked on: *"What is Nietzsche's view on nihilism and the will to power?"*  
Setup: RTX 3060, EmbeddingGemma-300M on CUDA, ChromaDB (~5,700 chunks), `RETRIEVAL_K=5`.

---

## Full Comparison Table

| Model | Provider | Latency | Quality | RPM | TPM | RPD | TPD | Notes |
|---|---|---|---|---|---|---|---|---|
| **Gemini 2.5 Flash Lite** | Google | ~2 s | ★★★☆☆ | 15 | 250K | 1,000 | — | Fastest Google; concise answers |
| **Gemini 2.5 Flash** | Google | ~7 s | ★★★★☆ | 10 | 250K | 250 | — | Best speed/quality balance |
| **Gemini 2.5 Pro** | Google | fast* | ★★★★★ | 5 | 250K | 100 | — | Highest quality; very tight daily quota |
| **Gemini 2.0 Flash** | Google | fast* | ★★★★☆ | 10 | 250K | 250 | — | Solid, same quota as 2.5 Flash |
| **Gemma 4 MoE 26B** | Google | ~65 s | ★★★★☆ | — | — | — | — | Rate limits not published; deep analysis |
| **Gemma 4 Dense 31B** | Google | ~25 s | ★★★★☆ | — | — | — | — | Rate limits not published; faster than MoE |
| **Llama 3.1 8B** | Groq | ~2 s | ★★☆☆☆ | 14,400 | 6K | 14,400 | 500K | Lightest model; best for high-volume use |
| **Llama 4 Scout 17B** | Groq | ~1.5 s | ★★★★☆ | 1,000 | 30K | 1,000 | 500K | Fastest quality model overall |
| **Llama 3.3 70B** | Groq | ~4.5 s | ★★★★★ | 1,000 | 12K | 1,000 | 100K | Best Groq quality; tighter token quota |
| **Qwen3 32B** | Groq | ~5 s | ★★★★★ | 1,000 | 6K | 1,000 | 500K | Chain-of-thought reasoning; deepest Groq |
| **Nvidia Nemotron 120B** | OpenRouter | ~75 s | ★★★★★ | 20 | — | 50** | — | Exceptional depth; very slow on free tier |
| **OpenAI OSS 120B** | OpenRouter | ~22 s | ★★★★★ | 20 | — | 50** | — | Best free OR option; high quality |
| **DeepSeek V4 Flash** | OpenRouter | ~5 s* | ★★★★☆ | 20 | — | 50** | — | 1M context window; fast when not throttled |
| **Llama 3.3 70B** | OpenRouter | ~4 s* | ★★★★★ | 20 | — | 50** | — | Same weights as Groq; use Groq for reliability |
| **Qwen3 Next 80B** | OpenRouter | ~8 s* | ★★★★★ | 20 | — | 50** | — | Strong reasoning; frequently throttled |
| **Gemma 4 MoE 26B** | OpenRouter | ~5 s* | ★★★★☆ | 20 | — | 50** | — | Same weights as Google version |

*Latency measured when not throttled / estimated from short ping test  
**50 RPD without account credits; upgrades to 1,000 RPD with $10+ credit purchase  
— Not published by provider

---

## Rate Limit Deep-Dive

### Google AI Studio (free tier)
Source: [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)

| Model | RPM | TPM | RPD |
|---|---|---|---|
| Gemini 2.5 Pro | 5 | 250,000 | 100 |
| Gemini 2.5 Flash | 10 | 250,000 | 250 |
| Gemini 2.5 Flash Lite | 15 | 250,000 | 1,000 |
| Gemini 2.0 Flash | 10 | 250,000 | 250 |
| Gemma 4 MoE 26B | n/a | n/a | n/a |
| Gemma 4 Dense 31B | n/a | n/a | n/a |

> Gemma 4 models are available via the same `google.genai` SDK but rate limits are not documented on the public limits page.

---

### Groq (free tier)
Source: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)  
Live headers confirmed via API call.

| Model | RPM | TPM | RPD | TPD |
|---|---|---|---|---|
| Llama 3.3 70B versatile | 1,000 | 12,000 | 1,000 | 100,000 |
| Llama 4 Scout 17B | 1,000 | 30,000 | 1,000 | 500,000 |
| Qwen3 32B | 1,000 | 6,000 | 1,000 | 500,000 |
| Llama 3.1 8B instant | 14,400 | 6,000 | 14,400 | 500,000 |

> Groq is the most generous free tier by far for high-volume use. Note that `llama-3.3-70b-versatile` has a lower daily token cap (100K vs 500K) — for heavy use, prefer Llama 4 Scout or Qwen3 32B.

---

### OpenRouter (`:free` models)
Source: [openrouter.ai/docs/guides/routing/model-variants/free](https://openrouter.ai/docs/guides/routing/model-variants/free)

| Metric | Without credits | With $10+ credits |
|---|---|---|
| RPM | 20 | 20 |
| RPD | 50 | 1,000 |
| TPM / TPD | unlimited | unlimited |

> Rate limits are **identical for all `:free` models** regardless of model size. The 50 RPD cap is the reason for frequent 429 errors during testing — it's exhausted quickly. Provider-side throttling (from upstream like NVIDIA or DeepSeek) adds an additional layer of 429s independent of OpenRouter's own quota.

---

## Provider Verdict

| Provider | Best for | Bottleneck |
|---|---|---|
| **Groq** | Production-grade free use, fast iteration | TPM (6K–30K) limits long contexts |
| **Google AI Studio** | High-quality reasoning, large token budgets | RPD caps (100–1,000/day) |
| **OpenRouter** | Accessing massive models (120B+) for free | RPD cap of 50/day without credits |

---

## Recommendations

| Use case | Best choice |
|---|---|
| Default / most reliable | Gemini 2.5 Flash [Google] |
| Lowest latency | Llama 4 Scout 17B [Groq] |
| Deepest philosophical reasoning | Qwen3 32B [Groq] or Llama 3.3 70B [Groq] |
| Maximum context window | DeepSeek V4 Flash [OR] — 1M tokens |
| Highest model quality (when quota allows) | Gemini 2.5 Pro [Google] or Nvidia Nemotron 120B [OR] |
| High-volume / many requests per day | Llama 3.1 8B [Groq] — 14,400 RPD |

---

## Running the benchmark

```bash
python test_models.py
```

Requires `.env` with at least one key:
```
GOOGLE_API_KEY=...      # ai.google.dev
GROQ_API_KEY=...        # console.groq.com
OPENROUTER_API_KEY=...  # openrouter.ai
```
