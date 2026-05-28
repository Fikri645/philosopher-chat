# LLM Model Comparison

Benchmarked on the question: *"What is Nietzsche's view on nihilism and the will to power?"*  
Test setup: RTX 3060, EmbeddingGemma-300M on CUDA, ChromaDB vectorstore (~5 700 chunks), `RETRIEVAL_K=5`.

---

## Results

| Model | Provider | Status | Time (s) | Notes |
|---|---|---|---|---|
| **Gemini 2.5 Flash Lite** | Google | ✅ Pass | 1.7 – 2.0 | Fastest Google model. Concise, accurate. Best for low-latency use. |
| **Llama 4 Scout 17B** | Groq | ✅ Pass | 1.5 – 1.6 | Fastest overall. Snappy, well-grounded answers. |
| **Llama 3.1 8B** | Groq | ✅ Pass | 1.9 – 2.0 | Smallest / lightest model. Good enough for simple queries. |
| **Gemini 2.5 Flash** | Google | ✅ Pass | 6 – 7 | Strong reasoning, balanced speed. Good default choice. |
| **Llama 3.3 70B** | Groq | ✅ Pass | 4 – 5 | Strongest Groq model. Rich, citation-dense responses. |
| **Qwen3 32B** | Groq | ✅ Pass | 5 – 6 | Thinking-mode model (`<think>` chain-of-thought). Excellent depth. |
| **Nvidia Nemotron 120B** | OpenRouter | ✅ Pass | 70 – 80 | Highest-quality free OR model. Deep philosophical analysis. Slow. |
| **OpenAI OSS 120B** | OpenRouter | ✅ Pass | 20 – 25 | Strong response quality. Best free OpenRouter option for speed/quality. |
| **Gemma 4 MoE 26B** | Google | ✅ Pass | 60 – 65 | Most accurate Gemma model on philosophical nuance. Slow (long context). |
| **Gemma 4 Dense 31B** | Google | ✅ Pass | 24 – 26 | Denser version of Gemma 4. Solid quality, faster than MoE. |
| **Gemini 2.5 Pro** | Google | ⚠️ Rate limit | — | Free tier: ~2 RPM / 50 RPD. Best model when quota allows. |
| **Gemini 2.0 Flash** | Google | ⚠️ Rate limit | — | Free tier daily quota exhausted during test. Works fine in fresh sessions. |
| **DeepSeek V4 Flash** | OpenRouter | ⚠️ Rate limit | — | Valid model; 429 due to free-tier provider cap. Works off-peak. |
| **Llama 3.3 70B** | OpenRouter | ⚠️ Rate limit | — | Same model as Groq variant; use Groq for reliability. |
| **Qwen3 Next 80B** | OpenRouter | ⚠️ Rate limit | — | Heavy model, free tier heavily rate-limited. |
| **Gemma 4 MoE 26B** | OpenRouter | ⚠️ Rate limit | — | Same weights as Google version; use Google for reliability. |

---

## Provider Summary

### Google AI Studio
- **Free tier**: generous RPM, daily quota per model
- **Best models**: Gemini 2.5 Flash (speed+quality), Gemini 2.5 Flash Lite (fastest)
- **Latency**: 2–65 s depending on model size
- **API key**: [ai.google.dev](https://ai.google.dev)

### Groq
- **Free tier**: fast LPU inference, ~30 RPM / 14 400 RPD
- **Best models**: Llama 3.3 70B (quality), Llama 4 Scout 17B (speed), Qwen3 32B (deep reasoning)
- **Latency**: 1.5–5 s — consistently the fastest provider
- **API key**: [console.groq.com](https://console.groq.com)

### OpenRouter (`:free` models)
- **Free tier**: 20 RPM / ~200 RPD per model; provider-side 429s common during peak hours
- **Best models**: OpenAI OSS 120B (speed/quality), Nvidia Nemotron 120B (depth)
- **Latency**: 20–80 s on free tier
- **API key**: [openrouter.ai](https://openrouter.ai)

---

## Recommendations

| Use case | Recommended model |
|---|---|
| Best quality (no quota concern) | Gemini 2.5 Flash or Llama 3.3 70B [Groq] |
| Fastest response | Llama 4 Scout 17B [Groq] |
| Deep philosophical reasoning | Qwen3 32B [Groq] or Nvidia Nemotron 120B [OR] |
| Lowest latency + free | Gemini 2.5 Flash Lite [Google] |
| No Google/Groq key | OpenAI OSS 120B [OpenRouter] |

---

## Running the benchmark yourself

```bash
python test_models.py
```

Requires `.env` with at least one key set:
```
GOOGLE_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```
