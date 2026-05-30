---
title: Philosopher Chat
emoji: 🏛️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
pinned: false
license: mit
---

# Philosopher Chat

A RAG (Retrieval-Augmented Generation) chatbot grounded in Western philosophical primary texts.
Ask questions about nihilism, existentialism, epistemology, ethics, and more — answers are
cited directly from 12 primary texts (~5,700 chunks).

**Live demo:** [fikri0o0/philosopher-chat on HuggingFace Spaces](https://huggingface.co/spaces/fikri0o0/philosopher-chat)

---

## Features

| Feature | Detail |
|---|---|
| **Query rewriting** | Multi-query expansion (LLM paraphrases) fused with RRF for better recall |
| **Two-stage retrieval** | Hybrid (dense + BM25) → RRF → cross-encoder reranking |
| **Corrective RAG** | Abstains when retrieval confidence is low instead of hallucinating |
| **RAGAS evaluation** | 4 metrics, 3-stage ablation — each component quantified, not assumed |
| **Streaming** | Token-by-token via Google / Groq / OpenRouter |
| **15 LLMs** | Gemma 4, Gemini, Llama 4, Qwen3, DeepSeek, Nemotron — all free tier |
| **Think blocks** | Qwen3 / DeepSeek reasoning rendered as collapsible chains-of-thought |
| **UMAP viz** | 2D projection of all 5,700+ embeddings coloured by philosopher |
| **Model comparison** | Side-by-side latency + quality comparison across any two models |
| **Extendable KB** | Upload your own PDF/TXT to add new philosophers |

---

## Knowledge Base

| Philosopher | Works |
|---|---|
| Nietzsche | *Thus Spoke Zarathustra*, *Beyond Good and Evil*, *On the Genealogy of Morality*, *The Birth of Tragedy* |
| Schopenhauer | *Essays of Arthur Schopenhauer* |
| Hume | *An Enquiry Concerning Human Understanding* |
| Russell | *The Problems of Philosophy* |
| Marcus Aurelius | *Meditations* |
| Plato | *The Republic* |
| Mill | *Utilitarianism* |
| Epictetus | *The Enchiridion* |
| Kant | *Fundamental Principles of the Metaphysic of Morals* |

All texts are public domain, sourced from [Project Gutenberg](https://www.gutenberg.org).

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM routing | 15 models via Google AI Studio, Groq, OpenRouter (all free tier) |
| Embeddings | `google/embeddinggemma-300m` (HuggingFace, 768-dim) |
| Query transform | Multi-query rewriting (LLM paraphrases → RRF) |
| Retrieval | Hybrid (dense + BM25) → RRF fusion → cross-encoder rerank |
| Reranker | `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder) |
| Guardrail | Corrective RAG — cosine-gated abstention on out-of-corpus queries |
| Evaluation | RAGAS metrics (faithfulness, relevancy, context precision/recall) |
| RAG Framework | LangChain LCEL (no chains, direct composition) |
| UI | Gradio 6 |
| Deployment | HuggingFace Spaces |

---

## Retrieval Architecture

```
Question
   │
   ├─ Query rewriting   (LLM → original + paraphrases)            ─┐
   │     each variant ↓                                            │
   ├─ Dense retrieval   (EmbeddingGemma-300M → ChromaDB cosine)    ├─ RRF fusion → top-20 pool
   ├─ Sparse retrieval  (BM25 / rank-bm25)                        ─┘
   │
   ├─ Cross-encoder rerank  (BGE-reranker-v2-m3) → top-6
   │
   ├─ Corrective gate  (cosine < threshold → abstain)
   │
   └─ LLM answer  (grounded + cited from top-6 chunks)
```

The pattern follows modern production RAG: cheap recall first (multi-query hybrid),
a precise cross-encoder rerank of the small pool, and an abstention gate so
out-of-corpus questions get an honest "I don't know" instead of a hallucination.

## Evaluation

The pipeline is **measured, not assumed**. [`evaluate.py`](evaluate.py) generates
answers for a curated question set with reference answers across a **3-stage ablation**
(baseline → + reranker → + query rewrite), then an LLM judge scores four
[RAGAS](https://docs.ragas.io) metrics. Results render live in the app's
**📊 Evaluation** tab; full analysis in the
[evaluation notebook](notebooks/rag_evaluation.ipynb).

### Measuring each component (12 questions, LLM-as-judge)

| Metric | Baseline (Hybrid) | + Reranker | + Query Rewrite | Δ (full) |
|---|:---:|:---:|:---:|:---:|
| **Faithfulness** | 0.40 | 0.44 | 0.43 | +0.03 |
| **Answer Relevancy** | 0.94 | 0.90 | 0.91 | −0.03 |
| **Context Precision** | 1.00 | 1.00 | 0.99 | −0.01 |
| **Context Recall** | 0.38 | 0.51 | 0.43 | +0.06 |

**The reranker is the clear win** — Context Recall jumps +0.14 (0.38 → 0.51) and
Faithfulness rises, with no real cost elsewhere. **Query rewriting did *not* help
this corpus** — it slightly *reduced* recall (0.51 → 0.43) and relevancy. That is the
point of measuring: the data says ship the reranker and treat query rewriting as
corpus-dependent rather than assuming more components are always better. Two-phase
eval (generation, then judging) keeps it reproducible:

```bash
pip install -r requirements.txt
pip install --no-deps ragas && pip install -r requirements-eval.txt
python evaluate.py --generate   # phase A: real retrieval + generation → eval_samples.json
# phase B: an LLM judge scores eval_samples.json → eval_results.json
```

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/Fikri645/philosopher-chat
cd philosopher-chat
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
# Create .env with your keys:
GOOGLE_API_KEY=...       # https://ai.google.dev  (free)
GROQ_API_KEY=...         # https://console.groq.com  (free)
OPENROUTER_API_KEY=...   # https://openrouter.ai  (free)
HF_TOKEN=...             # https://huggingface.co/settings/tokens  (for gated EmbeddingGemma)
```

### 3. Build the vectorstore (run once)

```bash
python ingest.py
```

Downloads 12 texts from Project Gutenberg, chunks them, embeds with EmbeddingGemma-300M,
and persists to `vectorstore/`. Takes ~5–10 min on first run (model download + embedding).

### 4. Run the app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Deploying to HuggingFace Spaces

1. Fork or push to a new Space (SDK: **Gradio**)
2. In **Space Settings → Variables and Secrets**, add:
   - `GOOGLE_API_KEY`
   - `GROQ_API_KEY`
   - `OPENROUTER_API_KEY`
   - `HF_TOKEN` (your HF token — needed to download the gated EmbeddingGemma model)
3. On first boot the Space auto-ingests all 12 texts (~10 min); subsequent boots load the cached vectorstore.

---

## Project Structure

```
philosopher-chat/
├── app.py              ← Gradio UI + event handlers
├── rag_chain.py        ← LangChain RAG pipeline (retrieval + LLM routing)
├── ingest.py           ← Data ingestion from Project Gutenberg
├── config.py           ← LLM options, embedding model, RAG parameters
├── requirements.txt
├── .gitignore
└── README.md
```
