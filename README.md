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
| **Two-stage retrieval** | Hybrid (dense + BM25) fused with RRF → cross-encoder reranking |
| **RAGAS evaluation** | 4 metrics measured with/without reranking — reranking quantified, not assumed |
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
| Retrieval | Hybrid (dense + BM25) → RRF fusion → cross-encoder rerank |
| Reranker | `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder) |
| Evaluation | RAGAS metrics (faithfulness, relevancy, context precision/recall) |
| RAG Framework | LangChain LCEL (no chains, direct composition) |
| UI | Gradio 6 |
| Deployment | HuggingFace Spaces |

---

## Retrieval Architecture

```
Question
   │
   ├─ Dense retrieval   (EmbeddingGemma-300M → ChromaDB cosine)  ─┐
   ├─ Sparse retrieval  (BM25 / rank-bm25)                        ├─ RRF fusion → top-20 pool
   │                                                             ─┘
   ├─ Cross-encoder rerank  (BGE-reranker-v2-m3) → top-6
   │
   └─ LLM answer  (grounded + cited from top-6 chunks)
```

Two-stage retrieval is the modern production pattern: cheap recall first (hybrid), then
a precise but expensive cross-encoder reranks the small candidate pool. The reranker
scores each `(query, chunk)` pair jointly rather than comparing pre-computed vectors.

## Evaluation

The pipeline is measured, not assumed. [`evaluate.py`](evaluate.py) runs four
[RAGAS](https://docs.ragas.io) metrics over a curated question set with reference
answers, **with and without** the reranker, and writes `eval_results.json` (rendered
live in the app's **📊 Evaluation** tab). See the
[evaluation notebook](notebooks/rag_evaluation.ipynb) for the full analysis.

```bash
pip install -r requirements.txt -r requirements-eval.txt
python evaluate.py        # ~12 min; writes eval_results.json
```

| Metric | Measures |
|---|---|
| **Faithfulness** | Answer claims supported by retrieved context (anti-hallucination) |
| **Answer Relevancy** | Answer actually addresses the question |
| **Context Precision** | Relevant chunks ranked near the top |
| **Context Recall** | Reference answer covered by retrieved context |

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
