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
| **Hybrid RAG** | BM25 + semantic cosine similarity ensemble |
| **Streaming** | Token-by-token via Google / Groq / OpenRouter |
| **16 LLMs** | Gemma 4, Gemini, Llama 4, Qwen3, DeepSeek, Nemotron — all free tier |
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
| LLM routing | 16 models via Google AI Studio, Groq, OpenRouter (all free tier) |
| Embeddings | `google/embeddinggemma-300m` (HuggingFace, 768-dim) |
| Retrieval | Hybrid BM25 + ChromaDB semantic search |
| RAG Framework | LangChain LCEL (no chains, direct composition) |
| UI | Gradio 6 |
| Deployment | HuggingFace Spaces |

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
