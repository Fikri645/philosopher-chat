# Philosopher Chat

A RAG (Retrieval-Augmented Generation) chatbot grounded in Western philosophical primary texts.
Ask questions about nihilism, pessimism, epistemology, and ethics — and get answers cited directly
from the source texts.

**Live demo:** [HuggingFace Spaces — link after deploy]

---

## Knowledge Base

| Philosopher | Work |
|---|---|
| Nietzsche | *Thus Spoke Zarathustra*, *Beyond Good and Evil*, *On the Genealogy of Morality* |
| Schopenhauer | *Essays of Arthur Schopenhauer* |
| Hume | *An Enquiry Concerning Human Understanding* |
| Russell | *The Problems of Philosophy* |

All texts are public domain, sourced from [Project Gutenberg](https://www.gutenberg.org).

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Gemma 4 (`gemma-4-27b-it`) via Google Gemini API |
| Embeddings | `text-embedding-004` (Google) |
| RAG Framework | LangChain |
| Vector Store | ChromaDB (persistent) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/philosopher-chat
cd philosopher-chat
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY from https://ai.google.dev
```

### 3. Build the vectorstore (run once)

```bash
python ingest.py
```

This downloads ~6 philosophical texts from Project Gutenberg, chunks them, embeds with
`text-embedding-004`, and persists the ChromaDB vectorstore to `vectorstore/`.
Takes ~2–5 minutes depending on your connection and API rate limits.

### 4. Run the app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Deploying to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) — SDK: **Gradio**
2. In Space Settings → Secrets, add `GOOGLE_API_KEY`
3. Commit and push (include `vectorstore/` — it's the pre-built index):

```bash
git add .
git commit -m "initial commit with pre-built vectorstore"
git push
```

The app will load the pre-committed vectorstore on startup — no re-indexing needed.

---

## Experiment Tracking (MLflow)

To compare retrieval configurations (chunk size, k, etc.):

```bash
pip install mlflow
mlflow ui  # open http://localhost:5000
```

See `notebooks/retrieval_eval.ipynb` for the experiment setup.

---

## Project Structure

```
philosopher-chat/
├── app.py              ← Gradio UI
├── rag_chain.py        ← LangChain RAG pipeline
├── ingest.py           ← One-time data ingestion script
├── config.py           ← Constants (models, paths, sources)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── texts/          ← Cached raw texts (gitignored)
└── vectorstore/        ← ChromaDB persistent store (committed)
```
