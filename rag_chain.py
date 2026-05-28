import re
from functools import lru_cache
from pathlib import Path
from typing import Generator

from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    GOOGLE_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
    LLM_OPTIONS, DEFAULT_LLM,
    EMBEDDING_MODEL, VECTORSTORE_DIR, RETRIEVAL_K,
    CHUNK_SIZE, CHUNK_OVERLAP, DEVICE, PROVIDER_KEYS,
    USE_HYBRID_SEARCH, MAX_HISTORY_TURNS,
)

SYSTEM_PROMPT = (
    "You are a philosophical assistant with deep knowledge of Western philosophy, "
    "particularly nihilism, absurdism, pessimism, existentialism, and epistemology. "
    "Your answers are grounded in the primary texts provided as context.\n\n"
    "Rules:\n"
    "- Draw directly from the retrieved context passages.\n"
    "- Always cite the philosopher and work "
    "(e.g., 'As Nietzsche writes in *Thus Spoke Zarathustra*...').\n"
    "- Be intellectually rigorous but accessible.\n"
    "- If the context is insufficient, say so clearly.\n"
    "- Present the philosophers' views faithfully without moralizing."
)


def _trim_history(history: list[dict] | None) -> list[dict]:
    """Keep only the most recent MAX_HISTORY_TURNS turns (user+assistant pairs).

    One turn = one user message + one assistant message = 2 list items.
    Older turns are dropped to stay within 32 K-context model limits.
    """
    if not history:
        return []
    # Each turn is 2 items; keep the last MAX_HISTORY_TURNS * 2 messages
    cutoff = MAX_HISTORY_TURNS * 2
    return history[-cutoff:] if len(history) > cutoff else list(history)


def _clean_for_history(text: str) -> str:
    """Strip HTML tags and source footer from stored assistant messages.

    Assistant responses contain <details>/<div> think blocks and a
    '--- **Sources:**' footer injected by the UI — remove both before
    passing prior turns as LLM history, so models see clean prose only.
    """
    text = re.sub(r"<[^>]+>", " ", text)                              # strip HTML
    text = re.sub(r"\n\n---\n\*\*Sources:\*\*.*$", "", text,          # strip footer
                  flags=re.DOTALL)
    return " ".join(text.split())                                      # normalise whitespace


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    return genai.Client(api_key=GOOGLE_API_KEY)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"prompt_name": "document", "normalize_embeddings": True},
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="philosophers",
        embedding_function=_get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


@lru_cache(maxsize=1)
def _get_bm25_retriever():
    """Build BM25 index over the full KB (cached after first call)."""
    from langchain_community.retrievers import BM25Retriever  # requires rank-bm25
    result = _get_vectorstore().get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(result["documents"], result["metadatas"])
        if d.strip()
    ]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = RETRIEVAL_K
    return retriever


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "chroma.sqlite3").exists()


def get_all_philosophers() -> list[str]:
    if not vectorstore_exists():
        return ["All"]
    result = _get_vectorstore().get(include=["metadatas"])
    names = sorted({m["philosopher"] for m in result["metadatas"] if "philosopher" in m})
    return ["All"] + names


def get_kb_stats() -> dict:
    if not vectorstore_exists():
        return {"total": 0, "sources": {}}
    result = _get_vectorstore().get(include=["metadatas"])
    sources: dict[str, set] = {}
    for m in result["metadatas"]:
        phil = m.get("philosopher", "Unknown")
        title = m.get("title", "Unknown")
        sources.setdefault(phil, set()).add(title)
    return {"total": len(result["ids"]), "sources": sources}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_docs(
    input_text: str, philosopher: str = "All"
) -> tuple[list[Document], list[float]]:
    """Hybrid BM25 + semantic retrieval.

    Returns (docs, scores) where scores are cosine relevance ∈ [0, 1].
    BM25-only results are tagged with score -1.0 (no embedding similarity).
    """
    vectorstore = _get_vectorstore()
    search_kwargs: dict = {"k": RETRIEVAL_K}
    if philosopher != "All":
        search_kwargs["filter"] = {"philosopher": philosopher}

    pairs = vectorstore.similarity_search_with_relevance_scores(input_text, **search_kwargs)

    if USE_HYBRID_SEARCH and philosopher == "All":
        try:
            bm25_docs = _get_bm25_retriever().invoke(input_text)
            seen = {doc.page_content for doc, _ in pairs}
            for doc in bm25_docs[:2]:
                if doc.page_content not in seen:
                    pairs.append((doc, -1.0))
                    seen.add(doc.page_content)
        except Exception:
            pass

    # Sort: semantic scores descending, BM25 appended at end
    semantic = sorted([(d, s) for d, s in pairs if s >= 0], key=lambda x: x[1], reverse=True)
    bm25_only = [(d, s) for d, s in pairs if s < 0]
    pairs = (semantic + bm25_only)[: RETRIEVAL_K + 2]

    return [d for d, _ in pairs], [s for _, s in pairs]


# ---------------------------------------------------------------------------
# LLM calls — non-streaming
# ---------------------------------------------------------------------------

def _call_llm(
    provider: str, model_id: str, context_str: str, input_text: str,
    history: list[dict] | None = None,
) -> str:
    final_user = f"Context from philosophical texts:\n{context_str}\n\nQuestion: {input_text}"

    if provider == "google":
        if not GOOGLE_API_KEY:
            env_var, site = PROVIDER_KEYS["google"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        contents = []
        for turn in _trim_history(history):
            role = "model" if turn["role"] == "assistant" else "user"
            content = _clean_for_history(turn["content"]) if turn["role"] == "assistant" else turn["content"]
            if content:
                contents.append({"role": role, "parts": [content]})
        contents.append({"role": "user", "parts": [final_user]})
        response = _get_genai_client().models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.3
            ),
        )
        return response.text

    elif provider == "groq":
        if not GROQ_API_KEY:
            env_var, site = PROVIDER_KEYS["groq"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            env_var, site = PROVIDER_KEYS["openrouter"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://github.com/Fikri645/philosopher-chat"},
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in _trim_history(history):
        role = "assistant" if turn["role"] == "assistant" else "user"
        content = _clean_for_history(turn["content"]) if turn["role"] == "assistant" else turn["content"]
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": final_user})
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# LLM calls — streaming
# ---------------------------------------------------------------------------

def stream_llm(
    provider: str, model_id: str, context_str: str, input_text: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Yield text chunks for real-time streaming.

    history: previous turns as [{"role": "user"|"assistant", "content": "..."}].
    Pass all completed turns so the model understands follow-up questions.
    """
    final_user = f"Context from philosophical texts:\n{context_str}\n\nQuestion: {input_text}"

    if provider == "google":
        if not GOOGLE_API_KEY:
            env_var, site = PROVIDER_KEYS["google"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        contents = []
        for turn in _trim_history(history):
            role = "model" if turn["role"] == "assistant" else "user"
            content = _clean_for_history(turn["content"]) if turn["role"] == "assistant" else turn["content"]
            if content:
                contents.append({"role": role, "parts": [content]})
        contents.append({"role": "user", "parts": [final_user]})
        for chunk in _get_genai_client().models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.3
            ),
        ):
            if chunk.text:
                yield chunk.text

    elif provider in ("groq", "openrouter"):
        if provider == "groq":
            if not GROQ_API_KEY:
                env_var, site = PROVIDER_KEYS["groq"]
                raise ValueError(f"{env_var} not set. Get a free key at {site}")
            from openai import OpenAI
            client = OpenAI(
                api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
            )
        else:
            if not OPENROUTER_API_KEY:
                env_var, site = PROVIDER_KEYS["openrouter"]
                raise ValueError(f"{env_var} not set. Get a free key at {site}")
            from openai import OpenAI
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Fikri645/philosopher-chat"
                },
            )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in _trim_history(history):
            role = "assistant" if turn["role"] == "assistant" else "user"
            content = _clean_for_history(turn["content"]) if turn["role"] == "assistant" else turn["content"]
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": final_user})
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

def query(
    input_text: str, philosopher: str = "All", llm_label: str = DEFAULT_LLM
) -> dict:
    """Non-streaming query. Returns answer + context + scores."""
    provider, model_id = LLM_OPTIONS.get(llm_label, LLM_OPTIONS[DEFAULT_LLM])
    docs, scores = retrieve_docs(input_text, philosopher)
    context_str = "\n\n".join(d.page_content for d in docs)
    answer = _call_llm(provider, model_id, context_str, input_text)
    return {"answer": answer, "context": docs, "scores": scores}


# ---------------------------------------------------------------------------
# UMAP embedding visualization
# ---------------------------------------------------------------------------

def get_umap_data() -> dict | None:
    """Compute 2D UMAP projection of all KB embeddings.

    Returns dict ready for plotly, or None if unavailable.
    """
    import numpy as np

    try:
        import umap as umap_module  # type: ignore
    except ImportError:
        return None

    if not vectorstore_exists():
        return None

    result = _get_vectorstore().get(include=["embeddings", "metadatas", "documents"])
    embeddings_raw = result.get("embeddings")
    if embeddings_raw is None or len(embeddings_raw) == 0:
        return None

    embeddings = np.array(embeddings_raw)
    reducer = umap_module.UMAP(
        n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
    )
    coords = reducer.fit_transform(embeddings)

    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "philosopher": [m.get("philosopher", "Unknown") for m in result["metadatas"]],
        "title": [m.get("title", "Unknown") for m in result["metadatas"]],
        "preview": [d[:120].replace("\n", " ") + "…" for d in result["documents"]],
    }


# ---------------------------------------------------------------------------
# KB management
# ---------------------------------------------------------------------------

def add_to_kb(file_path: str | Path, author: str, title: str) -> int:
    """Chunk, embed, and add a file to the vectorstore. Returns chunk count."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = "\n\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        raise ValueError("Could not extract text from the uploaded file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "philosopher": author.strip(),
                "title": title.strip(),
                "source": f"{author.strip()} — *{title.strip()}*",
            },
        )
        for chunk in splitter.split_text(text)
    ]

    _get_vectorstore().add_documents(docs)
    _get_bm25_retriever.cache_clear()  # invalidate BM25 index after KB change
    return len(docs)
