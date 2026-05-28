from functools import lru_cache
from pathlib import Path
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
    CHUNK_SIZE, CHUNK_OVERLAP, DEVICE, PROVIDER_KEYS
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


def get_all_philosophers() -> list[str]:
    if not vectorstore_exists():
        return ["All"]
    result = _get_vectorstore().get(include=["metadatas"])
    names = sorted({m["philosopher"] for m in result["metadatas"] if "philosopher" in m})
    return ["All"] + names


def get_kb_stats() -> dict:
    """Returns dict with total chunks and source breakdown."""
    if not vectorstore_exists():
        return {"total": 0, "sources": {}}
    result = _get_vectorstore().get(include=["metadatas"])
    sources: dict[str, set] = {}
    for m in result["metadatas"]:
        phil = m.get("philosopher", "Unknown")
        title = m.get("title", "Unknown")
        sources.setdefault(phil, set()).add(title)
    return {"total": len(result["ids"]), "sources": sources}


def get_retriever(philosopher: str = "All"):
    vectorstore = _get_vectorstore()
    search_kwargs: dict = {"k": RETRIEVAL_K}
    if philosopher != "All":
        search_kwargs["filter"] = {"philosopher": philosopher}
    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def _call_llm(provider: str, model_id: str, context_str: str, input_text: str) -> str:
    user_content = f"Context from philosophical texts:\n{context_str}\n\nQuestion: {input_text}"

    if provider == "google":
        if not GOOGLE_API_KEY:
            env_var, site = PROVIDER_KEYS["google"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        client = _get_genai_client()
        response = client.models.generate_content(
            model=model_id,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        return response.text

    elif provider == "groq":
        if not GROQ_API_KEY:
            env_var, site = PROVIDER_KEYS["groq"]
            raise ValueError(f"{env_var} not set. Get a free key at {site}")
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content

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
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def query(input_text: str, philosopher: str = "All", llm_label: str = DEFAULT_LLM) -> dict:
    provider, model_id = LLM_OPTIONS.get(llm_label, LLM_OPTIONS[DEFAULT_LLM])
    retriever = get_retriever(philosopher)
    docs: list[Document] = retriever.invoke(input_text)
    context_str = "\n\n".join(d.page_content for d in docs)
    answer = _call_llm(provider, model_id, context_str, input_text)
    return {"answer": answer, "context": docs}


def add_to_kb(file_path: str | Path, author: str, title: str) -> int:
    """Chunk, embed, and add a file to the existing vectorstore. Returns chunk count."""
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
    chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "philosopher": author.strip(),
                "title": title.strip(),
                "source": f"{author.strip()} — *{title.strip()}*",
            },
        )
        for chunk in chunks
    ]

    _get_vectorstore().add_documents(docs)
    return len(docs)


def vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "chroma.sqlite3").exists()
