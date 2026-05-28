from functools import lru_cache
from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import (
    GOOGLE_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    VECTORSTORE_DIR, RETRIEVAL_K, DEVICE
)

SYSTEM_PROMPT = (
    "You are a philosophical assistant with deep knowledge of Western philosophy, "
    "particularly nihilism, absurdism, pessimism, existentialism, and epistemology. "
    "Your answers are grounded in the primary texts of Nietzsche, Schopenhauer, Hume, and Russell.\n\n"
    "Rules:\n"
    "- Draw directly from the retrieved context passages below.\n"
    "- Always cite the philosopher and work (e.g., 'As Nietzsche writes in *Thus Spoke Zarathustra*...').\n"
    "- Be intellectually rigorous but accessible.\n"
    "- If the context doesn't contain enough to answer, say so clearly.\n"
    "- Do not moralize — present the philosophers' views faithfully."
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


def get_retriever(philosopher: str = "All"):
    vectorstore = _get_vectorstore()
    search_kwargs: dict = {"k": RETRIEVAL_K}
    if philosopher != "All":
        search_kwargs["filter"] = {"philosopher": philosopher}
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def query(input_text: str, philosopher: str = "All") -> dict:
    retriever = get_retriever(philosopher)
    docs: list[Document] = retriever.invoke(input_text)
    context_str = "\n\n".join(d.page_content for d in docs)

    client = _get_genai_client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=f"Context from philosophical texts:\n{context_str}\n\nQuestion: {input_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )

    return {"answer": response.text, "context": docs}


def vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "chroma.sqlite3").exists()
