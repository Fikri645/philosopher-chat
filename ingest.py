"""
Run this script once locally to download texts and build the ChromaDB vectorstore.

    python ingest.py

The resulting `vectorstore/` directory should be committed to the repo so HuggingFace
Spaces can load it without rebuilding on every cold start.
"""

import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    DATA_DIR, VECTORSTORE_DIR, GOOGLE_API_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SOURCES, DEVICE
)

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
BATCH_SIZE = 50
RATE_LIMIT_SLEEP = 2  # seconds between embedding batches


def download_gutenberg(gutenberg_id: int, title: str) -> str:
    url = GUTENBERG_URL.format(id=gutenberg_id)
    print(f"  Downloading from {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def build_documents() -> list[Document]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs: list[Document] = []

    for source in SOURCES:
        philosopher = source["philosopher"]
        title = source["title"]
        print(f"\n[{philosopher}] {title}")

        raw = download_gutenberg(source["gutenberg_id"], title)
        if not raw:
            print("  SKIPPED (download failed)")
            continue

        cleaned = strip_gutenberg_boilerplate(raw)

        # Cache locally so you can re-run without re-downloading
        safe_name = f"{philosopher}_{title[:40].replace(' ', '_')}.txt"
        cache_path = DATA_DIR / safe_name
        cache_path.write_text(cleaned, encoding="utf-8")

        chunks = splitter.split_text(cleaned)
        for chunk in chunks:
            all_docs.append(Document(
                page_content=chunk,
                metadata={
                    "philosopher": philosopher,
                    "title": title,
                    "source": f"{philosopher} — *{title}*",
                },
            ))

        print(f"  -> {len(chunks)} chunks")
        time.sleep(1)

    return all_docs


def embed_and_store(docs: list[Document]) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Embedding device: {DEVICE}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"prompt_name": "document", "normalize_embeddings": True},
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
    )

    print(f"\nEmbedding {len(docs)} chunks in batches of {BATCH_SIZE}...")
    vectorstore = None
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name="philosophers",
                persist_directory=str(VECTORSTORE_DIR),
            )
        else:
            vectorstore.add_documents(batch)

        if batch_num < total_batches:
            time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nVectorstore saved to: {VECTORSTORE_DIR}")


def main() -> None:
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")

    docs = build_documents()
    if not docs:
        raise RuntimeError("No documents were loaded. Check your internet connection.")

    embed_and_store(docs)
    print("\nDone! Commit the `vectorstore/` directory to include it in HuggingFace Spaces.")


if __name__ == "__main__":
    main()
