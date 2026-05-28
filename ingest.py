"""
Build or update the ChromaDB vectorstore from philosophical texts.

    python ingest.py           # incremental: skips already-indexed sources
    python ingest.py --rebuild # wipes and rebuilds from scratch
"""

import sys
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    DATA_DIR, VECTORSTORE_DIR,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SOURCES, DEVICE
)

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2


def download_gutenberg(gutenberg_id: int, title: str) -> str:
    url = GUTENBERG_URL.format(id=gutenberg_id)
    print(f"  Downloading {url}")
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


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model on {DEVICE}...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"prompt_name": "document", "normalize_embeddings": True},
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
    )


def get_indexed_titles(vectorstore: Chroma) -> set[str]:
    result = vectorstore.get(include=["metadatas"])
    return {m.get("title", "") for m in result["metadatas"]}


def ingest_source(source: dict, vectorstore: Chroma, splitter: RecursiveCharacterTextSplitter) -> int:
    raw = download_gutenberg(source["gutenberg_id"], source["title"])
    if not raw:
        return 0

    cleaned = strip_gutenberg_boilerplate(raw)

    # Cache locally
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{source['philosopher']}_{source['title'][:40].replace(' ', '_')}.txt"
    (DATA_DIR / safe_name).write_text(cleaned, encoding="utf-8")

    chunks = splitter.split_text(cleaned)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "philosopher": source["philosopher"],
                "title": source["title"],
                "source": f"{source['philosopher']} — *{source['title']}*",
            },
        )
        for chunk in chunks
    ]

    for i in range(0, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[i : i + BATCH_SIZE])
        if i + BATCH_SIZE < len(docs):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return len(docs)


def main() -> None:
    rebuild = "--rebuild" in sys.argv

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if rebuild and VECTORSTORE_DIR.exists():
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
        VECTORSTORE_DIR.mkdir()
        print("Vectorstore wiped for rebuild.")

    vectorstore = Chroma(
        collection_name="philosophers",
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    already_indexed = get_indexed_titles(vectorstore) if not rebuild else set()
    total_new = 0

    for source in SOURCES:
        print(f"\n[{source['philosopher']}] {source['title']}")
        if source["title"] in already_indexed:
            print("  SKIPPED (already indexed)")
            continue

        n = ingest_source(source, vectorstore, splitter)
        if n:
            print(f"  -> {n} chunks added")
            total_new += n
        time.sleep(1)

    if total_new:
        print(f"\nDone. {total_new} new chunks added to vectorstore.")
    else:
        print("\nNothing new to index.")


if __name__ == "__main__":
    main()
