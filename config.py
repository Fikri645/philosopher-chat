import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "texts"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Gemma 4 via Gemini API — "gemma-4-26b-a4b-it" (MoE, lighter) or "gemma-4-31b-it" (dense)
LLM_MODEL = "gemma-4-26b-a4b-it"
EMBEDDING_MODEL = "google/embeddinggemma-300m"

# Auto-detect GPU; falls back to CPU if CUDA unavailable
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5

SOURCES = [
    {
        "philosopher": "Nietzsche",
        "title": "Thus Spoke Zarathustra",
        "gutenberg_id": 1998,
    },
    {
        "philosopher": "Nietzsche",
        "title": "Beyond Good and Evil",
        "gutenberg_id": 4363,
    },
    {
        "philosopher": "Nietzsche",
        "title": "On the Genealogy of Morality",
        "gutenberg_id": 52319,
    },
    {
        "philosopher": "Schopenhauer",
        "title": "Essays of Arthur Schopenhauer",
        "gutenberg_id": 11945,
    },
    {
        "philosopher": "Hume",
        "title": "An Enquiry Concerning Human Understanding",
        "gutenberg_id": 9662,
    },
    {
        "philosopher": "Russell",
        "title": "The Problems of Philosophy",
        "gutenberg_id": 5827,
    },
]

PHILOSOPHER_NAMES = ["All"] + sorted({s["philosopher"] for s in SOURCES})
