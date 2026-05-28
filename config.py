import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "texts"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# LLM options — (provider, model_id)
# Providers: "google" | "groq" | "openrouter"
# ---------------------------------------------------------------------------
LLM_OPTIONS: dict[str, tuple[str, str]] = {
    # ── Google AI Studio (free tier) ──────────────────────────────────────
    "Gemma 4 MoE 26B  [Google]":         ("google",      "gemma-4-26b-a4b-it"),
    "Gemma 4 Dense 31B  [Google]":       ("google",      "gemma-4-31b-it"),
    "Gemini 2.5 Flash  [Google]":        ("google",      "gemini-2.5-flash"),
    "Gemini 2.5 Flash Lite  [Google]":   ("google",      "gemini-2.5-flash-lite"),
    "Gemini 2.5 Pro  [Google]":          ("google",      "gemini-2.5-pro"),
    "Gemini 2.0 Flash  [Google]":        ("google",      "gemini-2.0-flash"),
    # ── Groq (free tier, very fast LPU inference) ─────────────────────────
    "Llama 3.3 70B  [Groq]":             ("groq",        "llama-3.3-70b-versatile"),
    "Llama 4 Scout 17B  [Groq]":         ("groq",        "meta-llama/llama-4-scout-17b-16e-instruct"),
    "Qwen3 32B  [Groq]":                 ("groq",        "qwen/qwen3-32b"),
    "Llama 3.1 8B  [Groq]":              ("groq",        "llama-3.1-8b-instant"),
    # ── OpenRouter free models (:free = no cost, rate-limited) ────────────
    "Nvidia Nemotron 120B  [OpenRouter]":("openrouter",  "nvidia/nemotron-3-super-120b-a12b:free"),
    "OpenAI OSS 120B  [OpenRouter]":     ("openrouter",  "openai/gpt-oss-120b:free"),
    "DeepSeek V4 Flash  [OpenRouter]":   ("openrouter",  "deepseek/deepseek-v4-flash:free"),
    "Llama 3.3 70B  [OpenRouter]":       ("openrouter",  "meta-llama/llama-3.3-70b-instruct:free"),
    "Qwen3 Next 80B  [OpenRouter]":      ("openrouter",  "qwen/qwen3-next-80b-a3b-instruct:free"),
    "Gemma 4 MoE 26B  [OpenRouter]":     ("openrouter",  "google/gemma-4-26b-a4b-it:free"),
}

DEFAULT_LLM = "Gemma 4 MoE 26B  [Google]"

PROVIDER_KEYS = {
    "google":     ("GOOGLE_API_KEY",     "ai.google.dev"),
    "groq":       ("GROQ_API_KEY",       "console.groq.com"),
    "openrouter": ("OPENROUTER_API_KEY", "openrouter.ai"),
}

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_OPTIONS = {
    "EmbeddingGemma 300M (active)": "google/embeddinggemma-300m",
    "BGE Large EN v1.5":            "BAAI/bge-large-en-v1.5",
    "Multilingual E5 Large":        "intfloat/multilingual-e5-large",
}
DEFAULT_EMBEDDING = "EmbeddingGemma 300M (active)"
EMBEDDING_MODEL   = EMBEDDING_OPTIONS[DEFAULT_EMBEDDING]

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K   = 5

# ---------------------------------------------------------------------------
# Knowledge base sources (Project Gutenberg)
# ---------------------------------------------------------------------------
SOURCES = [
    {"philosopher": "Nietzsche",       "title": "Thus Spoke Zarathustra",                              "gutenberg_id": 1998},
    {"philosopher": "Nietzsche",       "title": "Beyond Good and Evil",                                "gutenberg_id": 4363},
    {"philosopher": "Nietzsche",       "title": "On the Genealogy of Morality",                        "gutenberg_id": 52319},
    {"philosopher": "Nietzsche",       "title": "The Birth of Tragedy",                                "gutenberg_id": 51356},
    {"philosopher": "Schopenhauer",    "title": "Essays of Arthur Schopenhauer",                       "gutenberg_id": 11945},
    {"philosopher": "Hume",            "title": "An Enquiry Concerning Human Understanding",           "gutenberg_id": 9662},
    {"philosopher": "Russell",         "title": "The Problems of Philosophy",                          "gutenberg_id": 5827},
    {"philosopher": "Marcus Aurelius", "title": "Meditations",                                         "gutenberg_id": 2680},
    {"philosopher": "Plato",           "title": "The Republic",                                        "gutenberg_id": 1497},
    {"philosopher": "Mill",            "title": "Utilitarianism",                                      "gutenberg_id": 11224},
    {"philosopher": "Epictetus",       "title": "The Enchiridion",                                     "gutenberg_id": 45109},
    {"philosopher": "Kant",            "title": "Fundamental Principles of the Metaphysic of Morals", "gutenberg_id": 5682},
]
