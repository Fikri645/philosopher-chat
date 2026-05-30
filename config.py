import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Authenticate with HuggingFace Hub so gated models (e.g. EmbeddingGemma-300M) can be downloaded.
# On HF Spaces, set HF_TOKEN in Settings → Variables and secrets.
_HF_TOKEN = os.getenv("HF_TOKEN", "")
if _HF_TOKEN:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_HF_TOKEN, add_to_git_credential=False)

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
    # Limits verified from aistudio.google.com/rate-limit (2026-05)
    "Gemma 4 MoE 26B  [Google]":         ("google",      "gemma-4-26b-a4b-it"),    # 15 RPM | ∞ TPM | 1500 RPD
    "Gemma 4 Dense 31B  [Google]":       ("google",      "gemma-4-31b-it"),         # 15 RPM | ∞ TPM | 1500 RPD
    "Gemini 3.1 Flash Lite  [Google]":   ("google",      "gemini-3.1-flash-lite"),  # 15 RPM | 250K TPM | 500 RPD
    "Gemini 3.5 Flash  [Google]":        ("google",      "gemini-3.5-flash"),       #  5 RPM | 250K TPM |  20 RPD
    "Gemini 2.5 Flash  [Google]":        ("google",      "gemini-2.5-flash"),       #  5 RPM | 250K TPM |  20 RPD
    "Gemini 2.5 Flash Lite  [Google]":   ("google",      "gemini-2.5-flash-lite"),  # 10 RPM | 250K TPM |  20 RPD
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
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
RETRIEVAL_K       = 6       # final number of chunks passed to the LLM
USE_HYBRID_SEARCH = True    # BM25 + semantic ensemble (fused with RRF)

# ---------------------------------------------------------------------------
# Reranking (2-stage retrieval)
#   Stage 1: hybrid (semantic + BM25) → fetch RETRIEVAL_FETCH_K candidates,
#            merged with Reciprocal Rank Fusion (RRF).
#   Stage 2: cross-encoder reranker scores each (query, chunk) pair jointly
#            and keeps the top RETRIEVAL_K. Highest-ROI precision boost.
# ---------------------------------------------------------------------------
USE_RERANKER      = True
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"  # multilingual (handles ID queries)
RETRIEVAL_FETCH_K = 20      # candidates retrieved before reranking
RRF_K             = 60      # RRF damping constant (standard default)

# Max number of *turns* (1 turn = 1 user + 1 assistant message) to keep in
# LLM history. Each RAG turn adds ~7 000 tokens (6 chunks + Q + A), so 4 turns
# ≈ 28 K tokens — safely under the 32 K limit of Gemma/Qwen3 while leaving
# room for the system prompt and the new question+context.
MAX_HISTORY_TURNS = 4

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
