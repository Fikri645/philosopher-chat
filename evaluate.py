"""
RAGAS evaluation of the retrieval pipeline.

Uses the official RAGAS library (https://docs.ragas.io) to score the pipeline
on four metrics — once with the cross-encoder reranker OFF (hybrid baseline)
and once ON — over a curated question set with reference answers:

    • Faithfulness        — answer claims supported by the retrieved context
    • Answer Relevancy    — answer actually addresses the question
    • Context Precision   — relevant chunks ranked near the top (with reference)
    • Context Recall      — reference answer covered by the retrieved context

Results are written to ``eval_results.json``, which the Gradio app renders in
its "📊 Evaluation" tab.

The judge is Llama 3.1 8B (Groq, OpenAI-compatible) wrapped for RAGAS;
answer relevancy uses the project's EmbeddingGemma embeddings.

Usage:
    pip install -r requirements.txt -r requirements-eval.txt
    python evaluate.py            # full run, both configs
    python evaluate.py --quick    # first 4 questions only (smoke test)
"""

from __future__ import annotations

import io
import json
import sys
import time
import types
from datetime import datetime, timezone

# Force UTF-8 console output on Windows (Δ, etc.)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Compatibility shim: RAGAS 0.4.3 hard-imports
# ``langchain_community.chat_models.vertexai`` (used only in an internal list of
# supported model types), but that submodule was removed in langchain-community
# 0.4.x. We're on the modern LangChain 1.x stack, so we stub the removed module
# *before* importing ragas. The stub is never exercised — our judge is ChatOpenAI
# (Groq) — so this keeps the full langchain 1.x app stack intact with no downgrade.
# ---------------------------------------------------------------------------
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vx = types.ModuleType("langchain_community.chat_models.vertexai")
    _vx.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated
    sys.modules["langchain_community.chat_models.vertexai"] = _vx

import numpy as np  # noqa: E402

# NOTE: ragas / langchain_openai are imported lazily inside the judging path
# only. The ``--generate`` phase (torch: embedder + reranker + Chroma) then runs
# with exactly the app's import surface — importing the heavy ragas stack
# alongside torch was triggering a native segfault on this Windows / Python 3.14
# box. Generation and judging are therefore split into two processes.

import rag_chain  # noqa: E402
from config import GOOGLE_API_KEY, RERANKER_MODEL, RETRIEVAL_FETCH_K  # noqa: E402

# NOTE: RAGAS's batch ``evaluate()`` uses an async executor whose per-job
# ``asyncio.timeout`` is incompatible with Python 3.14's asyncio. We instead
# call each metric's synchronous ``single_turn_score`` in a plain loop — same
# RAGAS metric implementations and prompts, just driven sequentially.

# Judge — Gemini 3.1 Flash Lite via Google's OpenAI-compatible endpoint (httpx,
# NOT the grpc client, which segfaults alongside torch on Python 3.14). 500 RPD
# / 250K TPM gives headroom for a full 12-question A/B. (Groq free tiers proved
# too token-limited; Gemma's thinking mode breaks RAGAS JSON parsing.)
JUDGE_MODEL = "gemini-3.1-flash-lite"
GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Answer generation runs on Google Gemma (the app's default model) — a separate
# rate-limit bucket from the judge.
GEN_LLM_LABEL = "Gemma 4 MoE 26B  [Google]"
RESULTS_PATH = rag_chain.VECTORSTORE_DIR.parent / "eval_results.json"

METRIC_ORDER = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Curated evaluation set — questions + reference (ground-truth) answers
# grounded in the 12 primary texts in the knowledge base.
# ---------------------------------------------------------------------------
EVAL_SET: list[dict] = [
    {
        "question": "What does Nietzsche mean by the death of God?",
        "reference": (
            "Nietzsche uses the death of God to describe the collapse of "
            "belief in a transcendent source of meaning and morality. It is "
            "not a literal claim but a diagnosis of modern culture: with the "
            "divine foundation gone, inherited values lose their ground, "
            "risking nihilism, and humanity must create new values itself."
        ),
    },
    {
        "question": "How does Schopenhauer view suffering and the will to live?",
        "reference": (
            "Schopenhauer holds that the will to live is the blind, insatiable "
            "force underlying all existence. Because desire is endless and its "
            "satisfaction fleeting, life is dominated by suffering. Relief comes "
            "only through denial of the will, aesthetic contemplation, and "
            "compassion."
        ),
    },
    {
        "question": "What is Hume's argument about causation?",
        "reference": (
            "Hume argues that we never perceive a necessary connection between "
            "cause and effect, only the constant conjunction of events. Our "
            "belief in causation is a habit of the mind formed by repeated "
            "experience, not a truth derived from reason."
        ),
    },
    {
        "question": "Can we have certain knowledge of the external world according to Russell?",
        "reference": (
            "Russell distinguishes knowledge by acquaintance from knowledge by "
            "description. We are directly acquainted only with sense-data; the "
            "existence of physical objects is an inference. He argues the "
            "external world is the best hypothesis explaining our sense-data, "
            "though not known with absolute certainty."
        ),
    },
    {
        "question": "What is Kant's categorical imperative?",
        "reference": (
            "Kant's categorical imperative is an unconditional moral law: act "
            "only on a maxim you could will to become a universal law. It "
            "commands independently of desires or consequences, and requires "
            "treating humanity always as an end and never merely as a means."
        ),
    },
    {
        "question": "How does Mill justify the principle of utility?",
        "reference": (
            "Mill grounds morality in the greatest happiness principle: actions "
            "are right insofar as they promote happiness and wrong as they "
            "produce its reverse. He argues happiness is the sole thing desired "
            "as an end, and distinguishes higher (intellectual) from lower "
            "(bodily) pleasures by quality, not only quantity."
        ),
    },
    {
        "question": "What does Marcus Aurelius advise about things outside our control?",
        "reference": (
            "Marcus Aurelius, following Stoic doctrine, advises accepting what "
            "is outside our control as part of nature's order, and focusing only "
            "on our own judgments and actions. Externals cannot harm the rational "
            "self; disturbance comes from our opinions about events, not events "
            "themselves."
        ),
    },
    {
        "question": "What is Epictetus's distinction between what is in our power and what is not?",
        "reference": (
            "Epictetus opens the Enchiridion by dividing things into those in our "
            "power — our opinions, desires, and aversions — and those not — body, "
            "property, reputation. Tranquility comes from caring only about what "
            "is in our power and treating the rest with indifference."
        ),
    },
    {
        "question": "What is Plato's ideal society in The Republic?",
        "reference": (
            "Plato's ideal state is a just city ordered into three classes — "
            "rulers (philosopher-kings), guardians, and producers — mirroring the "
            "soul's reason, spirit, and appetite. Justice is each part performing "
            "its proper role. The guardian class shares property and family, and "
            "rulers are chosen for wisdom and educated to know the Good."
        ),
    },
    {
        "question": "What is the will to power in Nietzsche's thought?",
        "reference": (
            "For Nietzsche the will to power is the fundamental drive of life: "
            "not mere survival but the striving to grow, overcome, and impose "
            "form. It underlies values and actions, and the higher type affirms "
            "life by creating values out of this creative, self-overcoming force."
        ),
    },
    {
        "question": "How does Nietzsche characterize master and slave morality?",
        "reference": (
            "In the Genealogy of Morality, Nietzsche contrasts master morality, "
            "which originates in the strong and calls 'good' what is noble and "
            "powerful, with slave morality, which arises from the resentment of "
            "the weak and revalues humility, meekness, and pity as good while "
            "branding the strong as evil."
        ),
    },
    {
        "question": "What role does eternal recurrence play in Nietzsche's philosophy?",
        "reference": (
            "Eternal recurrence is the thought that one's life will repeat "
            "identically and infinitely. Nietzsche poses it as a test of life-"
            "affirmation: to will the eternal return of every moment, including "
            "suffering, is the highest expression of amor fati and saying yes to "
            "existence."
        ),
    },
]


# ---------------------------------------------------------------------------
# RAGAS plumbing
# ---------------------------------------------------------------------------

def _build_judge():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=GOOGLE_API_KEY,
        base_url=GOOGLE_OPENAI_BASE,
        temperature=0.0,
        max_tokens=3000,
        timeout=90,
        max_retries=4,       # absorb the occasional 15-RPM 429
    )
    return LangchainLLMWrapper(llm)


# Gemini 3.1 Flash Lite allows 15 RPM. A short pace between metrics keeps
# multi-call metrics under the per-minute request limit.
PACE_SECONDS = 5


def _score_sample(sample: SingleTurnSample, scorers: dict) -> dict[str, float | None]:
    """Score one sample on all four metrics; isolate failures per-metric."""
    out: dict[str, float | None] = {}
    for i, (canon, metric) in enumerate(scorers.items()):
        if i:
            time.sleep(PACE_SECONDS)  # respect the 6000 TPM bucket
        try:
            v = float(metric.single_turn_score(sample))
            out[canon] = None if v != v else round(v, 3)  # NaN → None
        except Exception as exc:
            print(f"      {canon} failed: {str(exc)[:70]}")
            out[canon] = None
    return out


# Generate with Gemma via Google's OpenAI-compatible endpoint (httpx). The
# native google.genai client uses grpc, which segfaults alongside torch on this
# Python 3.14 box — so we keep generation on the same httpx path as the judge.
GEN_MODEL_ID = "gemma-4-26b-a4b-it"


def _generate(question: str) -> dict:
    docs, _ = rag_chain.retrieve_docs(question, "All")  # torch retrieval (no grpc)
    context_str = "\n\n".join(d.page_content for d in docs)
    from openai import OpenAI
    client = OpenAI(api_key=GOOGLE_API_KEY, base_url=GOOGLE_OPENAI_BASE,
                    timeout=90, max_retries=2)  # avoid indefinite hangs
    user = (f"Relevant passages from your knowledge base:\n{context_str}\n\n"
            f"Question: {question}")
    resp = client.chat.completions.create(
        model=GEN_MODEL_ID,
        messages=[
            {"role": "system", "content": rag_chain.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        timeout=90,
    )
    return {"answer": resp.choices[0].message.content, "context": docs}


def _generate_with_retry(question: str, retries: int = 5):
    """RAG answer generation with backoff on Google RPM (429) limits."""
    for attempt in range(retries):
        try:
            return _generate(question)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 8 * (attempt + 1)
            print(f"    generation retry in {wait}s ({str(exc)[:50]})")
            time.sleep(wait)


def run_config(name: str, use_reranker: bool, eval_set: list[dict], scorers: dict) -> dict:
    from ragas import SingleTurnSample
    rag_chain.USE_RERANKER = use_reranker  # runtime toggle (see retrieve_docs)
    print(f"\n=== {name}  (reranker={'ON' if use_reranker else 'OFF'}) ===")
    per_question = []
    for i, item in enumerate(eval_set, 1):
        t0 = time.perf_counter()
        result = _generate_with_retry(item["question"])
        sample = SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=[d.page_content for d in result["context"]],
            reference=item["reference"],
        )
        scores = _score_sample(sample, scorers)
        per_question.append({"question": item["question"], **scores})
        fmt = lambda k: f"{scores[k]:.2f}" if scores[k] is not None else " — "
        print(f"  [{i}/{len(eval_set)}] {time.perf_counter() - t0:5.1f}s  "
              f"F={fmt('faithfulness')} AR={fmt('answer_relevancy')} "
              f"CP={fmt('context_precision')} CR={fmt('context_recall')}  "
              f"{item['question'][:46]}")

    agg = {}
    for m in METRIC_ORDER:
        vals = [r[m] for r in per_question if r[m] is not None]
        agg[m] = round(float(np.mean(vals)), 4) if vals else 0.0
    return {"aggregate": agg, "per_question": per_question}


SAMPLES_PATH = rag_chain.VECTORSTORE_DIR.parent / "eval_samples.json"

# (name, use_reranker, use_query_rewrite) — an incremental A/B/C ablation.
CONFIGS = [
    ("Baseline (Hybrid)", False, False),
    ("+ Reranker", True, False),
    ("+ Query Rewrite", True, True),
]


def generate_samples(eval_set: list[dict]) -> dict:
    """Phase A: run the real RAG pipeline (retrieval + generation) for every
    question under each config and dump the samples. No LLM judging here, so
    no rate limits and no torch+judge segfault — the judging is a separate phase.
    """
    rag_chain.USE_CORRECTIVE_RAG = False  # never abstain during evaluation
    out: dict[str, list[dict]] = {}
    for cfg_name, use_rr, use_rw in CONFIGS:
        rag_chain.USE_RERANKER = use_rr
        rag_chain.USE_QUERY_REWRITE = use_rw
        print(f"\n=== Generating: {cfg_name}  (rerank={use_rr}, rewrite={use_rw}) ===")
        rows = []
        for i, item in enumerate(eval_set, 1):
            res = _generate_with_retry(item["question"])
            rows.append({
                "question": item["question"],
                "reference": item["reference"],
                "answer": res["answer"],
                "contexts": [d.page_content for d in res["context"]],
            })
            print(f"  [{i}/{len(eval_set)}] {item['question'][:55]}")
        out[cfg_name] = rows
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    full = "--full" in sys.argv

    if "--generate" in sys.argv:
        eval_set = EVAL_SET[:4] if quick else EVAL_SET
        samples = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "gen_model": GEN_LLM_LABEL,
                "reranker_model": RERANKER_MODEL,
                "fetch_k": RETRIEVAL_FETCH_K,
                "n_questions": len(eval_set),
            },
            "samples": generate_samples(eval_set),
        }
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved {len(eval_set)} questions x {len(CONFIGS)} configs → {SAMPLES_PATH}")
        return
    # Default to 6 questions: RAGAS is token-heavy and Groq free tier is
    # 6000 TPM, so a full 12-question A/B (~24 samples) is ~70 min. Six
    # questions × 2 configs is a representative, completable run (~30 min).
    eval_set = EVAL_SET[:4] if quick else (EVAL_SET if full else EVAL_SET[:6])

    if not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY not set — needed for the judge model.")

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness, LLMContextPrecisionWithReference,
        LLMContextRecall, ResponseRelevancy,
    )
    judge = _build_judge()
    embeddings = LangchainEmbeddingsWrapper(rag_chain._get_embeddings())
    scorers = {
        "faithfulness": Faithfulness(llm=judge),
        # strictness=1 → asks the LLM for n=1 completion (Groq only allows n=1)
        "answer_relevancy": ResponseRelevancy(llm=judge, embeddings=embeddings, strictness=1),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }

    baseline = run_config("Baseline — Hybrid (no rerank)", False, eval_set, scorers)
    reranked = run_config("With Cross-Encoder Rerank", True, eval_set, scorers)

    deltas = {
        m: round(reranked["aggregate"].get(m, 0.0) - baseline["aggregate"].get(m, 0.0), 4)
        for m in METRIC_ORDER
    }

    import ragas
    out = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "framework": f"ragas {ragas.__version__}",
            "judge_model": JUDGE_MODEL,
            "gen_model": GEN_LLM_LABEL,
            "reranker_model": RERANKER_MODEL,
            "fetch_k": RETRIEVAL_FETCH_K,
            "n_questions": len(eval_set),
        },
        "configs": {
            "Baseline (Hybrid, no rerank)": baseline["aggregate"],
            "With Cross-Encoder Rerank": reranked["aggregate"],
        },
        "deltas": deltas,
        "per_question": {
            "Baseline (Hybrid, no rerank)": baseline["per_question"],
            "With Cross-Encoder Rerank": reranked["per_question"],
        },
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"{'Metric':<22}{'Baseline':>12}{'Reranked':>12}{'Δ':>10}")
    print("-" * 72)
    for m in METRIC_ORDER:
        b = baseline["aggregate"].get(m, 0.0)
        r = reranked["aggregate"].get(m, 0.0)
        print(f"{m:<22}{b:>12.3f}{r:>12.3f}{deltas[m]:>+10.3f}")
    print("=" * 72)
    print(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
