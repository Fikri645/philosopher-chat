import json
import re
import time
from pathlib import Path

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from rag_chain import (
    retrieve_docs, retrieve_corrective, stream_llm, query, add_to_kb,
    vectorstore_exists, get_all_philosophers, get_kb_stats, get_umap_data,
    ABSTAIN_MESSAGE,
)
from config import (
    LLM_OPTIONS, DEFAULT_LLM, EMBEDDING_OPTIONS, DEFAULT_EMBEDDING,
    USE_RERANKER, RERANKER_MODEL,
)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_PROVIDER_COLOR = {
    "Google": "#4285F4",
    "Groq":   "#FF4B36",
    "OpenRouter": "#6366F1",
}

_COMPARE_DEFAULT_B = "Llama 4 Scout 17B  [Groq]"

_THINK_STYLE = (
    "color:var(--body-text-color-subdued);font-size:0.88em;"
    "border-left:3px solid var(--border-color-primary);padding-left:12px;margin:6px 0"
)
_SUMMARY_STYLE = (
    "cursor:pointer;color:var(--body-text-color-subdued);"
    "font-style:italic;user-select:none"
)


def _format_think_blocks(text: str) -> str:
    """Render <think>…</think> as collapsible, muted sections.

    Mid-stream (</think> not yet seen): open <details> showing live reasoning.
    Complete block: closed <details> with 'click to expand' label.
    """
    if "<think>" not in text:
        return text

    if "</think>" not in text:
        # Partial — think block still streaming
        idx = text.index("<think>")
        pre, thinking = text[:idx], text[idx + 7:]
        return (
            pre
            + f'<details open><summary style="{_SUMMARY_STYLE}">🤔 Thinking…</summary>'
            + f'<div style="{_THINK_STYLE}">{thinking}</div></details>'
        )

    def _wrap(m: re.Match) -> str:
        content = m.group(1).strip()
        return (
            f'<details><summary style="{_SUMMARY_STYLE}">'
            "🤔 Chain of thought (click to expand)</summary>"
            f'<div style="{_THINK_STYLE}">{content}</div></details>\n\n'
        )

    return re.sub(r"<think>(.*?)</think>", _wrap, text, flags=re.DOTALL)


def _score_bar(score: float, width: int = 10) -> str:
    filled = max(0, min(width, round(score * width)))
    return "█" * filled + "░" * (width - filled)


def _format_sources(docs: list, scores: list[float]) -> str:
    if not docs:
        return ""
    seen: set = set()
    lines: list[str] = []
    for doc, score in zip(docs, scores):
        key = doc.metadata.get("source", "Unknown source")
        if key not in seen:
            seen.add(key)
            tag = f"`{score:.2f}` " if score >= 0 else "`BM25` "
            lines.append(f"- {tag}{key}")
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def _format_retrieved_chunks(docs: list, scores: list[float]) -> str:
    if not docs:
        return "_No chunks retrieved._"

    pos_scores = [s for s in scores if s >= 0]
    avg = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    has_bm25 = any(s < 0 for s in scores)
    if USE_RERANKER:
        method = "Hybrid (RRF) → Cross-Encoder Rerank"
        score_label = "avg relevance"
    elif has_bm25:
        method = "Hybrid BM25 + Semantic"
        score_label = "avg similarity"
    else:
        method = "Semantic"
        score_label = "avg similarity"

    lines = [
        f"**{len(docs)} chunks** &nbsp;·&nbsp; {method}"
        f" &nbsp;·&nbsp; {score_label}: **{avg:.3f}**\n"
    ]
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        phil  = doc.metadata.get("philosopher", "?")
        title = doc.metadata.get("title", "?")
        if score >= 0:
            tag = f"`{score:.3f}` {_score_bar(score)}"
        else:
            tag = "`BM25 ` ──────────"
        text = doc.page_content[:200].replace("\n", " ").strip()
        lines.append(
            f"**{i}.** {tag} &nbsp; *{phil}* · {title}  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;*\"{text}...\"*\n"
        )
    return "\n".join(lines)


def _format_metrics(
    retrieve_s: float, llm_s: float, n_docs: int, n_sources: int
) -> str:
    return (
        f"⏱ &nbsp;Retrieval **{retrieve_s:.2f}s** &nbsp;·&nbsp; "
        f"LLM **{llm_s:.2f}s** &nbsp;·&nbsp; "
        f"Total **{retrieve_s + llm_s:.2f}s** &nbsp;·&nbsp; "
        f"**{n_docs}** chunks from **{n_sources}** source(s)"
    )


def _kb_markdown() -> str:
    stats = get_kb_stats()
    if not stats["total"]:
        return "_Knowledge base is empty._"
    lines = []
    for phil in sorted(stats["sources"]):
        lines.append(f"**{phil}**")
        for title in sorted(stats["sources"][phil]):
            lines.append(f"&nbsp;&nbsp;- *{title}*")
    lines.append(f"\n_{stats['total']:,} total chunks_")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def respond_stream(message: str, history: list, philosopher: str, llm_label: str):
    if not message.strip():
        yield history, "", gr.update(), gr.update()
        return

    if not vectorstore_exists():
        err = "⏳ Knowledge base is still being built on first run (~10 min). Please wait and try again."
        yield history + [{"role": "assistant", "content": err}], "", gr.update(), gr.update()
        return

    # — Build retrieval query —
    # Always enrich with the previous user message when there is history.
    # Follow-up questions ("bahas lebih lanjut", "how does this relate…") contain
    # pronouns / references that are meaningless without context; even standalone
    # questions benefit from the extra topic signal staying in the same thread.
    retrieval_query = message
    if history:
        last_user = next(
            (t["content"] for t in reversed(history) if t["role"] == "user"), ""
        )
        if last_user:
            retrieval_query = f"{last_user} {message}"

    # — Retrieval (fast, happens before streaming) —
    t0 = time.perf_counter()
    docs, scores, confidence = retrieve_corrective(retrieval_query, philosopher)
    retrieve_time = time.perf_counter() - t0
    context_str = "\n\n".join(d.page_content for d in docs)

    chunks_md = _format_retrieved_chunks(docs, scores)

    history = history + [
        {"role": "user", "content": message},
        {
            "role": "assistant",
            "content": (
                "<em style='color:var(--body-text-color-subdued)'>"
                "⏳ Retrieving context and generating response…"
                "</em>"
            ),
        },
    ]
    # Show user bubble + loading message immediately
    yield history, "", gr.update(value=chunks_md), gr.update()

    # Corrective RAG — abstain when retrieval confidence is too low (off-corpus
    # question) instead of hallucinating from weak context.
    if confidence == "low":
        history[-1]["content"] = ABSTAIN_MESSAGE
        metrics_md = _format_metrics(retrieve_time, 0.0, len(docs), 0)
        yield history, "", gr.update(value=chunks_md), gr.update(value=metrics_md)
        return

    provider, model_id = LLM_OPTIONS.get(llm_label, LLM_OPTIONS[DEFAULT_LLM])
    t1 = time.perf_counter()
    full_response = ""
    try:
        for text_chunk in stream_llm(provider, model_id, context_str, message, history=history[:-2]):
            full_response += text_chunk
            history[-1]["content"] = _format_think_blocks(full_response)
            yield history, "", gr.update(value=chunks_md), gr.update()

        llm_time = time.perf_counter() - t1
        unique_sources = len({d.metadata.get("source") for d in docs})
        metrics_md = _format_metrics(retrieve_time, llm_time, len(docs), unique_sources)

        history[-1]["content"] = (
            _format_think_blocks(full_response) + _format_sources(docs, scores)
        )
        yield history, "", gr.update(value=chunks_md), gr.update(value=metrics_md)

    except Exception as exc:
        history[-1]["content"] = f"⚠️ **Error:** {exc}"
        yield history, "", gr.update(value=chunks_md), gr.update()


def compare_respond(message: str, philosopher: str, llm_a: str, llm_b: str):
    if not message.strip():
        return "Enter a question above.", "", "Enter a question above.", ""
    if not vectorstore_exists():
        msg = "⏳ Knowledge base is still being built on first run (~10 min). Please wait and try again."
        return msg, "", msg, ""

    def _run(llm_label: str) -> tuple[str, str]:
        t0 = time.perf_counter()
        result = query(message, philosopher, llm_label)
        elapsed = time.perf_counter() - t0
        n_src = len({d.metadata.get("source") for d in result["context"]})
        sem_scores = [s for s in result["scores"] if s >= 0]
        avg = sum(sem_scores) / len(sem_scores) if sem_scores else 0.0
        metrics = (
            f"⏱ **{elapsed:.2f}s** &nbsp;·&nbsp; "
            f"**{len(result['context'])}** chunks from **{n_src}** source(s)"
            f" &nbsp;·&nbsp; avg similarity **{avg:.3f}**"
        )
        return result["answer"], metrics

    ans_a, met_a = _run(llm_a)
    ans_b, met_b = _run(llm_b)
    return ans_a, met_a, ans_b, met_b


def upload_source(file, author: str, title: str):
    if file is None:
        return gr.update(value="Please upload a file first."), gr.update()
    if not author.strip() or not title.strip():
        return gr.update(value="Please fill in both Author and Title."), gr.update()
    try:
        n = add_to_kb(file, author.strip(), title.strip())
        msg = f"Added {n:,} chunks from *{title}* by {author}."
    except Exception as e:
        msg = f"Error: {e}"
    return (
        gr.update(value=msg),
        gr.update(choices=get_all_philosophers(), value="All"),
    )


def refresh_kb():
    return gr.update(value=_kb_markdown())


def build_umap_plot():
    data = get_umap_data()
    if data is None:
        return None
    df = pd.DataFrame(data)
    fig = px.scatter(
        df, x="x", y="y",
        color="philosopher",
        hover_data={"title": True, "preview": True, "x": False, "y": False},
        title="Knowledge Base — Semantic Embedding Space (UMAP 2D)",
        labels={"x": "UMAP-1", "y": "UMAP-2"},
        opacity=0.75,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=540,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14),
        font=dict(color="rgba(220,220,220,0.9)"),
        legend=dict(
            title_text="",
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(20,20,20,0.5)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        margin=dict(l=40, r=20, t=48, b=36),
    )
    return fig


# ---------------------------------------------------------------------------
# RAGAS evaluation results
# ---------------------------------------------------------------------------

_EVAL_PATH = Path(__file__).parent / "eval_results.json"
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
_METRIC_DESC = {
    "faithfulness": "Share of answer claims supported by retrieved context (anti-hallucination)",
    "answer_relevancy": "How directly the answer addresses the question",
    "context_precision": "Are the relevant chunks ranked near the top?",
    "context_recall": "Share of the reference answer covered by retrieved context",
}


def _load_eval() -> dict | None:
    if not _EVAL_PATH.exists():
        return None
    try:
        return json.loads(_EVAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_eval_table() -> str:
    data = _load_eval()
    if data is None:
        return (
            "_No evaluation results yet. Run_ `python evaluate.py` _to generate "
            "`eval_results.json` (RAGAS metrics, ~12 min)._"
        )
    cfgs = list(data["configs"].keys())
    first, last = cfgs[0], cfgs[-1]
    meta = data.get("metadata", {})

    header = "| Metric | " + " | ".join(cfgs) + " | Δ (full) |"
    sep = "|---|" + "|".join([":---:"] * (len(cfgs) + 1)) + "|"
    lines = [
        f"**Evaluated with `{meta.get('framework', 'ragas')}`** &nbsp;·&nbsp; "
        f"{meta.get('n_questions', '?')} questions "
        f"&nbsp;·&nbsp; judge: `{meta.get('judge_model', '?')}`\n",
        header, sep,
    ]
    for m in _METRIC_LABELS:
        vals = " | ".join(f"{data['configs'][c].get(m, 0.0):.3f}" for c in cfgs)
        d = data.get("deltas", {}).get(m, data["configs"][last].get(m, 0.0) - data["configs"][first].get(m, 0.0))
        arrow = "🟢" if d > 0.005 else ("🔴" if d < -0.005 else "⚪")
        lines.append(
            f"| **{_METRIC_LABELS[m]}**<br><sub>{_METRIC_DESC[m]}</sub> "
            f"| {vals} | {arrow} {d:+.3f} |"
        )
    lines.append(
        f"\n_Δ = full pipeline vs. baseline. Generated {meta.get('generated_at', '?')} "
        "· metrics follow [RAGAS](https://docs.ragas.io) definitions (LLM-as-judge)._"
    )
    return "\n".join(lines)


def build_eval_chart():
    data = _load_eval()
    if data is None:
        return None
    cfgs = list(data["configs"].keys())
    metrics = list(_METRIC_LABELS.keys())
    labels = [_METRIC_LABELS[m] for m in metrics]
    colors = ["#6366F1", "#22C55E", "#F59E0B", "#EF4444"]

    fig = go.Figure()
    for i, cfg in enumerate(cfgs):
        fig.add_bar(
            name=cfg,
            x=labels,
            y=[data["configs"][cfg].get(m, 0.0) for m in metrics],
            marker_color=colors[i % len(colors)],
            text=[f"{data['configs'][cfg].get(m, 0.0):.2f}" for m in metrics],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        title="Retrieval Quality across pipeline stages",
        title_font=dict(size=14),
        height=460,
        yaxis=dict(range=[0, 1.05], title="score", gridcolor="rgba(255,255,255,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(220,220,220,0.9)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=40),
    )
    return fig


def refresh_eval():
    return gr.update(value=build_eval_table()), build_eval_chart()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What is Nietzsche's view on nihilism and the death of God?",
    "How does Schopenhauer view suffering and the will to live?",
    "What does Hume say about causality and the limits of reason?",
    "Can we have certain knowledge of the external world?",
    "Is morality objective or invented?",
    "Explain the concept of Eternal Return",
    "How does Marcus Aurelius advise dealing with suffering?",
    "What is Plato's ideal society in The Republic?",
    "Compare Schopenhauer and Nietzsche on the will",
    "What is Kant's categorical imperative?",
    "How does Mill justify utilitarianism?",
    "What does Epictetus say about what is in our control?",
]

CSS = """
footer { display: none !important; }
.section-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--body-text-color-subdued);
    margin-bottom: 2px;
}
.metric-bar p { font-size: 0.82rem; color: var(--body-text-color-subdued); margin: 4px 0; }
.status-box textarea { font-size: 0.82rem !important; }

/* Fix double scrollbar: prevent inner message wrappers from scrolling */
.chatbot .overflow-y-auto { scrollbar-width: thin; scrollbar-color: var(--border-color-primary) transparent; }
.chatbot .message-wrap { overflow: visible !important; }
.chatbot .message-wrap > div { overflow: visible !important; max-height: none !important; }
/* Prevent long markdown lines from adding a horizontal inner scroll */
.chatbot .prose { overflow-x: hidden !important; overflow-wrap: break-word; word-break: break-word; }
"""

with gr.Blocks(title="Philosopher Chat") as demo:

    gr.Markdown(
        """
# 📚 Philosopher Chat
**RAG chatbot grounded in Western philosophical primary texts**

Hybrid retrieval + cross-encoder reranking + corrective abstention &nbsp;·&nbsp; Streaming
&nbsp;·&nbsp; Multi-provider routing &nbsp;·&nbsp; RAGAS-evaluated &nbsp;·&nbsp; 12 primary texts · ~5 700 chunks
        """
    )

    with gr.Tabs():

        # ── Tab 1 ─ Chat ─────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=False):

                # Left: chat area
                with gr.Column(scale=3):
                    chatbot_ui = gr.Chatbot(
                        height=480,
                        show_label=False,
                        placeholder="*Ask a philosophical question to get started...*",
                    )
                    msg_input = gr.Textbox(
                        placeholder="Ask a philosophical question…",
                        show_label=False,
                        autofocus=True,
                        submit_btn=True,
                    )
                    metrics_display = gr.Markdown(
                        value="", elem_classes="metric-bar"
                    )
                    with gr.Accordion("📄 Retrieved Chunks & Scores", open=False):
                        retrieved_display = gr.Markdown(
                            value="_Submit a question to see retrieved context._"
                        )
                    with gr.Accordion("💡 Example Questions", open=False):
                        gr.Examples(
                            examples=[[q] for q in EXAMPLE_QUESTIONS],
                            inputs=[msg_input],
                            label=None,
                        )

                # Right: settings sidebar
                with gr.Column(scale=1, min_width=240):
                    with gr.Group():
                        gr.Markdown("**⚙️ Chat Settings**", elem_classes="section-label")
                        llm_dropdown = gr.Dropdown(
                            choices=list(LLM_OPTIONS.keys()),
                            value=DEFAULT_LLM,
                            label="LLM Model",
                        )
                        embedding_display = gr.Dropdown(
                            choices=list(EMBEDDING_OPTIONS.keys()),
                            value=DEFAULT_EMBEDDING,
                            label="Embedding Model",
                            info="Change requires rebuilding index (ingest.py)",
                            interactive=False,
                        )
                        philosopher_filter = gr.Dropdown(
                            choices=get_all_philosophers(),
                            value="All",
                            label="Filter by Philosopher",
                        )

                    with gr.Group():
                        gr.Markdown("**ℹ️ Stack**", elem_classes="section-label")
                        gr.Markdown(
                            "- Retrieval: **Hybrid (RRF) + Rerank**\n"
                            "- Reranker: **BGE-reranker-v2-m3**\n"
                            "- Guard: **Corrective RAG / abstention**\n"
                            "- Embeddings: **EmbeddingGemma-300M**\n"
                            "- Vector DB: **ChromaDB**\n"
                            "- Framework: **LangChain LCEL**\n"
                            "- UI: **Gradio 6**"
                        )

        # ── Tab 2 ─ Compare Models ───────────────────────────────────────
        with gr.Tab("⚖️ Compare Models"):
            gr.Markdown(
                "Run the same question through two models and compare quality, "
                "latency, and retrieval coverage side by side."
            )
            with gr.Row():
                compare_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a philosophical question…",
                    scale=4,
                )
                compare_philosopher = gr.Dropdown(
                    choices=get_all_philosophers(),
                    value="All",
                    label="Philosopher Filter",
                    scale=1,
                )
            compare_btn = gr.Button("▶ Compare", variant="primary")

            with gr.Row():
                with gr.Column():
                    model_a = gr.Dropdown(
                        choices=list(LLM_OPTIONS.keys()),
                        value=DEFAULT_LLM,
                        label="Model A",
                    )
                    response_a = gr.Markdown(label="Response A")
                    metrics_a  = gr.Markdown(elem_classes="metric-bar")

                with gr.Column():
                    model_b = gr.Dropdown(
                        choices=list(LLM_OPTIONS.keys()),
                        value=_COMPARE_DEFAULT_B,
                        label="Model B",
                    )
                    response_b = gr.Markdown(label="Response B")
                    metrics_b  = gr.Markdown(elem_classes="metric-bar")

        # ── Tab 3 ─ Knowledge Base ───────────────────────────────────────
        with gr.Tab("🗺️ Knowledge Base"):
            with gr.Row(equal_height=False):

                # Left: UMAP visualization
                with gr.Column(scale=2):
                    gr.Markdown(
                        "**Semantic Embedding Space**  \n"
                        "Each point is one text chunk. Clusters indicate semantic similarity — "
                        "nearby chunks share philosophical themes regardless of source."
                    )
                    umap_plot = gr.Plot()
                    umap_btn  = gr.Button(
                        "Generate Embedding Visualization", variant="secondary"
                    )
                    gr.Markdown(
                        "_UMAP projects ~5,700 × 768-dim embeddings to 2D. "
                        "First run takes ~1–2 min on CPU._"
                    )

                # Right: stats + upload
                with gr.Column(scale=1, min_width=280):
                    with gr.Group():
                        with gr.Row():
                            gr.Markdown(
                                "**📚 Knowledge Base**", elem_classes="section-label"
                            )
                            refresh_kb_btn = gr.Button("↻", size="sm", min_width=32)
                        kb_display = gr.Markdown(_kb_markdown())

                    with gr.Group():
                        gr.Markdown(
                            "**📤 Add Source**", elem_classes="section-label"
                        )
                        file_upload = gr.File(
                            label="Upload PDF or TXT",
                            file_types=[".pdf", ".txt"],
                        )
                        with gr.Row():
                            author_input = gr.Textbox(label="Author", scale=1)
                            title_input  = gr.Textbox(label="Title",  scale=1)
                        upload_btn = gr.Button(
                            "Add to Knowledge Base", variant="secondary", size="sm"
                        )
                        upload_status = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            placeholder="Upload status will appear here…",
                            elem_classes="status-box",
                        )

        # ── Tab 4 ─ Evaluation ───────────────────────────────────────────
        with gr.Tab("📊 Evaluation"):
            gr.Markdown(
                "### Does reranking actually help?\n"
                "The retrieval pipeline is measured with four **RAGAS** metrics "
                "over a curated question set with reference answers — once with the "
                "cross-encoder reranker **off** (hybrid baseline) and once **on**. "
                "This quantifies the impact of each retrieval component instead of "
                "guessing. _(Computed offline by_ `evaluate.py`_; an LLM acts as judge.)_"
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    eval_table = gr.Markdown(build_eval_table())
                with gr.Column(scale=1):
                    eval_chart = gr.Plot(build_eval_chart())
            refresh_eval_btn = gr.Button("↻ Reload results", size="sm")
            gr.Markdown(
                "**Metric definitions** &nbsp;·&nbsp; "
                "**Faithfulness**: answer grounded in context (anti-hallucination) &nbsp;·&nbsp; "
                "**Answer Relevancy**: answer addresses the question &nbsp;·&nbsp; "
                "**Context Precision**: relevant chunks ranked high &nbsp;·&nbsp; "
                "**Context Recall**: reference answer covered by context."
            )

    # ── Event wiring ─────────────────────────────────────────────────────

    msg_input.submit(
        respond_stream,
        inputs=[msg_input, chatbot_ui, philosopher_filter, llm_dropdown],
        outputs=[chatbot_ui, msg_input, retrieved_display, metrics_display],
    )

    compare_btn.click(
        compare_respond,
        inputs=[compare_input, compare_philosopher, model_a, model_b],
        outputs=[response_a, metrics_a, response_b, metrics_b],
    )

    umap_btn.click(build_umap_plot, outputs=umap_plot)

    refresh_kb_btn.click(refresh_kb, outputs=kb_display)

    upload_btn.click(
        upload_source,
        inputs=[file_upload, author_input, title_input],
        outputs=[upload_status, philosopher_filter],
    ).then(refresh_kb, outputs=kb_display)

    refresh_eval_btn.click(refresh_eval, outputs=[eval_table, eval_chart])


def _auto_ingest() -> None:
    """Trigger background KB build on first Spaces run (non-blocking)."""
    if not vectorstore_exists():
        print("[startup] Vectorstore missing — starting background ingest (~10 min)…")
        import threading

        def _run() -> None:
            try:
                import ingest
                ingest.main()
                print("[startup] Ingest complete. Knowledge base is now ready.")
            except Exception as exc:
                print(f"[startup] Ingest failed: {exc}")

        threading.Thread(target=_run, daemon=True).start()


_auto_ingest()

if __name__ == "__main__":
    demo.launch(css=CSS)
