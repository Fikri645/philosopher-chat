import gradio as gr
from rag_chain import (
    query, add_to_kb, vectorstore_exists,
    get_all_philosophers, get_kb_stats
)
from config import LLM_OPTIONS, DEFAULT_LLM, EMBEDDING_OPTIONS, DEFAULT_EMBEDDING

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_sources(context_docs: list) -> str:
    if not context_docs:
        return ""
    seen: set = set()
    lines: list = []
    for doc in context_docs:
        key = doc.metadata.get("source", "Unknown source")
        if key not in seen:
            seen.add(key)
            lines.append(f"- {key}")
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


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

def respond(message: str, history: list, philosopher: str, llm_label: str):
    if not message.strip():
        return history, ""

    if not vectorstore_exists():
        error = "Knowledge base not found. Run `python ingest.py` first."
        return history + [{"role": "assistant", "content": error}], ""

    result = query(message, philosopher, llm_label)
    answer = result.get("answer", "No answer generated.")
    sources = _format_sources(result.get("context", []))

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer + sources},
    ]
    return history, ""


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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.sidebar { background: var(--block-background-fill); }
.section-header { font-size: 0.85rem; font-weight: 600; color: var(--body-text-color-subdued);
                  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.status-box textarea { font-size: 0.82rem !important; color: var(--body-text-color-subdued) !important; }
footer { display: none !important; }
"""

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

with gr.Blocks(title="Philosopher Chat") as demo:

    # --- Header ---
    gr.Markdown(
        """
# 📚 Philosopher Chat
### A RAG chatbot grounded in Western philosophical primary texts
Ask questions about nihilism, epistemology, ethics, and existence — answers are retrieved
directly from the original works of Nietzsche, Schopenhauer, Hume, Russell, and more.
        """
    )

    with gr.Row(equal_height=False):

        # ── Left: Chat ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                height=520,
                show_label=False,
                placeholder="*Ask a philosophical question to get started...*",
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a philosophical question...",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                    submit_btn=True,
                )

            with gr.Accordion("Example Questions", open=False):
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[msg_input],
                    label=None,
                )

        # ── Right: Sidebar ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):

            # Chat settings
            with gr.Group():
                gr.Markdown("**⚙️ Chat Settings**", elem_classes="section-header")
                llm_dropdown = gr.Dropdown(
                    choices=list(LLM_OPTIONS.keys()),
                    value=DEFAULT_LLM,
                    label="LLM Model",
                )
                embedding_display = gr.Dropdown(
                    choices=list(EMBEDDING_OPTIONS.keys()),
                    value=DEFAULT_EMBEDDING,
                    label="Embedding Model",
                    info="Changing requires rebuilding the index with ingest.py",
                    interactive=False,
                )
                philosopher_filter = gr.Dropdown(
                    choices=get_all_philosophers(),
                    value="All",
                    label="Filter by Philosopher",
                )

            # Upload
            with gr.Group():
                gr.Markdown("**📤 Add Source**", elem_classes="section-header")
                file_upload = gr.File(
                    label="Upload PDF or TXT",
                    file_types=[".pdf", ".txt"],
                )
                with gr.Row():
                    author_input = gr.Textbox(label="Author / Philosopher", scale=1)
                    title_input  = gr.Textbox(label="Work Title", scale=1)
                upload_btn = gr.Button("Add to Knowledge Base", variant="secondary", size="sm")
                upload_status = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    placeholder="Upload status will appear here...",
                    elem_classes="status-box",
                )

            # Knowledge base overview
            with gr.Group():
                with gr.Row():
                    gr.Markdown("**📚 Knowledge Base**", elem_classes="section-header")
                    gr.Button("↻", size="sm", min_width=32).click(
                        refresh_kb, outputs=gr.Markdown()
                    )
                kb_display = gr.Markdown(_kb_markdown())

    # --- Wire events ---

    msg_input.submit(
        respond,
        inputs=[msg_input, chatbot_ui, philosopher_filter, llm_dropdown],
        outputs=[chatbot_ui, msg_input],
    )

    upload_btn.click(
        upload_source,
        inputs=[file_upload, author_input, title_input],
        outputs=[upload_status, philosopher_filter],
    ).then(refresh_kb, outputs=kb_display)


if __name__ == "__main__":
    demo.launch()
