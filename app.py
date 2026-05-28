import gradio as gr
from rag_chain import query, vectorstore_exists
from config import PHILOSOPHER_NAMES


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


def respond(message: str, history: list, philosopher: str) -> str:
    if not message.strip():
        return ""

    if not vectorstore_exists():
        return (
            "Knowledge base not found. "
            "Run `python ingest.py` and commit the `vectorstore/` directory."
        )

    result = query(message, philosopher)
    answer: str = result.get("answer", "No answer generated.")
    sources: str = _format_sources(result.get("context", []))
    return answer + sources


philosopher_filter = gr.Dropdown(
    choices=PHILOSOPHER_NAMES,
    value="All",
    label="Filter by Philosopher",
    info="Restrict retrieval to one philosopher's texts, or query all.",
)

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[philosopher_filter],
    title="Philosopher Chat",
    description=(
        "Ask questions grounded in Western philosophical primary texts: "
        "Nietzsche, Schopenhauer, Hume, and Russell. "
        "Every answer is retrieved directly from their original works."
    ),
    examples=[
        ["What is the meaning of life according to Nietzsche?", "Nietzsche"],
        ["How does Schopenhauer view human suffering and the will?", "Schopenhauer"],
        ["What does Hume say about causality and the limits of reason?", "Hume"],
        ["Can we have certain knowledge of the external world?", "Russell"],
        ["Is morality objective or invented?", "All"],
        ["Explain the concept of Eternal Return", "Nietzsche"],
        ["What is the relationship between will and representation?", "Schopenhauer"],
        ["How should one respond to the absence of inherent meaning?", "All"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
