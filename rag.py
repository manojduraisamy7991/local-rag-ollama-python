"""RAG logic: load -> chunk -> embed -> retrieve -> generate (all local via Ollama)."""
import json
from pathlib import Path

import numpy as np
import ollama

LLM_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"
DOCS_DIR = Path("docs")
INDEX_FILE = Path("index.json")
CHUNK_SIZE = 500   # characters per chunk
OVERLAP = 100      # shared characters between neighbouring chunks
TOP_K = 3          # chunks sent to the LLM


def load_docs():
    docs = []
    for f in sorted(DOCS_DIR.glob("*")):
        if f.suffix.lower() in {".txt", ".md"}:
            docs.append({"source": f.name, "text": f.read_text(encoding="utf-8", errors="ignore")})
    return docs


def chunk_text(text):
    text = " ".join(text.split())
    step = CHUNK_SIZE - OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i:i + CHUNK_SIZE].strip()]


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def build_index():
    chunks = []
    for doc in load_docs():
        for piece in chunk_text(doc["text"]):
            chunks.append({"source": doc["source"], "text": piece})

    if not chunks:
        raise SystemExit(f"No .txt or .md files found in {DOCS_DIR}/")

    vectors = []
    for i in range(0, len(chunks), 32):
        vectors += embed([c["text"] for c in chunks[i:i + 32]])

    INDEX_FILE.write_text(json.dumps({"chunks": chunks, "vectors": vectors}))
    return len(chunks)


def load_index():
    if not INDEX_FILE.exists():
        raise SystemExit("No index found. Run: python main.py index")
    data = json.loads(INDEX_FILE.read_text())
    vectors = np.array(data["vectors"], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return data["chunks"], vectors


def retrieve(question, chunks, vectors):
    q = np.array(embed([question])[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in top]


def answer(question, chunks, vectors):
    hits = retrieve(question, chunks, vectors)
    context = "\n\n".join(h["text"] for h in hits)

    messages = [
        {
            "role": "system",
            "content": "You answer questions using the provided context. Be short and direct.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer from the context above:",
        },
    ]

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0},
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)

    print("\n\nSources:", ", ".join(sorted({h["source"] for h in hits})))
    hits = retrieve(question, chunks, vectors)
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

    prompt = (
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)

    print("\n\nSources:", ", ".join(sorted({h["source"] for h in hits})))