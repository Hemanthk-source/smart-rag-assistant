"""Core RAG logic for Smart RAG Assistant."""
import io
import os
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import anthropic

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You are a Smart RAG Assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "could not find it in the uploaded documents. Cite sources like [1], [2] "
    "matching the numbered context chunks."
)


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def load_text(filename: str, data: bytes) -> str:
    """Extract text from PDF, TXT or MD files."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks, preferring sentence ends."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class RAGStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.sources: set[str] = set()
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def add_document(self, filename: str, data: bytes) -> int:
        if filename in self.sources:
            return 0
        pieces = chunk_text(load_text(filename, data))
        if not pieces:
            return 0
        vecs = self.embedder.encode(pieces, normalize_embeddings=True)
        self.chunks += [Chunk(p, filename, i) for i, p in enumerate(pieces)]
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
        self.sources.add(filename)
        return len(pieces)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.embeddings is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, question: str, history: list[dict] | None = None, k: int = 4):
        hits = self.search(question, k)
        if not hits:
            return "Please upload at least one document first.", []
        context = "\n\n".join(
            f"[{n}] (source: {c.source})\n{c.text}" for n, (c, _) in enumerate(hits, 1)
        )
        messages = (history or [])[-6:] + [
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        resp = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text, hits
