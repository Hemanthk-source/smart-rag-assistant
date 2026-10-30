# Smart RAG Assistant

A simple Retrieval-Augmented Generation app: upload PDF/TXT/MD files and chat with them.

## How it works
1. **Load** – extract text from uploaded files
2. **Chunk** – split into overlapping ~800-character pieces
3. **Embed** – `all-MiniLM-L6-v2` (sentence-transformers) turns chunks into vectors
4. **Retrieve** – cosine similarity finds the top-k chunks for a question
5. **Generate** – Claude answers using only those chunks, with citations

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"      # Windows: set ANTHROPIC_API_KEY=your-key
streamlit run app.py
```

Optional: `export CLAUDE_MODEL="claude-sonnet-5-5"` to change the model.

## Structure
- `rag.py` – loading, chunking, embeddings, retrieval, generation
- `app.py` – Streamlit chat UI

## Ideas to extend
- Swap the in-memory store for FAISS / ChromaDB with persistence
- Add DOCX/web-page loaders
- Add a reranker for better retrieval
- Stream responses token by token
