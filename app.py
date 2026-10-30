import streamlit as st
from rag import RAGStore

st.set_page_config(page_title="Smart RAG Assistant", page_icon="🧠")
st.title("🧠 Smart RAG Assistant")
st.caption("Upload documents, then ask questions grounded in their content.")


@st.cache_resource
def get_store() -> RAGStore:
    return RAGStore()


store = get_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: document upload
with st.sidebar:
    st.header("📄 Documents")
    files = st.file_uploader(
        "Upload PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if files:
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n = store.add_document(f.name, f.read())
            if n:
                st.success(f"{f.name}: {n} chunks indexed")
    st.write(f"**Indexed files:** {len(store.sources)}")
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, hits = store.answer(question, st.session_state.messages, k=top_k)
        st.markdown(answer)
        if hits:
            with st.expander("Sources"):
                for n, (chunk, score) in enumerate(hits, 1):
                    st.markdown(f"**[{n}] {chunk.source}** (similarity {score:.2f})")
                    st.caption(chunk.text[:400] + "...")
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
