import streamlit as st
from dotenv import load_dotenv
from src.pdf_loader import PDFLoader
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_client import LLMClient

load_dotenv()

st.set_page_config(
    page_title="Indian Gov Services Assistant",
    page_icon="🇮🇳",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    language = st.radio("Response language", ["English", "Hindi"])
    st.markdown("---")
    st.markdown("**Services covered:**")
    st.markdown("- 🛂 Passport\n- 🗳️ Voter ID\n- 📋 RTI\n- 💰 Income Tax\n- 🚗 Driving Licence\n- 🏥 Ayushman Bharat")
    st.markdown("---")
    st.caption("Answers sourced from official government PDFs.")

# ── Load pipeline (once, cached) ─────────────────────────
@st.cache_resource(show_spinner="Loading documents...")
def load_pipeline():
    embedder = Embedder()
    documents, embeddings = embedder.load_cached()

    if documents is None:
        loader = PDFLoader()
        documents = loader.load_all_pdfs()
        embeddings = embedder.embed_documents(documents)

    retriever = Retriever(documents, embeddings)
    llm = LLMClient()
    return embedder, retriever, llm

embedder, retriever, llm = load_pipeline()

# ── Main UI ───────────────────────────────────────────────
st.title("🇮🇳 Indian Government Services Assistant")
st.caption("Ask anything about Passport, Voter ID, RTI, Income Tax, Driving Licence, or Ayushman Bharat.")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Sources"):
                for doc_name, text, score in msg["sources"]:
                    st.markdown(f"**{doc_name}** — relevance: `{score:.2f}`")
                    st.caption(text[:200] + "...")

# user input
query = st.chat_input("Ask your question...")

if query:
    # show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            q_emb = embedder.embed_query(query)
            results = retriever.retrieve(q_emb)
            answer = llm.generate_answer(query, results, language)

        st.markdown(answer)

        if results:
            with st.expander("📚 Sources"):
                for doc_name, text, score in results:
                    st.markdown(f"**{doc_name}** — relevance: `{score:.2f}`")
                    st.caption(text[:200] + "...")

    # save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": results
    })