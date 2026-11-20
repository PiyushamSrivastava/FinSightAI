
import streamlit as st
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from finsightai.config import BASE_DIR, CHROMA_PATH
from finsightai.embeddings import get_embeddings
from finsightai.vectorstore import load_vector_store
from finsightai.ingestion import split_documents
from finsightai.retrieval import get_retriever
from finsightai.rag_chain import (
    get_llm,
    create_rag_chain,
    get_answer
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

RAW_FOLDER = BASE_DIR / "data" / "raw"
RAW_FOLDER.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CUSTOM CSS
# =====================================================


st.markdown("""
<style>

/* Main application */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F8FAFC !important;
    color: #172554 !important;
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
header[data-testid="stHeader"] {
    background: #F8FAFC !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #EEF2F7 !important;
    border-right: 1px solid #DCE3EC;
}

section[data-testid="stSidebar"] * {
    color: #172554 !important;
}

/* Main headings */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554 !important;
}

.subtitle {
    color: #64748B !important;
    font-size: 16px;
}

/* All buttons */
.stButton > button {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-weight: 600;
}

.stButton > button:hover {
    background: #DBEAFE !important;
    border-color: #2563EB !important;
    color: #1D4ED8 !important;
}

/* PDF uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #94A3B8 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
}

[data-testid="stFileUploader"] * {
    color: #172554 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px;
    padding: 12px;
}

/* Chat input */
[data-testid="stBottom"] {
    background: #F8FAFC !important;
}

[data-testid="stChatInput"] {
    background: #F8FAFC !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #CBD5E1 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px;
    padding: 15px;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px;
}

/* Footer */
.footer {
    color: #64748B !important;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)



# =====================================================
# 3. LOAD AI COMPONENTS
# =====================================================

@st.cache_resource
def load_rag_components():

    embeddings = get_embeddings()

    vector_store = load_vector_store(embeddings)

    document_count = vector_store._collection.count()

    if document_count == 0:
        return None

    retriever = get_retriever(vector_store)

    llm = get_llm()

    chain = create_rag_chain(llm)

    return vector_store, retriever, chain


# =====================================================
# 4. PDF SAVING FUNCTION
# =====================================================

def save_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    original_name = Path(uploaded_file.name).name

    file_path = RAW_FOLDER / original_name

    # Handle same-name files with different contents
    if file_path.exists():

        existing_bytes = file_path.read_bytes()

        if existing_bytes != file_bytes:

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()[:8]

            file_path = RAW_FOLDER / (
                f"{file_path.stem}_{file_hash}.pdf"
            )

    # Save only if it does not already exist
    if not file_path.exists():

        file_path.write_bytes(file_bytes)

    return file_path


# =====================================================
# 5. INDEX PDF FUNCTION
# =====================================================

def index_pdf(file_path, vector_store):

    loader = PyPDFLoader(str(file_path))

    documents = loader.load()

    if not documents:
        raise ValueError("No pages found in this PDF.")

    # Check whether PDF contains extractable text
    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            "This PDF has no extractable text. "
            "It may be a scanned PDF requiring OCR."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    new_documents = []
    new_ids = []

    # Generate stable IDs
    for doc in chunks:

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        unique_text = (
            f"{source}|{page}|{doc.page_content}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        new_documents.append(doc)
        new_ids.append(chunk_id)

    # Check which chunks already exist
    existing_data = vector_store._collection.get(
        ids=new_ids
    )

    existing_ids = set(existing_data["ids"])

    documents_to_add = []
    ids_to_add = []

    for doc, chunk_id in zip(new_documents, new_ids):

        if chunk_id not in existing_ids:

            documents_to_add.append(doc)
            ids_to_add.append(chunk_id)

    if not documents_to_add:

        return {
            "pages": len(documents),
            "chunks": len(chunks),
            "added": 0,
            "message": "This PDF is already indexed."
        }

    # Add only new chunks in batches
    batch_size = 32

    progress = st.progress(0)

    total = len(documents_to_add)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        vector_store.add_documents(
            documents=documents_to_add[start:end],
            ids=ids_to_add[start:end]
        )

        progress.progress(end / total)

    progress.empty()

    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "added": total,
        "message": "PDF indexed successfully."
    }


# =====================================================
# 6. GET INDEXED DOCUMENTS
# =====================================================

def get_indexed_documents():

    collection = load_vector_store(
        get_embeddings()
    )

    result = collection._collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result["metadatas"]:

        if not metadata:
            continue

        source = metadata.get("source", "Unknown")

        filename = Path(source).name

        if filename not in documents:
            documents[filename] = 0

        documents[filename] += 1

    return documents


# =====================================================
# 7. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 💹 FinSight AI")

    st.caption("Financial Intelligence Assistant")

    st.divider()

    # PDF upload section
    st.subheader("📄 Add Documents")

    uploaded_files = st.file_uploader(
        "Upload financial PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to your knowledge base."
    )

    if st.button(
        "⬆️ Upload and Index PDFs",
        use_container_width=True,
        disabled=not uploaded_files
    ):

        try:

            embeddings = get_embeddings()

            vector_store = load_vector_store(embeddings)

            success_count = 0

            for uploaded_file in uploaded_files:

                st.write(
                    f"Processing: {uploaded_file.name}"
                )

                try:

                    file_path = save_pdf(uploaded_file)

                    result = index_pdf(
                        file_path,
                        vector_store
                    )

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{result['message']} "
                        f"({result['added']} new chunks)"
                    )

                    success_count += 1

                except Exception as error:

                    st.error(
                        f"{uploaded_file.name}: {error}"
                    )

            if success_count > 0:

                load_rag_components.clear()

                st.success("Document processing finished!")

                st.rerun()

        except Exception as error:

            st.error(f"Indexing failed: {error}")

    st.divider()

    # About section
    st.subheader("About")

    st.write(
        """
        FinSight AI is an AI-powered financial document
        assistant that uses Retrieval Augmented Generation
        to answer questions from your PDFs.
        """
    )

    st.divider()

    # Technology section
    st.subheader("Technology Stack")

    st.markdown("""
    🧠 **LLM:** Groq

    🔎 **Embeddings:** Jina AI

    🗄️ **Vector Database:** ChromaDB

    🔗 **Framework:** LangChain

    🖥️ **Frontend:** Streamlit
    """)

    st.divider()

    # Chat controls
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# =====================================================
# 9. MAIN HEADER
# =====================================================

st.markdown(
    '<div class="main-title">FinSight AI 💹</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent financial document assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 10. INITIALIZE BACKEND
# =====================================================

try:

    with st.spinner("Initializing FinSight AI..."):

        components = load_rag_components()

    if components is None:

        st.warning(
            "Your knowledge base is empty. "
            "Upload a financial PDF using the sidebar to get started."
        )

        st.stop()

    vector_store, retriever, chain = components

except Exception as error:

    st.error(f"Backend initialization failed: {error}")

    st.info(
        "Check your API keys, internet connection "
        "and ChromaDB configuration."
    )

    st.stop()


# =====================================================
# 11. DASHBOARD METRICS
# =====================================================

total_chunks = vector_store._collection.count()

indexed_documents = get_indexed_documents()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Indexed Documents",
        len(indexed_documents)
    )

with col2:
    st.metric(
        "Total Chunks",
        total_chunks
    )

with col3:
    st.metric(
        "AI Model",
        "Groq"
    )

st.divider()


# =====================================================
# 12. DOCUMENT LIBRARY
# =====================================================

with st.expander("📚 View Knowledge Base"):

    if indexed_documents:

        for filename, count in sorted(indexed_documents.items()):

            st.markdown(
                f"📄 **{filename}** — {count} chunks"
            )

    else:

        st.write("No documents indexed.")


# =====================================================
# 13. WELCOME SCREEN
# =====================================================

if not st.session_state.messages:

    st.markdown("### Welcome! 👋")

    st.write(
        """
        Ask questions about your financial documents.
        FinSight AI retrieves relevant information and
        generates answers with source references.
        """
    )

    st.markdown("#### Try asking")

    suggestions = [
        "What is financial planning?",
        "Explain the importance of insurance.",
        "What are the different investment options?",
        "What does the document say about savings accounts?"
    ]

    col1, col2 = st.columns(2)

    for i, suggestion in enumerate(suggestions):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{i}"
            ):

                st.session_state.pending_question = suggestion

                st.rerun()


# =====================================================
# 14. DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 View Sources"):

                    for source in sources:

                        st.markdown(
                            f"**{source['file']} — "
                            f"Page {source['page']}**"
                        )

                        st.caption(source["preview"])

                        st.divider()


# =====================================================
# 15. USER INPUT
# =====================================================

question = st.chat_input(
    "Ask anything about your financial documents..."
)

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None


# =====================================================
# 16. GENERATE ANSWER
# =====================================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your financial documents..."):

            try:

                answer, documents = get_answer(
                    question,
                    retriever,
                    chain
                )

                # Remove duplicate source-page entries
                unique_sources = {}

                for doc in documents:

                    source = Path(
                        doc.metadata.get("source", "Unknown")
                    ).name

                    page = doc.metadata.get("page", "Unknown")

                    if isinstance(page, int):
                        page += 1

                    key = (source, page)

                    if key not in unique_sources:

                        unique_sources[key] = {
                            "file": source,
                            "page": page,
                            "preview": doc.page_content[:300]
                        }

                sources = list(unique_sources.values())

                st.markdown(answer)

                if sources:

                    with st.expander("📚 View Sources"):

                        for source in sources:

                            st.markdown(
                                f"**{source['file']} — "
                                f"Page {source['page']}**"
                            )

                            st.caption(source["preview"])

                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(f"Unable to generate answer: {error}")


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94A3B8;
                font-size:13px; padding:15px;">
        FinSight AI | Built with LangChain, Groq,
        Jina AI and ChromaDB
        <br>
        AI-generated answers should be verified
        against the original financial documents.
    </div>
    """,
    unsafe_allow_html=True
)
