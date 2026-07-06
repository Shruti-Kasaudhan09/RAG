import os
import streamlit as st

# ==============================================================================
# 1. PERMANENT THEME ENFORCEMENT CONFIGURATOR (AUTOMATIC DISK WRITER)
# ==============================================================================
# This block intercepts initialization and writes the system variables directly 
# to the disk. This permanently eliminates the white iFrame layout layers.
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")

theme_payload = """
[theme]
primaryColor = "#6366F1"
backgroundColor = "#0B1120"
secondaryBackgroundColor = "#111827"
textColor = "#F8FAFC"
font = "sans serif"
"""

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Check if file matches our exact palette targets; if not, overwrite and force refresh
if not os.path.exists(config_path) or open(config_path, "r").read().strip() != theme_payload.strip():
    with open(config_path, "w") as f:
        f.write(theme_payload.strip())
    st.rerun()

# ==============================================================================
# 2. CORE BACKEND IMPORTS
# ==============================================================================
from utils.pdf_loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import load_embedding_model
from utils.vector_db import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)
from utils.retriever import get_retriever
from utils.llm import load_llm
from utils.prompts import prompt

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HIGH-CONTRAST UI FINE-TUNING ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* Main System Font Canvas Wrap */
html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
}

/* Force high-contrast text rendering on custom elements */
h1, h2, h3, h4, h5, h6, p, span, li, label, strong, small {
    color: #F8FAFC !important;
}

/* Sidebar Structural Border Adjustment */
section[data-testid="stSidebar"] {
    border-right: 1px solid #334155 !important;
}

/* Custom Navigation Badges inside Sidebar */
.tech-tag {
    background-color: #0B1120 !important; 
    padding: 8px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 14px;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
}

/* File Uploader Container Box Enhancements */
div[data-testid="stFileUploader"] {
    border: 2px dashed #334155 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
div[data-testid="stFileUploader"] *, div[data-testid="stFileUploaderDropzone"] * {
    color: #CBD5E1 !important;
}

/* Chat Input Interactive Border Framing */
div[data-testid="stChatInput"] {
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] textarea {
    color: #F8FAFC !important;
}

/* Chat History Message Log Panels */
div[data-testid="stChatMessage"] {
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
}
div[data-testid="stChatMessage"] p, 
div[data-testid="stChatMessage"] span {
    color: #F8FAFC !important;
}
.assistant-marker {
    border-left: 4px solid #06B6D4; /* SECONDARY CYAN ACCENT */
    padding-left: 12px;
}

/* Main UI Action Pipeline Control Buttons */
.stButton button {
    background: linear-gradient(90deg, #6366F1, #2563EB) !important; /* PRIMARY Gradient */
    color: #F8FAFC !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 46px;
    width: 100%;
}
.stButton button:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5) !important;
}

/* Strip frame decorations */
#MainMenu, header, footer { visibility: hidden !important; }
hr { border-top: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE INIT ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ---------------- MAIN WINDOW HERO PANEL ---------------- #
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 40px; font-weight: 800; color: #F8FAFC; margin-bottom: 5px;">Welcome to <span style="color: #6366F1;">DocuMind AI</span></h1>
    <p style="color: #CBD5E1; font-size: 16px;">Upload a PDF and ask anything about your document.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR PANEL CONTAINER ---------------- #
with st.sidebar:
    st.markdown("# 🧠 DocuMind AI")
    st.write("Chat with your documents")
    st.markdown("---")
    
    st.markdown("<strong>Navigation</strong>", unsafe_allow_html=True)
    st.button("💬 Chat", key="nav_chat", use_container_width=True)
    st.button("📁 Uploads", key="nav_uploads", use_container_width=True)
    st.button("⚙️ Settings", key="nav_settings", use_container_width=True)
    st.markdown("---")
    
    st.markdown("<strong>Powered by</strong>", unsafe_allow_html=True)
    st.markdown("<div class='tech-tag'>🦜 LangChain</div>", unsafe_allow_html=True)
    st.markdown("<div class='tech-tag'>✨ Google Gemini</div>", unsafe_allow_html=True)
    st.markdown("<div class='tech-tag'>🗄️ FAISS</div>", unsafe_allow_html=True)
    st.markdown("<div class='tech-tag'>🤗 HuggingFace</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Version 1.0")


# ---------------- FILE INTAKE PIPELINE MODULE ---------------- #
st.markdown("### 📄 Upload PDF")
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None and not st.session_state.pdf_processed:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Process PDF & Create Database"):
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_file.name)
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📚 Processing PDF and generating embeddings..."):
            docs = load_pdf(pdf_path)
            chunks = split_documents(docs)
            embedding = load_embedding_model()
            vector_db = create_vector_store(chunks, embedding)
            save_vector_store(vector_db)
            
        st.session_state.pdf_processed = True
        st.success("✅ PDF Processed Successfully!")

if st.session_state.pdf_processed:
    st.info(f"📁 Active Document: {uploaded_file.name}")


st.markdown("---")
st.markdown("### 🗪 Chat with your PDF")


# ---------------- GRAPHICAL MESSAGES LAYER ---------------- #
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f'<div class="assistant-marker">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])


# ---------------- USER SUBMISSION BLOCK ---------------- #
if question := st.chat_input("Ask a question about your PDF..."):
    
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        embedding = load_embedding_model()
        vector_db = load_vector_store(embedding)
        retriever = get_retriever(vector_db)
        llm = load_llm()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = rag_chain.invoke(question)
                
                if uploaded_file:
                    answer += f"\n\n*📄 Source: {uploaded_file.name}*"
                
                st.markdown(f'<div class="assistant-marker">{answer}</div>', unsafe_allow_html=True)
                
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Please ensure your PDF is uploaded and processed before asking questions.")