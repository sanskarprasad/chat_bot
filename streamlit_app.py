import os
import json
import time
import hashlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# — Environment & constants —
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Set GEMINI_API_KEY in your .env")
    st.stop()

MODEL_NAME = "gemini-1.5-flash-latest"
LOG_FILE   = "chat_log.txt"
TIME_FMT   = "%Y-%m-%d %H:%M:%S"

# — Sidebar controls —
st.sidebar.title("RAG Chatbot Settings")
chunk_size    = st.sidebar.number_input("Chunk Size",    value=1000, min_value=100, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=100,  min_value=0,   step=50)

@st.cache_resource
def build_pipeline(_documents: list[Document], file_hash: str):
    """
    We include `file_hash` so that whenever the uploaded file changes,
    this function is re-run, rebuilding FAISS + embeddings + LLM.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_documents(_documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    store     = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    prompt = PromptTemplate(
        template=(
            "You are an AI assistant. Answer based ONLY on context.\n"
            "Context:\n{context}\nQuestion:\n{question}\nAnswer:"
        ),
        input_variables=["context", "question"]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: "\n\n".join(d.page_content for d in x["context"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    pipeline = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return pipeline

# — File upload & hashing —
st.title("RAG Chatbot")
upload = st.file_uploader("Upload KB (txt, pdf, json, csv)", type=["txt","pdf","json","csv"])
if upload:
    data = upload.read()
    # Compute hash of the bytes
    file_hash = hashlib.md5(data).hexdigest()
    # Write out a temp file with original extension
    ext       = upload.name.rsplit(".", 1)[-1].lower()
    tmp_path  = f"kb_temp.{ext}"
    with open(tmp_path, "wb") as f:
        f.write(data)
    file_path = tmp_path
else:
    # If no upload, fallback to existing default
    file_path = "innovatech_info.txt" if os.path.exists("innovatech_info.txt") else None
    file_hash = "default"  # constant for the default file

if not file_path:
    st.warning("Upload a file or add innovatech_info.txt")
    st.stop()

# — Load documents —
def load_documents(p: str) -> list[Document]:
    ext = p.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return PyPDFLoader(p).load()
    if ext == "json":
        data = json.load(open(p, encoding="utf-8"))
        docs = []
        if isinstance(data, list):
            for item in data:
                docs.append(Document(page_content=json.dumps(item, ensure_ascii=False)))
        else:
            docs.append(Document(page_content=json.dumps(data, ensure_ascii=False)))
        return docs
    if ext == "csv":
        df = pd.read_csv(p)
        docs = []
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            docs.append(Document(page_content=" | ".join(parts)))
        return docs
    return TextLoader(p, encoding="utf-8").load()

docs     = load_documents(file_path)
pipeline = build_pipeline(docs, file_hash)

# — Chat UI & logging —
if "history" not in st.session_state:
    st.session_state.history = []
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== Session start: {time.strftime(TIME_FMT)} ===\n")

def log_interaction(user_q: str, bot_a: str):
    t = time.strftime(TIME_FMT)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{t}] You: {user_q}\n[{t}] Bot: {bot_a}\n")

query = st.text_input("You:")
if st.button("Send") and query.strip():
    with st.spinner("Thinking..."):
        resp = pipeline.invoke(query)
        ans  = resp.get("answer", "I don't know.")
        st.session_state.history.append((query, ans))
        log_interaction(query, ans)

for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
