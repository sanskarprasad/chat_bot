import os
import json
from dotenv import load_dotenv
import time
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
time_format = "%Y-%m-%d %H:%M:%S"
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Constants
MODEL_NAME = "gemini-1.5-flash-latest"
LOG_FILE = "chat_log.txt"

# Sidebar settings
st.sidebar.title("RAG Chatbot Settings")
chunk_size = st.sidebar.number_input("Chunk Size", value=1000, min_value=100, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=100, min_value=0, step=50)

@st.cache_resource
# Prefix documents with underscore to avoid hashing errors
def initialize_pipeline(_documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_documents(_documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    prompt = PromptTemplate(
        template=(
            "You are an AI assistant. Answer based ONLY on context."
            " If unknown, say so. Context:\n{context}\nQuestion:\n{question}\nAnswer:"),
        input_variables=["context", "question"]
    )

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: "\n\n".join(d.page_content for d in x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    pipeline = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return pipeline

# App title and file upload
st.title("Innovatech Solutions RAG Chatbot")
uploaded_file = st.file_uploader(
    "Upload your knowledge base", type=["txt", "pdf", "json"]
)

# Determine file path & loader type
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    local_path = f"kb.{file_ext}"
    with open(local_path, "wb") as f:
        f.write(uploaded_file.read())
    file_path = local_path
elif os.path.exists("innovatech_info.txt"):
    file_path = "innovatech_info.txt"
else:
    st.warning("Please upload a .txt, .pdf, or .json file, or add 'innovatech_info.txt' in the directory.")
    st.stop()

# Load documents based on extension
def load_docs(path: str) -> list[Document]:
    ext = path.split('.')[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(path)
        return loader.load()
    if ext == "json":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        docs = []
        if isinstance(data, list):
            for item in data:
                docs.append(Document(page_content=json.dumps(item, ensure_ascii=False)))
        else:
            docs.append(Document(page_content=json.dumps(data, ensure_ascii=False)))
        return docs
    # default to text
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()

docs = load_docs(file_path)
pipeline = initialize_pipeline(docs)

# Initialize chat history and logging
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add header to log file
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== Chat Session Started: {time.strftime(time_format)} ===\n")

# User input and response
def log_interaction(user_q: str, bot_a: str):
    timestamp = time.strftime(time_format)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] You: {user_q}\n")
        log.write(f"[{timestamp}] Bot: {bot_a}\n")

user_question = st.text_input("You:")
if st.button("Send") and user_question.strip():
    with st.spinner("Thinking..."):
        result = pipeline.invoke(user_question)
        answer = result.get("answer", "I don't know.")
        st.session_state.chat_history.append((user_question, answer))
        log_interaction(user_question, answer)

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")

