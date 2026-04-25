import os
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


PDF_DIR = "pdfs"
DB_DIR = "faiss_db"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

st.set_page_config(page_title="PDF RAG with FAISS")
st.title("PDF RAG with FAISS")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2:1b")


def get_hash(data):
    return hashlib.md5(data).hexdigest()


def create_or_load_vectorstore(pdf):
    data = pdf.read()

    if not data:
        st.error("Uploaded PDF is empty.")
        return None

    file_id = get_hash(data)
    pdf_path = os.path.join(PDF_DIR, f"{file_id}.pdf")
    db_path = os.path.join(DB_DIR, file_id)

    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(data)

    embeddings = get_embeddings()

    if os.path.exists(db_path):
        st.success("Existing FAISS vector store loaded")
        st.info("RAG not regenerated")
        return FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    with st.spinner("Reading PDF..."):
        docs = PyPDFLoader(pdf_path).load()

    docs = [doc for doc in docs if doc.page_content.strip()]

    if not docs:
        st.error("No readable text found. PDF may be scanned/image-based.")
        return None

    with st.spinner("Splitting PDF into chunks..."):
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80
        ).split_documents(docs)

    if not chunks:
        st.error("Could not create chunks from PDF.")
        return None

    with st.spinner("Creating FAISS vector store..."):
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(db_path)

    st.success("Vector store Created")
    st.info(f"PDF segregated into {len(chunks)} chunks")

    return vectordb


def answer_question(vectordb, question):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    query = f"Explain: {question}"
    docs = retriever.invoke(query)

    if not docs:
        return "I could not find this in the PDF.", []

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are answering strictly from the given PDF context.

Rules:
- Use only the given context.
- If the answer is present, answer clearly.
- If the answer is not present, say exactly:
  I could not find this in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = get_llm().invoke(prompt)
    return answer, docs


pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    try:
        vectordb = create_or_load_vectorstore(pdf)

        if vectordb:
            question = st.text_input("Ask question from PDF")

            if question.strip():
                with st.spinner("Fetching answer from FAISS vector DB..."):
                    answer, source_docs = answer_question(vectordb, question)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Source chunks"):
                    for i, doc in enumerate(source_docs, 1):
                        st.write(f"Chunk {i}")
                        st.write(doc.page_content)

            else:
                st.info("Enter a question to search from the PDF.")

    except Exception as e:
        st.error("Something went wrong.")
        st.code(str(e))
else:
    st.warning("Please upload a PDF.")