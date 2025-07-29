# rag_engine.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

# Load PDF and split into chunks
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)
    print(f"✅ Loaded {len(docs)} chunks from {pdf_path}")
    return docs

# Create FAISS vector store
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Build RAG pipeline using Groq
def get_rag_chain():
    docs = load_and_split_pdf("data/sample.pdf")
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

    # Load Groq Chat LLM
    llm = ChatGroq(
        model="llama3-8b-8192",  # or use mixtral/gemma
        temperature=0.3,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain
