from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def get_rag_chain():
    # Load PDF and split into documents
    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"✅ Loaded {len(docs)} chunks from data/sample.pdf")

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_store")
    print("✅ Vector store saved to 'vector_store'")

    # Set up local Hugging Face LLM
    qa_pipeline = pipeline(
        "text-generation",
        model="gpt2",  # You can replace this with a more powerful model if needed
        max_length=512,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Set up RAG chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return rag_chain
