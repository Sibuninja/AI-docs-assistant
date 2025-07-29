
# MCP RAG & GenAI Model

This is a Retrieval-Augmented Generation (RAG) project using Groq's LLM (like LLaMA3), LangChain, FAISS, and HuggingFace embeddings to answer questions based on custom PDFs.

## Working :
          ┌────────────┐
          │  PDF Docs  │
          └────┬───────┘
               ▼
     ┌──────────────────┐
     │ Split + Embed    │ ← HuggingFace Embeddings
     └────┬─────────────┘
          ▼
     ┌─────────────┐
     │ Vector Store│ ← FAISS
     └────┬────────┘
          ▼
User → [RAG Retriever] → [Top Chunks] → [LLM via MCP] → ✨ GenAI Answer

## 🚀 Features

- Load and chunk PDF documents
- Convert document chunks into vector embeddings
- Store in FAISS vector database
- Use Groq's LLaMA3 or Mixtral LLM for answering user queries based on document context
- Simple terminal interface

## 📁 Project Structure

```
MCP RAG and GenAi model/
├── data/
│   └── sample.pdf
├── app.py
├── rag_engine.py
├── .env
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/mcp-rag-genai.git
cd mcp-rag-genai
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Environment Variables**

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the App**

```bash
python app.py
```

## 🧠 Ask Questions

Once the app starts, you can ask questions based on the content of `sample.pdf`. Type `exit` to quit.

## 📦 Requirements

- Python 3.10+
- `langchain`
- `langchain_groq`
- `sentence-transformers`
- `faiss-cpu`
- `python-dotenv`
- `torch`

## 📌 Notes

- Ensure your `sample.pdf` is placed in the `data/` folder.
- You can change the LLM model in `rag_engine.py` (`llama3-8b-8192`, `mixtral-8x7b`, etc.).


## 📄 License

This project is open-source and available under the MIT License.
