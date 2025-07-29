# app.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from rag_engine import get_rag_chain

if __name__ == "__main__":
    rag_chain = get_rag_chain()

    print("🤖 Ask your questions (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag_chain.invoke(query)
        print("Bot:", answer)
