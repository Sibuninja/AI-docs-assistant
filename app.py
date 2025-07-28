from rag_engine import get_rag_chain

rag_chain = get_rag_chain()

while True:
    query = input("\n🧠 Ask a question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = rag_chain.run(query)
    print("🤖", response)
