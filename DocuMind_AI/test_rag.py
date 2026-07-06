from rag_chain import rag_chain

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\nAnswer:\n")

    print(answer)