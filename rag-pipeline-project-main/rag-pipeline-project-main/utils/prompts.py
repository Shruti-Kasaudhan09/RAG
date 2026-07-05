from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not found in the context, say:

"I could not find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
""")