from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.llm import load_llm
from utils.vector_db import load_vector_store
from utils.embeddings import load_embedding_model
from utils.retriever import get_retriever
from utils.prompts import prompt


embedding = load_embedding_model()

vector_db = load_vector_store(embedding)

retriever = get_retriever(vector_db)

llm = load_llm()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)