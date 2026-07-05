from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector database from document chunks.
    """
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vector_db


def save_vector_store(vector_db):
    """
    Save the FAISS index locally.
    """
    vector_db.save_local("vector_store")


def load_vector_store(embedding_model):
    """
    Load the saved FAISS index.
    """
    vector_db = FAISS.load_local(
        "vector_store",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_db