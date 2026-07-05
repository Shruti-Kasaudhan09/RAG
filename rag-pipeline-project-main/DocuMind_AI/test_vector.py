from utils.pdf_loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import load_embedding_model
from utils.vector_db import create_vector_store, save_vector_store

print("=" * 60)
print("STEP 1 : Loading PDF")
print("=" * 60)

documents = load_pdf("data/sample.pdf")

print("Pages :", len(documents))


print("\n" + "=" * 60)
print("STEP 2 : Splitting")
print("=" * 60)

chunks = split_documents(documents)

print("Chunks :", len(chunks))


print("\n" + "=" * 60)
print("STEP 3 : Embedding Model")
print("=" * 60)

embedding = load_embedding_model()

print("Embedding Loaded")


print("\n" + "=" * 60)
print("STEP 4 : Creating Vector Database")
print("=" * 60)

vector_db = create_vector_store(chunks, embedding)

print("Vector DB Created")


save_vector_store(vector_db)

print("Vector DB Saved Successfully")


print("\nProject Working Successfully")