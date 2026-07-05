import fitz  # PyMuPDF
from langchain_core.documents import Document
import os


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Better extraction mode
        text = page.get_text("text")

        if text.strip():   # Ignore empty pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": pdf_path
                    }
                )
            )

    return documents


def load_multiple_pdfs(folder_path):

    all_docs = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            path = os.path.join(folder_path, file)

            all_docs.extend(load_pdf(path))

    return all_docs