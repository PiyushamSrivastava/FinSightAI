from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents():

    documents = []

    pdf_folder = Path("data/raw")

    for pdf in pdf_folder.glob("*.pdf"):

        loader = PyPDFLoader(str(pdf))

        pages = loader.load()

        documents.extend(pages)

    print("Total pages loaded:", len(documents))

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("Total chunks created:", len(chunks))

    return chunks