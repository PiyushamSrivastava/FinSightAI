from langchain_chroma import Chroma

from finsightai.config import CHROMA_PATH, COLLECTION_NAME


def load_vector_store(embeddings):

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    return vector_store


def create_vector_store(chunks, embeddings):

    vector_store = load_vector_store(embeddings)

    vector_store.add_documents(chunks)

    print("Documents stored in ChromaDB!")

    return vector_store