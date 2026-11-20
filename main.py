
import argparse
import hashlib
from pathlib import Path

from finsightai.ingestion import load_documents, split_documents
from finsightai.embeddings import get_embeddings
from finsightai.vectorstore import load_vector_store
from finsightai.retrieval import get_retriever
from finsightai.rag_chain import (
    get_llm,
    create_rag_chain,
    get_answer
)


def index_documents():

    print("Loading PDF documents...")

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)

    total_chunks = len(chunks)
    batch_size = 32

    print("Total chunks:", total_chunks)

    for start in range(0, total_chunks, batch_size):

        end = min(start + batch_size, total_chunks)
        batch = chunks[start:end]

        # Generate unique IDs using source, page and content
        ids = []

        for doc in batch:

            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")

            unique_text = f"{source}|{page}|{doc.page_content}"

            chunk_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        print(f"\nProcessing chunks {start + 1} to {end}")

        vector_store.add_documents(
            documents=batch,
            ids=ids
        )

        print(f"Processed {end}/{total_chunks} chunks")

    print("\nIndexing completed!")


def chat():

    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)

    if vector_store._collection.count() == 0:
        print("No documents indexed. Run with --index first.")
        return

    retriever = get_retriever(vector_store)

    llm = get_llm()
    chain = create_rag_chain(llm)

    print("\nFinSight AI is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer, documents = get_answer(
            question,
            retriever,
            chain
        )

        print("\nFinSight AI:", answer)
        print("\nSources:")

        unique_sources = set()

    for doc in documents:

            source = Path(
            doc.metadata.get("source", "Unknown")
        ).name

    page = doc.metadata.get("page", "Unknown")

    if isinstance(page, int):
        page += 1

    unique_sources.add((source, page))

for source, page in sorted(unique_sources):
    print(f"- {source}, Page {page}")

print("\n" + "-" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        action="store_true",
        help="Index PDF documents"
    )

    args = parser.parse_args()

    if args.index:
        index_documents()
    else:
        chat()
