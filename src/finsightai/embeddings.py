import requests

from langchain_core.embeddings import Embeddings
from finsightai.config import JINA_API_KEY, EMBEDDING_MODEL


class JinaEmbeddings(Embeddings):

    def __init__(self):
        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY is missing in .env")

        self.api_key = JINA_API_KEY
        self.model = EMBEDDING_MODEL
        self.batch_size = 32

    def _embed(self, texts, task):

        url = "https://api.jina.ai/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": task
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()["data"]

        # Maintain the original input order
        data.sort(key=lambda item: item["index"])

        return [item["embedding"] for item in data]

    def embed_documents(self, texts):

        all_embeddings = []

        # Send documents in smaller batches
        for start in range(0, len(texts), self.batch_size):

            batch = texts[start:start + self.batch_size]

            embeddings = self._embed(
                batch,
                task="retrieval.passage"
            )

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text):

        embeddings = self._embed(
            [text],
            task="retrieval.query"
        )

        return embeddings[0]


def get_embeddings():
    return JinaEmbeddings()