"""RAG: find past failures that look like the current one.

An embedding turns text into a list of numbers, so that texts with a similar
meaning end up with similar numbers. The vector store keeps one vector per
past failure. To search, we embed the question and compare it with every
stored vector using cosine similarity. The best matches are what we give
to the LLM as extra context. That retrieval step is the "R" in RAG.
"""
import hashlib
import os
import re

import numpy as np


def openai_embedder():
    from langchain_openai import OpenAIEmbeddings

    model = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        check_embedding_ctx_length=False,  # our texts are short, no tokenizer download needed
    )
    return model.embed_documents


def local_embedder(dims: int = 512):
    """Offline demo embedder: counts words. It matches words, not meaning."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(dims)
            for word in re.findall(r"[a-z0-9]{3,}", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1
            vectors.append(vec)
        return vectors

    return embed


def get_embedder():
    return openai_embedder() if os.getenv("OPENAI_API_KEY") else local_embedder()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self, documents: list[dict], embed):
        self.documents = documents
        self.embed = embed
        self.vectors = _normalize(np.array(embed([d["text"] for d in documents])))

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = _normalize(np.array(self.embed([query])))[0]
        scores = self.vectors @ query_vec
        best = np.argsort(scores)[::-1][:k]
        return [
            {
                "score": round(float(scores[i]), 3),
                "title": self.documents[i]["title"],
                "root_cause": self.documents[i]["root_cause"],
                "fix": self.documents[i]["fix"],
            }
            for i in best
        ]
