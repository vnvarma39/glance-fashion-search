"""
Vector Store Module
====================
Thin wrapper around ChromaDB that manages persistent collections of
embeddings with cosine-similarity search.

Handles batching transparently (ChromaDB has a per-call limit of ~5 000
items) and exposes a simple query interface with optional metadata
filtering.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.api.models.Collection import Collection

from config import VECTOR_STORE_DIR

# ChromaDB recommended max batch size
_CHROMA_BATCH_LIMIT = 5000


class VectorStore:
    """Persistent vector store backed by ChromaDB.

    Attributes:
        client: ChromaDB PersistentClient instance.
    """

    def __init__(self, persist_dir: Path = VECTOR_STORE_DIR) -> None:
        """Initialise the ChromaDB persistent client.

        Args:
            persist_dir: Directory where ChromaDB stores its data on disk.
        """
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VectorStore] Initialising ChromaDB at {self.persist_dir} ...")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        print("[VectorStore] ChromaDB client ready.")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self, name: str, metadata: dict | None = None
    ) -> Collection:
        """Create (or retrieve) a collection with cosine distance.

        Args:
            name: Name of the collection.
            metadata: Optional extra metadata for the collection.

        Returns:
            The ChromaDB ``Collection`` object.
        """
        col_metadata = {"hnsw:space": "cosine"}
        if metadata:
            col_metadata.update(metadata)

        collection = self.client.get_or_create_collection(
            name=name,
            metadata=col_metadata,
        )
        print(
            f"[VectorStore] Collection '{name}' ready "
            f"(existing items: {collection.count()})."
        )
        return collection

    # ------------------------------------------------------------------
    # Insertion (with automatic batching)
    # ------------------------------------------------------------------

    def add_to_collection(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add vectors and metadata to a collection.

        Large payloads are automatically split into batches that respect
        ChromaDB's per-call limit.

        Args:
            collection_name: Target collection name.
            ids: Unique string IDs (one per vector).
            embeddings: List of embedding vectors.
            metadatas: List of metadata dicts (one per vector).
        """
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        total = len(ids)
        print(f"[VectorStore] Adding {total} items to '{collection_name}' ...")

        for start in range(0, total, _CHROMA_BATCH_LIMIT):
            end = min(start + _CHROMA_BATCH_LIMIT, total)
            try:
                collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
            except Exception as exc:
                print(
                    f"[VectorStore] Error adding batch [{start}:{end}]: {exc}"
                )

        print(
            f"[VectorStore] Collection '{collection_name}' now has "
            f"{collection.count()} items."
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_collection(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
    ) -> dict:
        """Query a collection by embedding similarity.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: The query embedding vector.
            n_results: Maximum number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            ChromaDB results dict with keys ``ids``, ``distances``,
            ``metadatas``, etc.
        """
        collection = self.client.get_collection(name=collection_name)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = collection.query(**query_kwargs)
        except Exception as exc:
            print(f"[VectorStore] Query error: {exc}")
            results = {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of items in a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Integer count of stored items.
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except Exception as exc:
            print(f"[VectorStore] Could not get count for '{collection_name}': {exc}")
            return 0
