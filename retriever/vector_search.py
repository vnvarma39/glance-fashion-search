"""
Vector Search Module — Stages 1 & 2 of the Fashion Search Pipeline.

Stage 1: Metadata pre-filtering (ChromaDB ``where`` clause built from the
parsed query attributes).

Stage 2: Dual-channel vector search — the same query is embedded through both
CLIP (visual channel) and SentenceTransformer (text channel). Results are
fused with configurable weights to produce a single ranked candidate list.
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import open_clip

from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_WEIGHT,
    TEXT_WEIGHT,
    STAGE2_TOP_K,
    VISUAL_COLLECTION_NAME,
    TEXT_COLLECTION_NAME,
    VECTOR_STORE_DIR,
    DATASET_DIR,
)

from indexer.vector_store import VectorStore
from indexer.embedding_generator import EmbeddingGenerator


class VectorSearcher:
    """Performs dual-channel vector search with optional metadata filtering.

    The searcher maintains its own CLIP text encoder (loaded via *open_clip*)
    so that query text can be projected into the same embedding space as the
    indexed images.  A separate ``EmbeddingGenerator`` instance is used for
    SentenceTransformer text embeddings.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialise vector store, embedding generator, and CLIP text encoder.

        Parameters
        ----------
        device : str, optional
            PyTorch device string (default ``"cpu"``).
        """
        self.device = device

        # Shared vector store (ChromaDB)
        self.vector_store = VectorStore(persist_dir=VECTOR_STORE_DIR)

        # Embedding generator (SentenceTransformer text embeddings)
        self.embedding_generator = EmbeddingGenerator()

        # Load CLIP for query-text → visual-space encoding
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self.device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.clip_model.eval()

        print(
            f"[VectorSearcher] Initialised — CLIP: {CLIP_MODEL_NAME}, "
            f"device: {self.device}"
        )

    # ------------------------------------------------------------------
    # Metadata filter builder
    # ------------------------------------------------------------------

    def _build_metadata_filter(self, parsed_query: dict) -> Optional[dict]:
        """Build a ChromaDB ``where`` filter from the parsed query.

        Parameters
        ----------
        parsed_query : dict
            Output of ``QueryParser.parse_query``.

        Returns
        -------
        dict | None
            A ChromaDB-compatible ``where`` clause, or ``None`` if the parsed
            query contains no filterable attributes.
        """
        conditions: list[dict] = []

        # Colors — stored as a JSON-stringified list in metadata; use $contains
        colors = parsed_query.get("colors", [])
        for color in colors:
            conditions.append({"colors": {"$contains": color.lower()}})

        # Environment
        env = parsed_query.get("environment")
        if env:
            conditions.append({"environment": {"$eq": env.lower()}})

        # Style
        style = parsed_query.get("style")
        if style:
            conditions.append({"style": {"$eq": style.lower()}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_query_clip(self, query: str) -> list[float]:
        """Encode query text into the CLIP visual embedding space.

        Parameters
        ----------
        query : str
            The raw user query string.

        Returns
        -------
        list[float]
            L2-normalised CLIP text embedding.
        """
        tokens = self.clip_tokenizer([query]).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze(0).cpu().tolist()

    def _encode_query_text(self, query: str) -> list[float]:
        """Encode query text using SentenceTransformer (via EmbeddingGenerator).

        Parameters
        ----------
        query : str
            The raw user query string.

        Returns
        -------
        list[float]
            SentenceTransformer text embedding.
        """
        embedding = self.embedding_generator.generate_text_embedding(query)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        parsed_query: dict,
        top_k: int = STAGE2_TOP_K,
    ) -> list[dict]:
        """Run dual-channel vector search with optional metadata filtering.

        Parameters
        ----------
        query : str
            The raw user query.
        parsed_query : dict
            Structured query attributes from ``QueryParser``.
        top_k : int, optional
            Number of candidates to return (default ``STAGE2_TOP_K``).

        Returns
        -------
        list[dict]
            Ranked candidates with fused scores.  Each dict contains:
            ``id``, ``image_path``, ``caption``, ``garments``, ``colors``,
            ``environment``, ``style``, ``clip_score``, ``text_score``,
            ``fused_score``.
        """
        metadata_filter = self._build_metadata_filter(parsed_query)

        # Encode query in both embedding spaces
        clip_embedding = self._encode_query_clip(query)
        text_embedding = self._encode_query_text(query)

        # ---- Query visual index (CLIP) ----
        clip_results = self._query_index(
            VISUAL_COLLECTION_NAME, clip_embedding, top_k, metadata_filter,
        )

        # ---- Query text index (SentenceTransformer) ----
        text_results = self._query_index(
            TEXT_COLLECTION_NAME, text_embedding, top_k, metadata_filter,
        )

        # ---- Fuse scores ----
        candidates = self._fuse_scores(clip_results, text_results, top_k)
        print(f"[VectorSearcher] Returning {len(candidates)} fused candidates.")
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_index(
        self,
        collection_name: str,
        embedding: list[float],
        top_k: int,
        metadata_filter: Optional[dict],
    ) -> dict:
        """Query a single ChromaDB collection, falling back to unfiltered
        search if the filter is too restrictive.

        Returns the raw ChromaDB result dict.
        """
        results = self.vector_store.query_collection(
            collection_name=collection_name,
            query_embedding=embedding,
            n_results=top_k,
            where=metadata_filter,
        )

        # Check whether the filter was too restrictive
        n_returned = len(results.get("ids", [[]])[0]) if results else 0
        if metadata_filter is not None and n_returned < top_k:
            print(
                f"[VectorSearcher] Filter returned only {n_returned} results "
                f"from '{collection_name}' (need {top_k}). "
                f"Retrying WITHOUT metadata filter."
            )
            results = self.vector_store.query_collection(
                collection_name=collection_name,
                query_embedding=embedding,
                n_results=top_k,
                where=None,
            )

        return results

    def _fuse_scores(
        self,
        clip_results: dict,
        text_results: dict,
        top_k: int,
    ) -> list[dict]:
        """Fuse CLIP and text scores into a single ranked list.

        ChromaDB returns *distances* (lower = better for cosine).  We convert
        to similarities via ``1 - distance``.
        """
        candidate_map: dict[str, dict] = {}

        # Process CLIP results
        if clip_results and clip_results.get("ids"):
            ids = clip_results["ids"][0]
            distances = clip_results["distances"][0]
            metadatas = clip_results["metadatas"][0]
            for idx, doc_id in enumerate(ids):
                clip_score = 1.0 - distances[idx]  # cosine distance → similarity
                meta = metadatas[idx] if metadatas else {}
                candidate_map[doc_id] = {
                    "id": doc_id,
                    "image_path": str(DATASET_DIR / "images" / Path(meta.get("image_path", "")).name),
                    "caption": meta.get("caption", ""),
                    "garments": meta.get("garments", "[]"),
                    "colors": meta.get("colors", "[]"),
                    "environment": meta.get("environment", "unknown"),
                    "style": meta.get("style", "other"),
                    "clip_score": clip_score,
                    "text_score": 0.0,
                    "fused_score": 0.0,
                }

        # Process text results
        if text_results and text_results.get("ids"):
            ids = text_results["ids"][0]
            distances = text_results["distances"][0]
            metadatas = text_results["metadatas"][0]
            for idx, doc_id in enumerate(ids):
                text_score = 1.0 - distances[idx]
                if doc_id in candidate_map:
                    candidate_map[doc_id]["text_score"] = text_score
                else:
                    meta = metadatas[idx] if metadatas else {}
                    candidate_map[doc_id] = {
                        "id": doc_id,
                        "image_path": str(DATASET_DIR / "images" / Path(meta.get("image_path", "")).name),
                        "caption": meta.get("caption", ""),
                        "garments": meta.get("garments", "[]"),
                        "colors": meta.get("colors", "[]"),
                        "environment": meta.get("environment", "unknown"),
                        "style": meta.get("style", "other"),
                        "clip_score": 0.0,
                        "text_score": text_score,
                        "fused_score": 0.0,
                    }

        # Compute fused scores
        for cand in candidate_map.values():
            cand["fused_score"] = (
                CLIP_WEIGHT * cand["clip_score"]
                + TEXT_WEIGHT * cand["text_score"]
            )

        # Sort and return top-k
        ranked = sorted(
            candidate_map.values(), key=lambda c: c["fused_score"], reverse=True,
        )
        return ranked[:top_k]
