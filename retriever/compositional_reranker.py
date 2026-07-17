"""
Compositional Re-ranker — Stage 3 of the Fashion Search Pipeline.

Re-ranks Stage-2 candidates by measuring how well **each individual garment**
requested in the query is matched by the garments present in a candidate
image.  This captures *compositional* intent: "red shirt and blue pants"
should rank differently from "blue shirt and red pants".

Garment descriptors are embedded with SentenceTransformer and matched greedily
(Hungarian-style best-match without the scipy dependency).
"""

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    SBERT_MODEL_NAME,
    COMPOSITIONAL_WEIGHT,
    VECTOR_WEIGHT,
    STAGE3_TOP_K,
)


class CompositionalReranker:
    """Pairwise garment-level compositional re-ranker.

    For every candidate, we compute a *compositional score* that captures how
    well the set of garments in the candidate matches the set of garments
    requested in the query.  The compositional score is blended with the
    Stage-2 fused score to produce a ``reranked_score``.
    """

    def __init__(self) -> None:
        """Load the SentenceTransformer model for garment embedding."""
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print(
            f"[CompositionalReranker] Initialised with SBERT model: "
            f"{SBERT_MODEL_NAME}"
        )

    # ------------------------------------------------------------------
    # Garment similarity
    # ------------------------------------------------------------------

    def _compute_garment_similarity(
        self,
        query_garments: list[str],
        candidate_garments: list[str],
    ) -> float:
        """Compute compositional similarity between two garment lists.

        Each garment descriptor (e.g. "red blazer") is embedded, a pairwise
        cosine-similarity matrix is computed, and a greedy best-match
        assignment is used: for each query garment the highest-similarity
        candidate garment is selected and removed from the pool.

        Parameters
        ----------
        query_garments : list[str]
            Garments extracted from the user query.
        candidate_garments : list[str]
            Garments present in the candidate image's metadata.

        Returns
        -------
        float
            Average best-match cosine similarity (0–1 range).
            Returns 0.5 (neutral) if either list is empty.
        """
        if not query_garments or not candidate_garments:
            return 0.5

        # Embed both lists
        q_embeddings = self.sbert_model.encode(
            query_garments, convert_to_numpy=True, normalize_embeddings=True,
        )
        c_embeddings = self.sbert_model.encode(
            candidate_garments, convert_to_numpy=True, normalize_embeddings=True,
        )

        # Pairwise cosine similarity matrix (already L2-normalised → dot product)
        sim_matrix = q_embeddings @ c_embeddings.T  # shape (Q, C)

        # Greedy best-match assignment
        used_candidates: set[int] = set()
        match_scores: list[float] = []

        for q_idx in range(sim_matrix.shape[0]):
            best_score = -1.0
            best_c_idx = -1
            for c_idx in range(sim_matrix.shape[1]):
                if c_idx in used_candidates:
                    continue
                if sim_matrix[q_idx, c_idx] > best_score:
                    best_score = float(sim_matrix[q_idx, c_idx])
                    best_c_idx = c_idx
            if best_c_idx >= 0:
                used_candidates.add(best_c_idx)
                match_scores.append(best_score)
            else:
                # No remaining candidate to match — penalise with 0
                match_scores.append(0.0)

        return float(np.mean(match_scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[dict],
        query_garments: list[str],
    ) -> list[dict]:
        """Re-rank candidates using compositional garment matching.

        Parameters
        ----------
        candidates : list[dict]
            Stage-2 candidate list (must contain ``garments`` and
            ``fused_score`` keys).
        query_garments : list[str]
            Garments parsed from the user query.

        Returns
        -------
        list[dict]
            Top ``STAGE3_TOP_K`` candidates sorted by ``reranked_score``
            (descending).  Each dict is augmented with
            ``compositional_score`` and ``reranked_score``.
        """
        for candidate in candidates:
            # Parse candidate garments from JSON-stringified metadata
            raw_garments = candidate.get("garments", "[]")
            try:
                cand_garments = (
                    json.loads(raw_garments)
                    if isinstance(raw_garments, str)
                    else raw_garments
                )
            except json.JSONDecodeError:
                cand_garments = []

            comp_score = self._compute_garment_similarity(
                query_garments, cand_garments,
            )
            candidate["compositional_score"] = comp_score
            candidate["reranked_score"] = (
                VECTOR_WEIGHT * candidate["fused_score"]
                + COMPOSITIONAL_WEIGHT * comp_score
            )

        # Sort by reranked_score and take top-k
        candidates.sort(key=lambda c: c["reranked_score"], reverse=True)
        top_candidates = candidates[: STAGE3_TOP_K]

        print(
            f"[CompositionalReranker] Re-ranked {len(candidates)} → "
            f"top {len(top_candidates)} candidates."
        )
        return top_candidates
