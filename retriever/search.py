"""
Fashion Search Engine — Main Orchestrator.

Coordinates all four stages of the retrieval pipeline:

1. **Query Parsing** — LLM extracts structured attributes from the query.
2. **Vector Search** — Dual-channel (CLIP + SentenceTransformer) search with
   optional metadata filtering and score fusion.
3. **Compositional Re-ranking** — Garment-level pairwise matching re-ranks
   candidates to honour compositional intent.
4. **VLM-as-Judge** — A vision-language model scores each remaining candidate
   image for final ranking.

Usage::

    python -m retriever.search --query "red blazer with blue jeans"
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FINAL_TOP_K, RESULTS_DIR

from retriever.query_parser import QueryParser
from retriever.vector_search import VectorSearcher
from retriever.compositional_reranker import CompositionalReranker
from retriever.vlm_judge import VLMJudge


class FashionSearchEngine:
    """End-to-end fashion image search engine.

    Wraps QueryParser → VectorSearcher → CompositionalReranker → VLMJudge
    into a single ``search()`` call.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialise all pipeline components.

        Parameters
        ----------
        device : str, optional
            PyTorch device string for model inference (default ``"cpu"``).
        """
        print("=" * 60)
        print("  Initialising Fashion Search Engine")
        print("=" * 60)

        self.query_parser = QueryParser()
        self.vector_searcher = VectorSearcher(device=device)
        self.compositional_reranker = CompositionalReranker()
        self.vlm_judge = VLMJudge()

        print("=" * 60)
        print("  Fashion Search Engine — Ready")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Full search (all 4 stages)
    # ------------------------------------------------------------------

    def search(self, query: str, verbose: bool = True) -> list[dict]:
        """Run the full 4-stage search pipeline.

        Parameters
        ----------
        query : str
            Natural-language fashion search query.
        verbose : bool, optional
            Print progress & timing information (default ``True``).

        Returns
        -------
        list[dict]
            Top ``FINAL_TOP_K`` candidate dicts ranked by ``final_score``.
        """
        print(f"\n{'─' * 60}")
        print(f"  Query: {query}")
        print(f"{'─' * 60}\n")

        total_start = time.time()

        # ── Stage 1 + 2: Parse query → Vector search ──────────────
        stage_start = time.time()
        if verbose:
            print("[Stage 1] Parsing query …")

        parsed_query = self.query_parser.parse_query(query)

        if verbose:
            print(f"[Stage 1] Parsed attributes: {parsed_query}")
            print("[Stage 2] Running dual-channel vector search …")

        stage2_results = self.vector_searcher.search(query, parsed_query)

        if verbose:
            elapsed = time.time() - stage_start
            print(
                f"[Stage 2] ✓ {len(stage2_results)} candidates "
                f"({elapsed:.2f}s)\n"
            )

        # ── Stage 3: Compositional re-ranking ─────────────────────
        stage_start = time.time()
        if verbose:
            print("[Stage 3] Compositional re-ranking …")

        stage3_results = self.compositional_reranker.rerank(
            stage2_results, parsed_query.get("garments", []),
        )

        if verbose:
            elapsed = time.time() - stage_start
            print(
                f"[Stage 3] ✓ {len(stage3_results)} candidates "
                f"({elapsed:.2f}s)\n"
            )

        # ── Stage 4: VLM-as-Judge ─────────────────────────────────
        stage_start = time.time()
        if verbose:
            print("[Stage 4] VLM-as-Judge scoring …")

        final_results = self.vlm_judge.judge_candidates(stage3_results, query)

        if verbose:
            elapsed = time.time() - stage_start
            print(
                f"[Stage 4] ✓ {len(final_results)} results "
                f"({elapsed:.2f}s)\n"
            )

        total_elapsed = time.time() - total_start
        if verbose:
            print(f"{'─' * 60}")
            print(f"  Search complete — {len(final_results)} results "
                  f"in {total_elapsed:.2f}s")
            print(f"{'─' * 60}\n")

        return final_results

    # ------------------------------------------------------------------
    # Fast search (skip VLM stage)
    # ------------------------------------------------------------------

    def search_without_vlm(self, query: str) -> list[dict]:
        """Run Stages 1–3 only, skipping the slow VLM judge.

        Useful for fast iteration and testing.

        Parameters
        ----------
        query : str
            Natural-language fashion search query.

        Returns
        -------
        list[dict]
            Top ``FINAL_TOP_K`` candidates ranked by ``reranked_score``.
        """
        print(f"\n{'─' * 60}")
        print(f"  Query (no VLM): {query}")
        print(f"{'─' * 60}\n")

        total_start = time.time()

        # Stage 1+2
        print("[Stage 1] Parsing query …")
        parsed_query = self.query_parser.parse_query(query)
        print(f"[Stage 1] Parsed: {parsed_query}")

        print("[Stage 2] Dual-channel vector search …")
        stage2_results = self.vector_searcher.search(query, parsed_query)
        print(f"[Stage 2] ✓ {len(stage2_results)} candidates")

        # Stage 3
        print("[Stage 3] Compositional re-ranking …")
        stage3_results = self.compositional_reranker.rerank(
            stage2_results, parsed_query.get("garments", []),
        )
        print(f"[Stage 3] ✓ {len(stage3_results)} candidates")

        # Take top FINAL_TOP_K
        final_results = stage3_results[: FINAL_TOP_K]

        total_elapsed = time.time() - total_start
        print(f"\n  Search complete — {len(final_results)} results "
              f"in {total_elapsed:.2f}s\n")

        return final_results


# ======================================================================
# CLI entry point
# ======================================================================

def _print_results_table(results: list[dict]) -> None:
    """Pretty-print search results as a formatted table."""
    print(f"\n{'═' * 90}")
    print(f"{'Rank':<6}{'Score':>8}  {'Image':>30}  {'Caption'}")
    print(f"{'─' * 90}")

    for idx, r in enumerate(results, start=1):
        score = r.get(
            "final_score",
            r.get("reranked_score", r.get("fused_score", 0.0)),
        )
        image = Path(r.get("image_path", "")).name or "—"
        caption = r.get("caption", "")
        # Truncate long captions for table display
        if len(caption) > 40:
            caption = caption[:37] + "…"
        print(f"  {idx:<4}{score:>8.4f}  {image:>30}  {caption}")

        # Print VLM reasoning if available
        vlm_reasoning = r.get("vlm_reasoning")
        if vlm_reasoning:
            print(f"{'':>16}VLM: {vlm_reasoning}")

    print(f"{'═' * 90}\n")


def main() -> None:
    """CLI entry point for fashion search."""
    parser = argparse.ArgumentParser(
        description="Fashion Image Search Engine — DIC + VLM-as-Judge",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Natural-language fashion search query.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM-as-Judge stage (faster, but less accurate).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=FINAL_TOP_K,
        help=f"Number of results to return (default {FINAL_TOP_K}).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result visualisation as PNG to the results directory.",
    )

    args = parser.parse_args()

    # Build engine
    engine = FashionSearchEngine()

    # Run search
    if args.no_vlm:
        results = engine.search_without_vlm(args.query)
    else:
        results = engine.search(args.query)

    # Trim to requested top-k (in case it differs from default)
    results = results[: args.top_k]

    # Print results
    _print_results_table(results)

    # Optionally save visualisation
    if args.save:
        try:
            from utils.visualize import display_results

            save_path = RESULTS_DIR / "search_results.png"
            display_results(results, args.query, save_path=save_path)
        except ImportError as exc:
            print(f"[Search] Could not import visualisation utilities: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Search] Visualisation failed: {exc}")


if __name__ == "__main__":
    main()
