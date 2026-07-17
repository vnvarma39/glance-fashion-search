"""
Build Index – Full Indexing Pipeline
======================================
Orchestrates the complete indexing workflow for the fashion search engine:

1. Generate image captions  (BLIP)
2. Extract structured attributes  (LLM via OpenRouter)
3. Generate CLIP visual embeddings  (OpenCLIP)
4. Generate text embeddings  (SentenceTransformer)
5. Store everything in ChromaDB  (visual_index + text_index)

Run this script directly to build or update the index:

    python -m indexer.build_index
"""

import sys
import json
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import (
    DATASET_DIR,
    VISUAL_COLLECTION_NAME,
    TEXT_COLLECTION_NAME,
)

from indexer.caption_generator import CaptionGenerator
from indexer.attribute_extractor import AttributeExtractor
from indexer.embedding_generator import EmbeddingGenerator
from indexer.vector_store import VectorStore


def build_index(image_dir: Path = DATASET_DIR) -> None:
    """Run the full indexing pipeline.

    Args:
        image_dir: Directory containing the fashion images to index.
    """
    print("=" * 70)
    print("  GLANCE – Fashion Image Indexing Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Image directory: {image_dir}\n")

    # ==================================================================
    # STEP 1 / 5 – Generate captions
    # ==================================================================
    print("[STEP 1/5] Generating image captions with BLIP ...")
    caption_gen = CaptionGenerator(device=device)
    captions: dict[str, str] = caption_gen.generate_captions_batch(image_dir)
    print(f"[STEP 1/5] Done – {len(captions)} captions available.\n")

    # ==================================================================
    # STEP 2 / 5 – Extract structured attributes
    # ==================================================================
    print("[STEP 2/5] Extracting fashion attributes via LLM ...")
    attr_extractor = AttributeExtractor()
    attributes: dict[str, dict] = attr_extractor.extract_all(captions)
    print(f"[STEP 2/5] Done – {len(attributes)} attribute sets available.\n")

    # ==================================================================
    # STEP 3 / 5 – Generate CLIP visual embeddings
    # ==================================================================
    print("[STEP 3/5] Generating CLIP visual embeddings ...")
    emb_gen = EmbeddingGenerator(device=device)
    clip_embeddings: dict[str, list[float]] = emb_gen.generate_clip_embeddings_batch(
        image_dir
    )
    print(f"[STEP 3/5] Done – {len(clip_embeddings)} CLIP embeddings.\n")

    # ==================================================================
    # STEP 4 / 5 – Generate text embeddings for captions
    # ==================================================================
    print("[STEP 4/5] Generating text embeddings for captions ...")
    text_embeddings: dict[str, list[float]] = emb_gen.generate_text_embeddings_batch(
        captions
    )
    print(f"[STEP 4/5] Done – {len(text_embeddings)} text embeddings.\n")

    # ==================================================================
    # STEP 5 / 5 – Store in ChromaDB
    # ==================================================================
    print("[STEP 5/5] Storing embeddings in ChromaDB ...")
    store = VectorStore()

    # Determine the common set of filenames across all data sources
    common_files = sorted(
        set(captions.keys())
        & set(attributes.keys())
        & set(clip_embeddings.keys())
        & set(text_embeddings.keys())
    )
    print(f"[STEP 5/5] {len(common_files)} images have all data – indexing ...")

    if not common_files:
        print("[STEP 5/5] WARNING: No images with complete data to index.")
        return

    # Build parallel lists for ChromaDB insertion
    ids: list[str] = []
    clip_embs: list[list[float]] = []
    text_embs: list[list[float]] = []
    metadatas: list[dict] = []

    for fname in common_files:
        attrs = attributes.get(fname, {})
        meta = {
            "caption": captions[fname],
            "garments": json.dumps(attrs.get("garments", [])),
            "colors": json.dumps(attrs.get("colors", [])),
            "environment": attrs.get("environment", "unknown"),
            "style": attrs.get("style", "other"),
            "image_path": str(image_dir / fname),
        }
        ids.append(fname)
        clip_embs.append(clip_embeddings[fname])
        text_embs.append(text_embeddings[fname])
        metadatas.append(meta)

    # --- visual_index (CLIP embeddings) ----------------------------------
    store.create_collection(VISUAL_COLLECTION_NAME)
    store.add_to_collection(
        collection_name=VISUAL_COLLECTION_NAME,
        ids=ids,
        embeddings=clip_embs,
        metadatas=metadatas,
    )

    # --- text_index (SentenceTransformer embeddings) ----------------------
    store.create_collection(TEXT_COLLECTION_NAME)
    store.add_to_collection(
        collection_name=TEXT_COLLECTION_NAME,
        ids=ids,
        embeddings=text_embs,
        metadatas=metadatas,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    visual_count = store.get_collection_count(VISUAL_COLLECTION_NAME)
    text_count = store.get_collection_count(TEXT_COLLECTION_NAME)

    print("\n" + "=" * 70)
    print("  Indexing Summary")
    print("=" * 70)
    print(f"  Total images indexed : {len(common_files)}")
    print(f"  visual_index count   : {visual_count}")
    print(f"  text_index count     : {text_count}")
    print("=" * 70)
    print("  Indexing complete!")
    print("=" * 70 + "\n")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    start = time.time()
    build_index()
    elapsed = time.time() - start

    minutes, seconds = divmod(elapsed, 60)
    print(f"Total time: {int(minutes)}m {seconds:.1f}s")
