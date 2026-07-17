"""
Embedding Generator Module
============================
Produces two kinds of embeddings for the fashion search engine:

1. **CLIP visual embeddings** – generated from raw images via OpenCLIP.
2. **SentenceTransformer text embeddings** – generated from caption strings.

Both are returned as plain Python lists of floats for easy serialisation
and storage in ChromaDB.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED, SBERT_MODEL_NAME, MAX_IMAGES


class EmbeddingGenerator:
    """Generates CLIP and SentenceTransformer embeddings.

    Attributes:
        device: Torch device string.
        clip_model: OpenCLIP vision model.
        clip_preprocess: OpenCLIP image transform pipeline.
        sbert_model: SentenceTransformer model for text embedding.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Load CLIP and SentenceTransformer models.

        Args:
            device: Torch device to run inference on ('cpu' or 'cuda').
        """
        self.device = device

        # --- CLIP -----------------------------------------------------------
        print(
            f"[EmbeddingGenerator] Loading CLIP model: "
            f"{CLIP_MODEL_NAME} (pretrained={CLIP_PRETRAINED}) on {device} ..."
        )
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
            )
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        # --- SentenceTransformer --------------------------------------------
        print(f"[EmbeddingGenerator] Loading SentenceTransformer: {SBERT_MODEL_NAME} ...")
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=self.device)

        print("[EmbeddingGenerator] Models loaded successfully.")

    # ------------------------------------------------------------------
    # Single-item helpers
    # ------------------------------------------------------------------

    def generate_clip_embedding(self, image: Image.Image) -> list[float]:
        """Return a normalised CLIP embedding for a single PIL image.

        Args:
            image: A PIL Image (RGB).

        Returns:
            List of floats representing the normalised embedding vector.
        """
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu().tolist()

    def generate_text_embedding(self, text: str) -> list[float]:
        """Return a SentenceTransformer embedding for a text string.

        Args:
            text: Input text (e.g. a caption).

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.sbert_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def generate_clip_embeddings_batch(
        self, image_dir: Path
    ) -> dict[str, list[float]]:
        """Generate CLIP embeddings for all images in a directory.

        Args:
            image_dir: Directory containing image files.

        Returns:
            Dictionary mapping ``{filename: embedding}`` for every
            successfully processed image.
        """
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        all_images = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_extensions
        )

        # Respect MAX_IMAGES limit
        all_images = all_images[:MAX_IMAGES]

        embeddings: dict[str, list[float]] = {}

        for path in tqdm(all_images, desc="CLIP embeddings", unit="img"):
            try:
                img = Image.open(path).convert("RGB")
                embeddings[path.name] = self.generate_clip_embedding(img)
            except Exception as exc:
                print(f"[EmbeddingGenerator] Skipping {path.name}: {exc}")

        print(
            f"[EmbeddingGenerator] Generated {len(embeddings)} CLIP embeddings."
        )
        return embeddings

    def generate_text_embeddings_batch(
        self, captions: dict[str, str]
    ) -> dict[str, list[float]]:
        """Generate SentenceTransformer embeddings for a batch of captions.

        Args:
            captions: Mapping ``{filename: caption}``.

        Returns:
            Dictionary mapping ``{filename: embedding}``.
        """
        filenames = list(captions.keys())
        texts = list(captions.values())

        print(f"[EmbeddingGenerator] Encoding {len(texts)} captions ...")
        raw_embeddings = self.sbert_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        embeddings: dict[str, list[float]] = {}
        for fname, emb in zip(filenames, raw_embeddings):
            embeddings[fname] = emb.tolist()

        print(
            f"[EmbeddingGenerator] Generated {len(embeddings)} text embeddings."
        )
        return embeddings
