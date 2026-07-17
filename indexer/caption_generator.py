"""
Caption Generator Module
=========================
Generates descriptive captions for fashion images using the BLIP
(Bootstrapping Language-Image Pre-training) model from Salesforce.

Supports single-image and batch captioning with JSON caching so that
re-runs skip already-processed images.
"""

import sys
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from config import BLIP_MODEL_NAME, CAPTIONS_CACHE, MAX_IMAGES


class CaptionGenerator:
    """Generates natural-language captions for images using BLIP.

    Attributes:
        device: Torch device string ('cpu' or 'cuda').
        processor: BLIP image processor / tokenizer.
        model: BLIP conditional generation model.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialise the BLIP model and processor.

        Args:
            device: Torch device to run inference on ('cpu' or 'cuda').
        """
        self.device = device
        print(f"[CaptionGenerator] Loading BLIP model: {BLIP_MODEL_NAME} on {device} ...")
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
        self.model.eval()
        print("[CaptionGenerator] Model loaded successfully.")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a descriptive caption for a single PIL image.

        Args:
            image: A PIL Image in RGB mode.

        Returns:
            A string caption describing the image content.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
            )
        caption: str = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def generate_captions_batch(
        self, image_dir: Path, batch_size: int = 8
    ) -> dict[str, str]:
        """Generate captions for every image in *image_dir*.

        Already-captioned images (present in the JSON cache) are skipped so
        that the pipeline can be safely restarted without re-processing.

        Args:
            image_dir: Directory containing image files.
            batch_size: Number of images to process per forward pass.

        Returns:
            Dictionary mapping ``{filename: caption}`` for all images.
        """
        # ------------------------------------------------------------------
        # Load existing cache
        # ------------------------------------------------------------------
        captions: dict[str, str] = {}
        if CAPTIONS_CACHE.exists():
            try:
                with open(CAPTIONS_CACHE, "r", encoding="utf-8") as f:
                    captions = json.load(f)
                print(f"[CaptionGenerator] Loaded {len(captions)} cached captions.")
            except (json.JSONDecodeError, IOError) as exc:
                print(f"[CaptionGenerator] Warning: could not load cache – {exc}")

        # ------------------------------------------------------------------
        # Collect image paths that still need processing
        # ------------------------------------------------------------------
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        all_images = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_extensions
        )

        # Respect MAX_IMAGES limit
        all_images = all_images[:MAX_IMAGES]

        pending = [p for p in all_images if p.name not in captions]
        print(
            f"[CaptionGenerator] {len(all_images)} images found, "
            f"{len(captions)} cached, {len(pending)} to process."
        )

        if not pending:
            return captions

        # ------------------------------------------------------------------
        # Process in batches
        # ------------------------------------------------------------------
        for batch_start in tqdm(
            range(0, len(pending), batch_size),
            desc="Captioning",
            unit="batch",
        ):
            batch_paths = pending[batch_start : batch_start + batch_size]
            batch_images: list[Image.Image] = []
            valid_paths: list[Path] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(path)
                except Exception as exc:
                    print(f"[CaptionGenerator] Skipping {path.name}: {exc}")

            if not batch_images:
                continue

            try:
                inputs = self.processor(
                    images=batch_images, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=5,
                    )
                for idx, path in enumerate(valid_paths):
                    caption = self.processor.decode(
                        output_ids[idx], skip_special_tokens=True
                    )
                    captions[path.name] = caption.strip()
            except Exception as exc:
                print(f"[CaptionGenerator] Batch error: {exc}")
                # Fall back to one-by-one processing for this batch
                for img, path in zip(batch_images, valid_paths):
                    try:
                        captions[path.name] = self.generate_caption(img)
                    except Exception as inner_exc:
                        print(
                            f"[CaptionGenerator] Skipping {path.name}: {inner_exc}"
                        )

            # Incremental save so progress is not lost on crash
            try:
                with open(CAPTIONS_CACHE, "w", encoding="utf-8") as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)
            except IOError as exc:
                print(f"[CaptionGenerator] Warning: could not save cache – {exc}")

        print(f"[CaptionGenerator] Captioning complete – {len(captions)} total captions.")
        return captions
