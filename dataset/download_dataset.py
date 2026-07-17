"""
Dataset Download Script
=======================
Downloads and prepares the Fashionpedia dataset (limited to 1000 images).
Falls back to HuggingFace fashion-product-images if Fashionpedia is unavailable.

Usage:
    python dataset/download_dataset.py

The script saves images to dataset/images/ and creates a metadata.csv file.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import DATASET_DIR, DATASET_NAME, MAX_IMAGES, FALLBACK_DATASET


def download_fashionpedia(max_images: int = 1000) -> pd.DataFrame:
    """
    Download Fashionpedia dataset from HuggingFace and save images locally.
    
    Args:
        max_images: Maximum number of images to download.
        
    Returns:
        DataFrame with image metadata (filename, source, split).
    """
    from datasets import load_dataset
    
    print(f"[INFO] Loading Fashionpedia dataset from HuggingFace...")
    print(f"[INFO] This may take a few minutes on first run (downloading images)...")
    
    try:
        # Load the dataset - Fashionpedia has train/val splits
        dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
        
        # Shuffle and limit to max_images
        total_available = len(dataset)
        print(f"[INFO] Dataset has {total_available} images. Sampling {max_images}...")
        
        if total_available > max_images:
            indices = random.sample(range(total_available), max_images)
            dataset = dataset.select(indices)
        
        metadata_records = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Saving images")):
            # Fashionpedia items have an 'image' field
            img = item["image"]
            
            if not isinstance(img, Image.Image):
                continue
            
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            filename = f"fashionpedia_{idx:04d}.jpg"
            filepath = DATASET_DIR / filename
            
            # Resize to reasonable size (max 512px on longest side) to save space
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            img.save(filepath, "JPEG", quality=90)
            
            metadata_records.append({
                "filename": filename,
                "source": "fashionpedia",
                "original_index": idx,
            })
        
        return pd.DataFrame(metadata_records)
        
    except Exception as e:
        print(f"[WARNING] Fashionpedia download failed: {e}")
        print(f"[INFO] Falling back to {FALLBACK_DATASET}...")
        return download_fallback(max_images)


def download_fallback(max_images: int = 1000) -> pd.DataFrame:
    """
    Fallback: Download fashion-product-images-small from HuggingFace.
    These are product images on white backgrounds - clean and easy to process.
    
    Args:
        max_images: Maximum number of images to download.
        
    Returns:
        DataFrame with image metadata.
    """
    from datasets import load_dataset
    
    print(f"[INFO] Loading fallback dataset: {FALLBACK_DATASET}...")
    
    dataset = load_dataset(FALLBACK_DATASET, split="train", trust_remote_code=True)
    
    total_available = len(dataset)
    print(f"[INFO] Fallback dataset has {total_available} images. Sampling {max_images}...")
    
    if total_available > max_images:
        indices = random.sample(range(total_available), max_images)
        dataset = dataset.select(indices)
    
    metadata_records = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Saving images")):
        img = item.get("image")
        
        if img is None or not isinstance(img, Image.Image):
            continue
            
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        filename = f"fashion_{idx:04d}.jpg"
        filepath = DATASET_DIR / filename
        
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        img.save(filepath, "JPEG", quality=90)
        
        metadata_records.append({
            "filename": filename,
            "source": "fashion-product-images",
            "original_index": idx,
        })
    
    return pd.DataFrame(metadata_records)


def verify_dataset(metadata: pd.DataFrame) -> None:
    """Print dataset statistics for verification."""
    total = len(metadata)
    sources = metadata["source"].value_counts().to_dict()
    
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total images saved: {total}")
    print(f"Sources: {sources}")
    print(f"Location: {DATASET_DIR}")
    
    # Count actual files
    actual_files = list(DATASET_DIR.glob("*.jpg"))
    print(f"Files on disk: {len(actual_files)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    
    # Check if images already exist
    existing = list(DATASET_DIR.glob("*.jpg"))
    if len(existing) >= MAX_IMAGES:
        print(f"[INFO] Found {len(existing)} images already in {DATASET_DIR}. Skipping download.")
        print(f"[INFO] Delete the images folder to re-download.")
        sys.exit(0)
    
    print(f"[INFO] Target: {MAX_IMAGES} images")
    print(f"[INFO] Saving to: {DATASET_DIR}")
    
    # Download
    metadata = download_fashionpedia(MAX_IMAGES)
    
    # Save metadata
    metadata_path = DATASET_DIR.parent / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    print(f"[INFO] Metadata saved to {metadata_path}")
    
    # Verify
    verify_dataset(metadata)
    
    print("[DONE] Dataset ready for indexing.")
