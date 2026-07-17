"""
Attribute Extractor Module
===========================
Extracts structured fashion attributes (garments, colours, environment,
style) from natural-language image captions by calling a Large Language
Model via the OpenRouter API.

Rate limiting and JSON-cache support are built in so the pipeline can
be restarted without burning duplicate API calls.
"""

import sys
import json
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from tqdm import tqdm

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    ATTRIBUTE_EXTRACTION_PROMPT,
    API_REQUESTS_PER_MINUTE,
    API_RETRY_DELAY,
    ATTRIBUTES_CACHE,
)

# Default attribute dict returned when extraction fails
_DEFAULT_ATTRIBUTES: dict = {
    "garments": [],
    "colors": [],
    "environment": "unknown",
    "style": "other",
}

class AttributeExtractor:
    """Extracts structured fashion attributes from captions using fast keyword matching.
    
    This completely bypasses API rate limits and runs instantly offline.
    """

    def __init__(self) -> None:
        """Initialize the keyword vocabularies."""
        print("[AttributeExtractor] Initialised with Offline Keyword Matcher (No API limits!).")
        
        self.garment_vocab = [
            "shirt", "t-shirt", "pants", "jeans", "dress", "skirt", "jacket", 
            "coat", "sweater", "hoodie", "suit", "blazer", "shorts", "tie", 
            "hat", "shoes", "sneakers", "boots", "glasses", "sunglasses", "bag"
        ]
        
        self.color_vocab = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey", 
            "brown", "orange", "pink", "purple", "beige", "navy", "maroon"
        ]
        
        self.env_vocab = {
            "indoor": ["indoor", "room", "inside", "office", "home", "building", "studio"],
            "outdoor": ["outdoor", "outside", "nature", "park", "street", "city", "urban", "road"],
            "urban": ["urban", "city", "street", "building"],
            "nature": ["nature", "park", "tree", "grass", "forest"]
        }
        
        self.style_vocab = {
            "formal": ["formal", "suit", "tie", "blazer", "business", "office", "professional"],
            "casual": ["casual", "t-shirt", "jeans", "hoodie", "sneakers", "relaxed"],
            "winter": ["winter", "coat", "jacket", "snow", "cold", "sweater"],
            "summer": ["summer", "shorts", "sunglasses", "beach", "warm"]
        }

    # ------------------------------------------------------------------
    # Single caption → structured attributes
    # ------------------------------------------------------------------

    def extract_attributes(self, caption: str) -> dict:
        """Extract attributes by matching keywords in the caption."""
        words = caption.lower().replace(",", "").replace(".", "").split()
        
        # Extract garments
        garments = list(set([word for word in words if word in self.garment_vocab]))
        
        # Extract colors
        colors = list(set([word for word in words if word in self.color_vocab]))
        
        # Determine environment
        environment = "unknown"
        for env, keywords in self.env_vocab.items():
            if any(k in words for k in keywords):
                environment = env
                break
                
        # Determine style
        style = "other"
        for s, keywords in self.style_vocab.items():
            if any(k in words for k in keywords):
                style = s
                break
                
        return {
            "garments": garments,
            "colors": colors,
            "environment": environment,
            "style": style
        }

    # ------------------------------------------------------------------
    # Batch extraction 
    # ------------------------------------------------------------------

    def extract_all(self, captions: dict[str, str]) -> dict[str, dict]:
        """Extract attributes for every caption instantly."""
        # Load existing cache ------------------------------------------------
        all_attributes: dict[str, dict] = {}
        if ATTRIBUTES_CACHE.exists():
            try:
                with open(ATTRIBUTES_CACHE, "r", encoding="utf-8") as f:
                    all_attributes = json.load(f)
                print(f"[AttributeExtractor] Loaded {len(all_attributes)} cached attributes.")
            except (json.JSONDecodeError, IOError):
                pass

        # Determine pending items --------------------------------------------
        pending = {k: v for k, v in captions.items() if k not in all_attributes}
        print(f"[AttributeExtractor] {len(captions)} captions total, {len(all_attributes)} cached, {len(pending)} to process.")

        if not pending:
            return all_attributes

        for filename, caption in tqdm(pending.items(), desc="Extracting attributes (Offline)", unit="img"):
            all_attributes[filename] = self.extract_attributes(caption)

        # Save all at once since it's instant
        try:
            with open(ATTRIBUTES_CACHE, "w", encoding="utf-8") as f:
                json.dump(all_attributes, f, indent=2, ensure_ascii=False)
        except IOError as exc:
            print(f"[AttributeExtractor] Warning: could not save cache – {exc}")

        print(f"[AttributeExtractor] Extraction complete – {len(all_attributes)} total attribute sets.")
        return all_attributes
