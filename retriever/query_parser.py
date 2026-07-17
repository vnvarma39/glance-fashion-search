"""
Query Parser Module — Stage 1 of the Fashion Search Pipeline.

Uses an OpenRouter-hosted LLM to parse a user's natural-language fashion query
into structured attributes (garments, colors, environment, style) that drive
metadata filtering and compositional re-ranking in later stages.
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is on sys.path so `config` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    QUERY_PARSING_PROMPT,
    API_RETRY_DELAY,
)


class QueryParser:
    """Parses natural-language fashion queries into structured attribute dicts.
    
    This completely bypasses API rate limits and runs instantly offline
    using keyword matching, perfectly mirroring the indexer's offline metadata.
    """

    def __init__(self) -> None:
        """Initialize the keyword vocabularies."""
        print("[QueryParser] Initialised with Offline Keyword Matcher (No API limits!).")
        
        self.garment_vocab = [
            "shirt", "t-shirt", "pants", "jeans", "dress", "skirt", "jacket", 
            "coat", "sweater", "hoodie", "suit", "blazer", "shorts", "tie", 
            "hat", "shoes", "sneakers", "boots", "glasses", "sunglasses", "bag",
            "raincoat", "blouse"
        ]
        
        self.color_vocab = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey", 
            "brown", "orange", "pink", "purple", "beige", "navy", "maroon"
        ]
        
        self.env_vocab = {
            "indoor": ["indoor", "room", "inside", "office", "home", "building", "studio", "runway"],
            "outdoor": ["outdoor", "outside", "nature", "park", "street", "city", "urban", "road"],
            "urban": ["urban", "city", "street", "building"],
            "nature": ["nature", "park", "tree", "grass", "forest"]
        }
        
        self.style_vocab = {
            "formal": ["formal", "suit", "tie", "blazer", "business", "office", "professional"],
            "casual": ["casual", "t-shirt", "jeans", "hoodie", "sneakers", "relaxed", "weekend"],
            "winter": ["winter", "coat", "jacket", "snow", "cold", "sweater"],
            "summer": ["summer", "shorts", "sunglasses", "beach", "warm"]
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_query(self, query: str, max_retries: int = 3) -> dict:
        """Parse a user query into structured fashion attributes.

        Parameters
        ----------
        query : str
            The raw user search query (e.g. "red blazer with blue jeans in
            a street setting").
        max_retries : int, optional
            Ignored in offline mode.

        Returns
        -------
        dict
            A dictionary with the keys garments, colors, environment, style.
        """
        words = query.lower().replace(",", "").replace(".", "").split()
        
        # Extract garments (keep context if color is nearby, but simpler here: just keywords)
        garments = list(set([word for word in words if word in self.garment_vocab]))
        
        # If no specific garments found, use the whole query to allow vector search to do the heavy lifting
        if not garments:
            garments = [query]
            
        # Extract colors
        colors = list(set([word for word in words if word in self.color_vocab]))
        
        # Determine environment
        environment = None
        for env, keywords in self.env_vocab.items():
            if any(k in words for k in keywords):
                environment = env
                break
                
        # Determine style
        style = None
        for s, keywords in self.style_vocab.items():
            if any(k in words for k in keywords):
                style = s
                break
                
        result = {
            "garments": garments,
            "colors": colors,
            "environment": environment,
            "style": style,
        }
        
        print(f"[QueryParser] Parsed query (Offline) → {result}")
        return result
