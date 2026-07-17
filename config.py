"""
Central configuration for the DIC + VLM-as-Judge fashion search engine.

All model names, paths, hyperparameters, and API settings are defined here.
No other file should hardcode these values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "images"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
RESULTS_DIR = PROJECT_ROOT / "results"
CAPTIONS_CACHE = PROJECT_ROOT / "dataset" / "captions.json"
ATTRIBUTES_CACHE = PROJECT_ROOT / "dataset" / "attributes.json"

# Create directories if they don't exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET
# ============================================================================
DATASET_NAME = "detection-datasets/fashionpedia"
MAX_IMAGES = 1000  # Limit to 1000 images from Fashionpedia
FALLBACK_DATASET = "ashraq/fashion-product-images-small"

# ============================================================================
# OPENROUTER API (LLM + VLM)
# ============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model for text-only attribute extraction (free tier)
LLM_MODEL = "google/gemini-2.0-flash-exp:free"

# Model for VLM-as-Judge with vision (free tier, supports images)
VLM_MODEL = "google/gemini-2.0-flash-exp:free"

# API rate limiting
API_REQUESTS_PER_MINUTE = 15
API_RETRY_DELAY = 5  # seconds between retries on rate limit

# ============================================================================
# ML MODELS (Local, run on CPU or GPU)
# ============================================================================

# BLIP for image captioning
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

# CLIP for visual embeddings
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# SentenceTransformer for caption text embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# CHROMADB COLLECTIONS
# ============================================================================
VISUAL_COLLECTION_NAME = "visual_index"
TEXT_COLLECTION_NAME = "text_index"

# ============================================================================
# RETRIEVAL HYPERPARAMETERS
# ============================================================================

# Stage 2: Dual vector search fusion weights
CLIP_WEIGHT = 0.4       # Weight for CLIP visual similarity
TEXT_WEIGHT = 0.6       # Weight for text-to-text similarity

# Stage 3: Compositional re-ranking
STAGE2_TOP_K = 20       # Candidates passed from Stage 2 → Stage 3
COMPOSITIONAL_WEIGHT = 0.4  # Weight of compositional score in final ranking
VECTOR_WEIGHT = 0.6         # Weight of Stage 2 score in final ranking

# Stage 4: VLM-as-Judge
STAGE3_TOP_K = 10       # Candidates passed from Stage 3 → Stage 4

# Final output
FINAL_TOP_K = 5         # Number of results returned to user

# ============================================================================
# PROMPTS
# ============================================================================

ATTRIBUTE_EXTRACTION_PROMPT = """Extract structured fashion attributes from this image description.
Return ONLY valid JSON (no markdown, no explanation) with these keys:
- "garments": list of "color garment-type" strings (e.g., ["red blazer", "navy trousers"])
- "colors": list of colors mentioned (e.g., ["red", "navy"])
- "environment": one of "office", "street", "park", "home", "store", "outdoor", "indoor", "unknown"
- "style": one of "formal", "casual", "sporty", "ethnic", "streetwear", "other"

Description: "{caption}"
"""

QUERY_PARSING_PROMPT = """Extract structured fashion search attributes from this user query.
Return ONLY valid JSON (no markdown, no explanation) with these keys:
- "garments": list of "color garment-type" strings the user is looking for (e.g., ["red blazer", "blue pants"])
- "colors": list of colors mentioned (e.g., ["red", "blue"])
- "environment": one of "office", "street", "park", "home", "store", "outdoor", "indoor", null (if not specified)
- "style": one of "formal", "casual", "sporty", "ethnic", "streetwear", null (if not specified)

Query: "{query}"
"""

VLM_JUDGE_PROMPT = """You are a fashion search quality judge. 

A user searched for: "{query}"

Look at this image carefully. Score how well it matches the search query on a scale of 1-10.

Return ONLY valid JSON (no markdown, no explanation) with these keys:
- "score": integer 1-10
- "garment_match": brief description of garment match quality
- "color_match": brief description of color match quality  
- "environment_match": brief description of environment/setting match quality
- "reasoning": 1-2 sentence explanation of the overall match
"""
