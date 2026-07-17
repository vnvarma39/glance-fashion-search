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
    """Extracts structured fashion attributes from captions using an LLM.

    Attributes:
        client: OpenAI-compatible client pointed at OpenRouter.
    """

    def __init__(self) -> None:
        """Create the OpenAI client configured for OpenRouter."""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        print("[AttributeExtractor] Initialised with OpenRouter API.")

    # ------------------------------------------------------------------
    # Single caption → structured attributes
    # ------------------------------------------------------------------

    def extract_attributes(self, caption: str) -> dict:
        """Send a caption to the LLM and parse the JSON response.

        The method includes retry logic for rate-limit (HTTP 429) errors
        and gracefully handles malformed JSON by returning a default dict.

        Args:
            caption: Natural-language image caption.

        Returns:
            Dictionary with keys ``garments``, ``colors``, ``environment``,
            ``style``.
        """
        prompt = ATTRIBUTE_EXTRACTION_PROMPT.format(caption=caption)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                raw_text: str = response.choices[0].message.content.strip()

                # Strip markdown fences if the model wraps its output
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[-1]
                if raw_text.endswith("```"):
                    raw_text = raw_text.rsplit("```", 1)[0]
                raw_text = raw_text.strip()

                attributes: dict = json.loads(raw_text)

                # Ensure expected keys exist
                for key, default in _DEFAULT_ATTRIBUTES.items():
                    attributes.setdefault(key, default)

                return attributes

            except json.JSONDecodeError:
                print(
                    f"[AttributeExtractor] JSON parse error (attempt {attempt}/{max_retries})."
                )
                if attempt < max_retries:
                    time.sleep(API_RETRY_DELAY)

            except Exception as exc:
                error_msg = str(exc).lower()
                if "429" in error_msg or "rate" in error_msg:
                    print(
                        f"[AttributeExtractor] Rate limited – retrying in "
                        f"{API_RETRY_DELAY}s (attempt {attempt}/{max_retries})."
                    )
                    time.sleep(API_RETRY_DELAY)
                else:
                    print(f"[AttributeExtractor] API error: {exc}")
                    if attempt < max_retries:
                        time.sleep(API_RETRY_DELAY)

        print("[AttributeExtractor] Returning default attributes after retries exhausted.")
        return dict(_DEFAULT_ATTRIBUTES)

    # ------------------------------------------------------------------
    # Batch extraction with rate limiting + caching
    # ------------------------------------------------------------------

    def extract_all(self, captions: dict[str, str]) -> dict[str, dict]:
        """Extract attributes for every caption with rate limiting.

        Already-processed filenames (present in the JSON cache) are
        skipped automatically.

        Args:
            captions: Mapping ``{filename: caption}``.

        Returns:
            Mapping ``{filename: attributes_dict}`` for all filenames.
        """
        # Load existing cache ------------------------------------------------
        all_attributes: dict[str, dict] = {}
        if ATTRIBUTES_CACHE.exists():
            try:
                with open(ATTRIBUTES_CACHE, "r", encoding="utf-8") as f:
                    all_attributes = json.load(f)
                print(
                    f"[AttributeExtractor] Loaded {len(all_attributes)} cached attributes."
                )
            except (json.JSONDecodeError, IOError) as exc:
                print(f"[AttributeExtractor] Warning: could not load cache – {exc}")

        # Determine pending items --------------------------------------------
        pending = {k: v for k, v in captions.items() if k not in all_attributes}
        print(
            f"[AttributeExtractor] {len(captions)} captions total, "
            f"{len(all_attributes)} cached, {len(pending)} to process."
        )

        if not pending:
            return all_attributes

        # Rate-limit interval (seconds between requests)
        interval = 60.0 / API_REQUESTS_PER_MINUTE

        for filename, caption in tqdm(
            pending.items(), desc="Extracting attributes", unit="img"
        ):
            start_time = time.time()

            attributes = self.extract_attributes(caption)
            all_attributes[filename] = attributes

            # Incremental save
            try:
                with open(ATTRIBUTES_CACHE, "w", encoding="utf-8") as f:
                    json.dump(all_attributes, f, indent=2, ensure_ascii=False)
            except IOError as exc:
                print(f"[AttributeExtractor] Warning: could not save cache – {exc}")

            # Respect rate limit
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

        print(
            f"[AttributeExtractor] Extraction complete – "
            f"{len(all_attributes)} total attribute sets."
        )
        return all_attributes
