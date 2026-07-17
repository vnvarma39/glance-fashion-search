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

    Uses the OpenRouter Chat Completions API (LLM_MODEL) with a
    system-level prompt that instructs the model to return **only** valid JSON
    with keys ``garments``, ``colors``, ``environment``, and ``style``.
    """

    def __init__(self) -> None:
        """Initialise the OpenAI client pointed at OpenRouter."""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = LLM_MODEL
        print(f"[QueryParser] Initialised with model: {self.model}")

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
            Number of retry attempts on transient / parse errors (default 3).

        Returns
        -------
        dict
            A dictionary with the following keys:

            * ``garments`` – list[str] of "color garment-type" descriptors
            * ``colors`` – list[str] of colours mentioned
            * ``environment`` – str | None  (e.g. "street", "office")
            * ``style`` – str | None  (e.g. "formal", "casual")
        """
        prompt = QUERY_PARSING_PROMPT.format(query=query)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful fashion search assistant. "
                                       "Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )

                raw_text = response.choices[0].message.content.strip()

                # The model sometimes wraps the JSON in markdown fences —
                # strip them if present.
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[-1]  # drop ```json line
                    raw_text = raw_text.rsplit("```", 1)[0]  # drop closing ```
                    raw_text = raw_text.strip()

                parsed: dict = json.loads(raw_text)

                # Normalise / guarantee keys
                result = {
                    "garments": parsed.get("garments", []),
                    "colors": parsed.get("colors", []),
                    "environment": parsed.get("environment"),
                    "style": parsed.get("style"),
                }
                print(f"[QueryParser] Parsed query → {result}")
                return result

            except json.JSONDecodeError as exc:
                print(
                    f"[QueryParser] JSON parse error on attempt {attempt}/{max_retries}: {exc}"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[QueryParser] API error on attempt {attempt}/{max_retries}: {exc}"
                )

            if attempt < max_retries:
                print(f"[QueryParser] Retrying in {API_RETRY_DELAY}s …")
                time.sleep(API_RETRY_DELAY)

        # Fallback: return a best-effort dict with the raw query as a
        # single garment entry so downstream stages can still function.
        print("[QueryParser] All retries exhausted — returning fallback parse.")
        return {
            "garments": [query],
            "colors": [],
            "environment": None,
            "style": None,
        }
