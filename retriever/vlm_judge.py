"""
VLM Judge — Stage 4 of the Fashion Search Pipeline.

Sends each candidate image (base64-encoded) alongside the user query to a
Vision-Language Model (VLM) hosted on OpenRouter.  The VLM acts as a *judge*,
returning a 1–10 relevance score with structured reasoning.  These scores are
blended with the Stage-3 reranked scores to produce the final ranking.
"""

import sys
import json
import time
import base64
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from tqdm import tqdm

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    VLM_MODEL,
    VLM_JUDGE_PROMPT,
    API_RETRY_DELAY,
    FINAL_TOP_K,
)


class VLMJudge:
    """Vision-Language Model judge for candidate relevance scoring.

    Each candidate image is sent to ``VLM_MODEL`` together with the user
    query.  The model returns a structured JSON response containing a
    1–10 score and match-quality descriptions.
    """

    def __init__(self) -> None:
        """Initialise the OpenAI client pointed at OpenRouter."""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = VLM_MODEL
        print(f"[VLMJudge] Initialised with model: {self.model}")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        """Read an image file and return its base64-encoded contents.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the image file.

        Returns
        -------
        str
            Base64-encoded string of the image bytes.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ------------------------------------------------------------------
    # Single-image judging
    # ------------------------------------------------------------------

    def judge_single(self, image_path: str, query: str) -> dict:
        """Judge a single candidate image against the query.

        Parameters
        ----------
        image_path : str
            Path to the candidate image.
        query : str
            The original user search query.

        Returns
        -------
        dict
            Keys: ``score`` (int 1–10), ``garment_match``, ``color_match``,
            ``environment_match``, ``reasoning``.  On failure, returns a
            default dict with ``score=5``.
        """
        default_result = {
            "score": 5,
            "garment_match": "unknown",
            "color_match": "unknown",
            "environment_match": "unknown",
            "reasoning": "VLM evaluation failed — default score assigned.",
        }

        try:
            b64_image = self._encode_image_base64(image_path)
            prompt_text = VLM_JUDGE_PROMPT.format(query=query)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.1,
            )

            raw_text = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
                raw_text = raw_text.rsplit("```", 1)[0]
                raw_text = raw_text.strip()

            parsed: dict = json.loads(raw_text)
            return {
                "score": int(parsed.get("score", 5)),
                "garment_match": parsed.get("garment_match", "unknown"),
                "color_match": parsed.get("color_match", "unknown"),
                "environment_match": parsed.get("environment_match", "unknown"),
                "reasoning": parsed.get("reasoning", "No reasoning provided."),
            }

        except json.JSONDecodeError as exc:
            print(f"[VLMJudge] JSON parse error for {image_path}: {exc}")
            return default_result
        except Exception as exc:  # noqa: BLE001
            print(f"[VLMJudge] Error judging {image_path}: {exc}")
            return default_result

    # ------------------------------------------------------------------
    # Batch judging
    # ------------------------------------------------------------------

    def judge_candidates(
        self,
        candidates: list[dict],
        query: str,
    ) -> list[dict]:
        """Judge all candidates and produce the final ranking.

        Parameters
        ----------
        candidates : list[dict]
            Stage-3 candidate list (must contain ``image_path`` and
            ``reranked_score``).
        query : str
            The original user search query.

        Returns
        -------
        list[dict]
            Top ``FINAL_TOP_K`` candidates sorted by ``final_score``
            (descending).  Each dict is augmented with ``vlm_score``,
            ``vlm_reasoning``, and ``final_score``.
        """
        print(f"[VLMJudge] Judging {len(candidates)} candidates …")

        for candidate in tqdm(candidates, desc="VLM Judging"):
            image_path = candidate.get("image_path", "")

            if not image_path or not Path(image_path).exists():
                print(f"[VLMJudge] Image not found: {image_path} — assigning default.")
                candidate["vlm_score"] = 0.5
                candidate["vlm_reasoning"] = "Image file not found."
                candidate["final_score"] = (
                    0.5 * candidate.get("reranked_score", 0.0) + 0.5 * 0.5
                )
                continue

            result = self.judge_single(image_path, query)

            # Normalise score to 0–1
            vlm_score = result["score"] / 10.0
            candidate["vlm_score"] = vlm_score
            candidate["vlm_reasoning"] = result["reasoning"]
            candidate["final_score"] = (
                0.5 * candidate.get("reranked_score", 0.0) + 0.5 * vlm_score
            )

            # Respect API rate limits
            time.sleep(API_RETRY_DELAY)

        # Sort by final score and return top-k
        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        top_candidates = candidates[: FINAL_TOP_K]

        print(
            f"[VLMJudge] Judging complete — returning top {len(top_candidates)} results."
        )
        return top_candidates
